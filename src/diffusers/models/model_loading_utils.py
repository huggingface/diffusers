# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import importlib
import inspect
import os
from array import array
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union
from zipfile import is_zipfile

import safetensors
import torch
from huggingface_hub import DDUFEntry
from huggingface_hub.utils import EntryNotFoundError

from ..quantizers import DiffusersQuantizer
from ..utils import (
    DEFAULT_HF_PARALLEL_LOADING_WORKERS,
    GGUF_FILE_EXTENSION,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    WEIGHTS_INDEX_NAME,
    _add_variant,
    _get_model_file,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    is_gguf_available,
    is_torch_available,
    is_torch_version,
    logging,
)


logger = logging.get_logger(__name__)

_CLASS_REMAPPING_DICT = {
    "Transformer2DModel": {
        "ada_norm_zero": "DiTTransformer2DModel",
        "ada_norm_single": "PixArtTransformer2DModel",
    }
}


if is_accelerate_available():
    from accelerate import infer_auto_device_map
    from accelerate.utils import get_balanced_memory, get_max_memory, offload_weight, set_module_tensor_to_device


# Adapted from `transformers` (see modeling_utils.py)
def _determine_device_map(
    model: torch.nn.Module, device_map, max_memory, torch_dtype, keep_in_fp32_modules=[], hf_quantizer=None
):
    if isinstance(device_map, str):
        special_dtypes = {}
        if hf_quantizer is not None:
            special_dtypes.update(hf_quantizer.get_special_dtypes_update(model, torch_dtype))
        special_dtypes.update(
            {
                name: torch.float32
                for name, _ in model.named_parameters()
                if any(m in name for m in keep_in_fp32_modules)
            }
        )

        target_dtype = torch_dtype
        if hf_quantizer is not None:
            target_dtype = hf_quantizer.adjust_target_dtype(target_dtype)

        no_split_modules = model._get_no_split_modules(device_map)
        device_map_kwargs = {"no_split_module_classes": no_split_modules}

        if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
            device_map_kwargs["special_dtypes"] = special_dtypes
        elif len(special_dtypes) > 0:
            logger.warning(
                "This model has some weights that should be kept in higher precision, you need to upgrade "
                "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
            )

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                dtype=torch_dtype,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **device_map_kwargs,
            )
        else:
            max_memory = get_max_memory(max_memory)

        if hf_quantizer is not None:
            max_memory = hf_quantizer.adjust_max_memory(max_memory)

        device_map_kwargs["max_memory"] = max_memory
        device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)

    return device_map


def _fetch_remapped_cls_from_config(config, old_class):
    previous_class_name = old_class.__name__
    remapped_class_name = _CLASS_REMAPPING_DICT.get(previous_class_name).get(config["norm_type"], None)

    # Details:
    # https://github.com/huggingface/diffusers/pull/7647#discussion_r1621344818
    if remapped_class_name:
        # load diffusers library to import compatible and original scheduler
        diffusers_library = importlib.import_module(__name__.split(".")[0])
        remapped_class = getattr(diffusers_library, remapped_class_name)
        logger.info(
            f"Changing class object to be of `{remapped_class_name}` type from `{previous_class_name}` type."
            f"This is because `{previous_class_name}` is scheduled to be deprecated in a future version. Note that this"
            " DOESN'T affect the final results."
        )
        return remapped_class
    else:
        return old_class


def _determine_param_device(param_name: str, device_map: Optional[Dict[str, Union[int, str, torch.device]]]):
    """
    Find the device of param_name from the device_map.
    """
    if device_map is None:
        return "cpu"
    else:
        module_name = param_name
        # find next higher level module that is defined in device_map:
        # bert.lm_head.weight -> bert.lm_head -> bert -> ''
        while len(module_name) > 0 and module_name not in device_map:
            module_name = ".".join(module_name.split(".")[:-1])
        if module_name == "" and "" not in device_map:
            raise ValueError(f"{param_name} doesn't have any device set.")
        return device_map[module_name]


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    dduf_entries: Optional[Dict[str, DDUFEntry]] = None,
    disable_mmap: bool = False,
    map_location: Union[str, torch.device] = "cpu",
):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    # TODO: maybe refactor a bit this part where we pass a dict here
    if isinstance(checkpoint_file, dict):
        return checkpoint_file
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            if dduf_entries:
                # tensors are loaded on cpu
                with dduf_entries[checkpoint_file].as_mmap() as mm:
                    return safetensors.torch.load(mm)
            if disable_mmap:
                return safetensors.torch.load(open(checkpoint_file, "rb").read())
            else:
                return safetensors.torch.load_file(checkpoint_file, device=map_location)
        elif file_extension == GGUF_FILE_EXTENSION:
            return load_gguf_checkpoint(checkpoint_file)
        else:
            extra_args = {}
            weights_only_kwarg = {"weights_only": True} if is_torch_version(">=", "1.13") else {}
            # mmap can only be used with files serialized with zipfile-based format.
            if (
                isinstance(checkpoint_file, str)
                and map_location != "meta"
                and is_torch_version(">=", "2.1.0")
                and is_zipfile(checkpoint_file)
                and not disable_mmap
            ):
                extra_args = {"mmap": True}
            return torch.load(checkpoint_file, map_location=map_location, **weights_only_kwarg, **extra_args)
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' at '{checkpoint_file}'. "
            )


def load_model_dict_into_meta(
    model,
    state_dict: OrderedDict,
    dtype: Optional[Union[str, torch.dtype]] = None,
    model_name_or_path: Optional[str] = None,
    hf_quantizer: Optional[DiffusersQuantizer] = None,
    keep_in_fp32_modules: Optional[List] = None,
    device_map: Optional[Dict[str, Union[int, str, torch.device]]] = None,
    unexpected_keys: Optional[List[str]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    offload_index: Optional[Dict] = None,
    state_dict_index: Optional[Dict] = None,
    state_dict_folder: Optional[Union[str, os.PathLike]] = None,
) -> List[str]:
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`
    """

    is_quantized = hf_quantizer is not None
    empty_state_dict = model.state_dict()

    for param_name, param in state_dict.items():
        if param_name not in empty_state_dict:
            continue

        set_module_kwargs = {}
        # We convert floating dtypes to the `dtype` passed. We also want to keep the buffers/params
        # in int/uint/bool and not cast them.
        # TODO: revisit cases when param.dtype == torch.float8_e4m3fn
        if dtype is not None and torch.is_floating_point(param):
            if keep_in_fp32_modules is not None and any(
                module_to_keep_in_fp32 in param_name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
            ):
                param = param.to(torch.float32)
                set_module_kwargs["dtype"] = torch.float32
            # For quantizers have save weights using torch.float8_e4m3fn
            elif hf_quantizer is not None and param.dtype == getattr(torch, "float8_e4m3fn", None):
                pass
            else:
                param = param.to(dtype)
                set_module_kwargs["dtype"] = dtype

        if is_accelerate_version(">", "1.8.1"):
            set_module_kwargs["non_blocking"] = True
            set_module_kwargs["clear_cache"] = False

        # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model, and which
        # uses `param.copy_(input_param)` that preserves the contiguity of the parameter in the model.
        # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29
        old_param = model
        splits = param_name.split(".")
        for split in splits:
            old_param = getattr(old_param, split)

        if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
            old_param = None

        if old_param is not None:
            if dtype is None:
                param = param.to(old_param.dtype)

            if old_param.is_contiguous():
                param = param.contiguous()

        param_device = _determine_param_device(param_name, device_map)

        # bnb params are flattened.
        # gguf quants have a different shape based on the type of quantization applied
        if empty_state_dict[param_name].shape != param.shape:
            if (
                is_quantized
                and hf_quantizer.pre_quantized
                and hf_quantizer.check_if_quantized_param(
                    model, param, param_name, state_dict, param_device=param_device
                )
            ):
                hf_quantizer.check_quantized_param_shape(param_name, empty_state_dict[param_name], param)
            else:
                model_name_or_path_str = f"{model_name_or_path} " if model_name_or_path is not None else ""
                raise ValueError(
                    f"Cannot load {model_name_or_path_str} because {param_name} expected shape {empty_state_dict[param_name].shape}, but got {param.shape}. If you want to instead overwrite randomly initialized weights, please make sure to pass both `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example."
                )
        if param_device == "disk":
            offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == "cpu" and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        elif is_quantized and (
            hf_quantizer.check_if_quantized_param(model, param, param_name, state_dict, param_device=param_device)
        ):
            hf_quantizer.create_quantized_param(
                model, param, param_name, param_device, state_dict, unexpected_keys, dtype=dtype
            )
        else:
            set_module_tensor_to_device(model, param_name, param_device, value=param, **set_module_kwargs)

    return offload_index, state_dict_index


def check_support_param_buffer_assignment(model_to_load, state_dict, start_prefix=""):
    """
    Checks if `model_to_load` supports param buffer assignment (such as when loading in empty weights) by first
    checking if the model explicitly disables it, then by ensuring that the state dict keys are a subset of the model's
    parameters.

    """
    if model_to_load.device.type == "meta":
        return False

    if len([key for key in state_dict if key.startswith(start_prefix)]) == 0:
        return False

    # Some models explicitly do not support param buffer assignment
    if not getattr(model_to_load, "_supports_param_buffer_assignment", True):
        logger.debug(
            f"{model_to_load.__class__.__name__} does not support param buffer assignment, loading will be slower"
        )
        return False

    # If the model does, the incoming `state_dict` and the `model_to_load` must be the same dtype
    first_key = next(iter(model_to_load.state_dict().keys()))
    if start_prefix + first_key in state_dict:
        return state_dict[start_prefix + first_key].dtype == model_to_load.state_dict()[first_key].dtype

    return False


def _load_shard_file(
    shard_file,
    model,
    model_state_dict,
    device_map=None,
    dtype=None,
    hf_quantizer=None,
    keep_in_fp32_modules=None,
    dduf_entries=None,
    loaded_keys=None,
    unexpected_keys=None,
    offload_index=None,
    offload_folder=None,
    state_dict_index=None,
    state_dict_folder=None,
    ignore_mismatched_sizes=False,
    low_cpu_mem_usage=False,
):
    state_dict = load_state_dict(shard_file, dduf_entries=dduf_entries)
    mismatched_keys = _find_mismatched_keys(
        state_dict,
        model_state_dict,
        loaded_keys,
        ignore_mismatched_sizes,
    )
    error_msgs = []
    if low_cpu_mem_usage:
        offload_index, state_dict_index = load_model_dict_into_meta(
            model,
            state_dict,
            device_map=device_map,
            dtype=dtype,
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=keep_in_fp32_modules,
            unexpected_keys=unexpected_keys,
            offload_folder=offload_folder,
            offload_index=offload_index,
            state_dict_index=state_dict_index,
            state_dict_folder=state_dict_folder,
        )
    else:
        assign_to_params_buffers = check_support_param_buffer_assignment(model, state_dict)

        error_msgs += _load_state_dict_into_model(model, state_dict, assign_to_params_buffers)
    return offload_index, state_dict_index, mismatched_keys, error_msgs


def _load_shard_files_with_threadpool(
    shard_files,
    model,
    model_state_dict,
    device_map=None,
    dtype=None,
    hf_quantizer=None,
    keep_in_fp32_modules=None,
    dduf_entries=None,
    loaded_keys=None,
    unexpected_keys=None,
    offload_index=None,
    offload_folder=None,
    state_dict_index=None,
    state_dict_folder=None,
    ignore_mismatched_sizes=False,
    low_cpu_mem_usage=False,
):
    # Do not spawn anymore workers than you need
    num_workers = min(len(shard_files), DEFAULT_HF_PARALLEL_LOADING_WORKERS)

    logger.info(f"Loading model weights in parallel with {num_workers} workers...")

    error_msgs = []
    mismatched_keys = []

    load_one = functools.partial(
        _load_shard_file,
        model=model,
        model_state_dict=model_state_dict,
        device_map=device_map,
        dtype=dtype,
        hf_quantizer=hf_quantizer,
        keep_in_fp32_modules=keep_in_fp32_modules,
        dduf_entries=dduf_entries,
        loaded_keys=loaded_keys,
        unexpected_keys=unexpected_keys,
        offload_index=offload_index,
        offload_folder=offload_folder,
        state_dict_index=state_dict_index,
        state_dict_folder=state_dict_folder,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with logging.tqdm(total=len(shard_files), desc="Loading checkpoint shards") as pbar:
            futures = [executor.submit(load_one, shard_file) for shard_file in shard_files]
            for future in as_completed(futures):
                result = future.result()
                offload_index, state_dict_index, _mismatched_keys, _error_msgs = result
                error_msgs += _error_msgs
                mismatched_keys += _mismatched_keys
                pbar.update(1)

    return offload_index, state_dict_index, mismatched_keys, error_msgs


def _find_mismatched_keys(
    state_dict,
    model_state_dict,
    loaded_keys,
    ignore_mismatched_sizes,
):
    mismatched_keys = []
    if ignore_mismatched_sizes:
        for checkpoint_key in loaded_keys:
            model_key = checkpoint_key
            # If the checkpoint is sharded, we may not have the key here.
            if checkpoint_key not in state_dict:
                continue

            if model_key in model_state_dict and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape:
                mismatched_keys.append(
                    (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                )
                del state_dict[checkpoint_key]
    return mismatched_keys


def _load_state_dict_into_model(
    model_to_load, state_dict: OrderedDict, assign_to_params_buffers: bool = False
) -> List[str]:
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix: str = "", assign_to_params_buffers: bool = False):
        local_metadata = {}
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers
        if assign_to_params_buffers and not is_torch_version(">=", "2.1"):
            logger.info("You need to have torch>=2.1 in order to load the model with assign_to_params_buffers=True")
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".", assign_to_params_buffers)

    load(model_to_load, assign_to_params_buffers=assign_to_params_buffers)

    return error_msgs


def _fetch_index_file(
    is_local,
    pretrained_model_name_or_path,
    subfolder,
    use_safetensors,
    cache_dir,
    variant,
    force_download,
    proxies,
    local_files_only,
    token,
    revision,
    user_agent,
    commit_hash,
    dduf_entries: Optional[Dict[str, DDUFEntry]] = None,
):
    if is_local:
        index_file = Path(
            pretrained_model_name_or_path,
            subfolder or "",
            _add_variant(SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        )
    else:
        index_file_in_repo = Path(
            subfolder or "",
            _add_variant(SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        ).as_posix()
        try:
            index_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=index_file_in_repo,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=None,
                user_agent=user_agent,
                commit_hash=commit_hash,
                dduf_entries=dduf_entries,
            )
            if not dduf_entries:
                index_file = Path(index_file)
        except (EntryNotFoundError, EnvironmentError):
            index_file = None

    return index_file


def _fetch_index_file_legacy(
    is_local,
    pretrained_model_name_or_path,
    subfolder,
    use_safetensors,
    cache_dir,
    variant,
    force_download,
    proxies,
    local_files_only,
    token,
    revision,
    user_agent,
    commit_hash,
    dduf_entries: Optional[Dict[str, DDUFEntry]] = None,
):
    if is_local:
        index_file = Path(
            pretrained_model_name_or_path,
            subfolder or "",
            SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME,
        ).as_posix()
        splits = index_file.split(".")
        split_index = -3 if ".cache" in index_file else -2
        splits = splits[:-split_index] + [variant] + splits[-split_index:]
        index_file = ".".join(splits)
        if os.path.exists(index_file):
            deprecation_message = f"This serialization format is now deprecated to standardize the serialization format between `transformers` and `diffusers`. We recommend you to remove the existing files associated with the current variant ({variant}) and re-obtain them by running a `save_pretrained()`."
            deprecate("legacy_sharded_ckpts_with_variant", "1.0.0", deprecation_message, standard_warn=False)
            index_file = Path(index_file)
        else:
            index_file = None
    else:
        if variant is not None:
            index_file_in_repo = Path(
                subfolder or "",
                SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME,
            ).as_posix()
            splits = index_file_in_repo.split(".")
            split_index = -2
            splits = splits[:-split_index] + [variant] + splits[-split_index:]
            index_file_in_repo = ".".join(splits)
            try:
                index_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=index_file_in_repo,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=None,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                    dduf_entries=dduf_entries,
                )
                index_file = Path(index_file)
                deprecation_message = f"This serialization format is now deprecated to standardize the serialization format between `transformers` and `diffusers`. We recommend you to remove the existing files associated with the current variant ({variant}) and re-obtain them by running a `save_pretrained()`."
                deprecate("legacy_sharded_ckpts_with_variant", "1.0.0", deprecation_message, standard_warn=False)
            except (EntryNotFoundError, EnvironmentError):
                index_file = None

    return index_file


def _gguf_parse_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        if data_type[0] != 9:
            raise ValueError("Received multiple types, therefore expected the first type to indicate an array.")
        data_type, array_data_type = data_type

    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type in [9]:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed tokenizer and config
    attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster and only loads the
            metadata in memory.
    """

    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader

        from ..quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)

    parsed_parameters = {}
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data.copy())
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights

    return parsed_parameters


def _find_mismatched_keys(state_dict, model_state_dict, loaded_keys, ignore_mismatched_sizes):
    mismatched_keys = []
    if not ignore_mismatched_sizes:
        return mismatched_keys
    for checkpoint_key in loaded_keys:
        model_key = checkpoint_key
        # If the checkpoint is sharded, we may not have the key here.
        if checkpoint_key not in state_dict:
            continue

        if model_key in model_state_dict and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape:
            mismatched_keys.append(
                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
            )
            del state_dict[checkpoint_key]
    return mismatched_keys


def _expand_device_map(device_map, param_names):
    """
    Expand a device map to return the correspondence parameter name to device.
    """
    new_device_map = {}
    for module, device in device_map.items():
        new_device_map.update(
            {p: device for p in param_names if p == module or p.startswith(f"{module}.") or module == ""}
        )
    return new_device_map


# Adapted from: https://github.com/huggingface/transformers/blob/0687d481e2c71544501ef9cb3eef795a6e79b1de/src/transformers/modeling_utils.py#L5859
def _caching_allocator_warmup(
    model, expanded_device_map: Dict[str, torch.device], dtype: torch.dtype, hf_quantizer: Optional[DiffusersQuantizer]
) -> None:
    """
    This function warm-ups the caching allocator based on the size of the model tensors that will reside on each
    device. It allows to have one large call to Malloc, instead of recursively calling it later when loading the model,
    which is actually the loading speed bottleneck. Calling this function allows to cut the model loading time by a
    very large margin.
    """
    factor = 2 if hf_quantizer is None else hf_quantizer.get_cuda_warm_up_factor()

    # Keep only accelerator devices
    accelerator_device_map = {
        param: torch.device(device)
        for param, device in expanded_device_map.items()
        if str(device) not in ["cpu", "disk"]
    }
    if not accelerator_device_map:
        return

    elements_per_device = defaultdict(int)
    for param_name, device in accelerator_device_map.items():
        try:
            p = model.get_parameter(param_name)
        except AttributeError:
            try:
                p = model.get_buffer(param_name)
            except AttributeError:
                raise AttributeError(f"Parameter or buffer with name={param_name} not found in model")
        # TODO: account for TP when needed.
        elements_per_device[device] += p.numel()

    # This will kick off the caching allocator to avoid having to Malloc afterwards
    for device, elem_count in elements_per_device.items():
        warmup_elems = max(1, elem_count // factor)
        _ = torch.empty(warmup_elems, dtype=dtype, device=device, requires_grad=False)
