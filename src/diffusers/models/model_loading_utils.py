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

import importlib
import inspect
import os
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union
from zipfile import is_zipfile

import safetensors
import torch
from huggingface_hub import DDUFEntry
from huggingface_hub.utils import EntryNotFoundError

from ..quantizers import DiffusersQuantizer
from ..utils import (
    GGUF_FILE_EXTENSION,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    WEIGHTS_INDEX_NAME,
    _add_variant,
    _get_model_file,
    deprecate,
    is_accelerate_available,
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

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(device_map=device_map)

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
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
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
            else:
                param = param.to(dtype)
                set_module_kwargs["dtype"] = dtype

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
            hf_quantizer.create_quantized_param(model, param, param_name, param_device, state_dict, unexpected_keys)
        else:
            set_module_tensor_to_device(model, param_name, param_device, value=param, **set_module_kwargs)

    return offload_index, state_dict_index


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
