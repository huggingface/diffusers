# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Union

import safetensors
import torch
from huggingface_hub.utils import EntryNotFoundError

from ..utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    WEIGHTS_INDEX_NAME,
    _add_variant,
    _get_model_file,
    is_accelerate_available,
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
    from accelerate.utils import get_balanced_memory, get_max_memory, set_module_tensor_to_device


# Adapted from `transformers` (see modeling_utils.py)
def _determine_device_map(model: torch.nn.Module, device_map, max_memory, torch_dtype):
    if isinstance(device_map, str):
        no_split_modules = model._get_no_split_modules(device_map)
        device_map_kwargs = {"no_split_module_classes": no_split_modules}

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

        device_map_kwargs["max_memory"] = max_memory
        device_map = infer_auto_device_map(model, dtype=torch_dtype, **device_map_kwargs)

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


def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safetensors.torch.load_file(checkpoint_file, device="cpu")
        else:
            weights_only_kwarg = {"weights_only": True} if is_torch_version(">=", "1.13") else {}
            return torch.load(
                checkpoint_file,
                map_location="cpu",
                **weights_only_kwarg,
            )
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
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    model_name_or_path: Optional[str] = None,
) -> List[str]:
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())

    unexpected_keys = []
    empty_state_dict = model.state_dict()
    for param_name, param in state_dict.items():
        if param_name not in empty_state_dict:
            unexpected_keys.append(param_name)
            continue

        if empty_state_dict[param_name].shape != param.shape:
            model_name_or_path_str = f"{model_name_or_path} " if model_name_or_path is not None else ""
            raise ValueError(
                f"Cannot load {model_name_or_path_str}because {param_name} expected shape {empty_state_dict[param_name]}, but got {param.shape}. If you want to instead overwrite randomly initialized weights, please make sure to pass both `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example."
            )

        if accepts_dtype:
            set_module_tensor_to_device(model, param_name, device, value=param, dtype=dtype)
        else:
            set_module_tensor_to_device(model, param_name, device, value=param)
    return unexpected_keys


def _load_state_dict_into_model(model_to_load, state_dict: OrderedDict) -> List[str]:
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix: str = ""):
        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)

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
            )
            index_file = Path(index_file)
        except (EntryNotFoundError, EnvironmentError):
            index_file = None

    return index_file
