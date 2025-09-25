# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import copy
import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import model_info
from huggingface_hub.constants import HF_HUB_OFFLINE

from ..models.modeling_utils import ModelMixin, load_state_dict
from ..utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    delete_adapter_layers,
    deprecate,
    get_adapter_name,
    is_accelerate_available,
    is_peft_available,
    is_peft_version,
    is_transformers_available,
    is_transformers_version,
    logging,
    recurse_remove_peft_layers,
    scale_lora_layers,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from ..utils.peft_utils import _create_lora_config
from ..utils.state_dict_utils import _load_sft_state_dict_metadata


if is_transformers_available():
    from transformers import PreTrainedModel

if is_peft_available():
    from peft.tuners.tuners_utils import BaseTunerLayer

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"
LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"


def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False, adapter_names=None):
    """
    Fuses LoRAs for the text encoder.

    Args:
        text_encoder (`torch.nn.Module`):
            The text encoder module to set the adapter layers for. If `None`, it will try to get the `text_encoder`
            attribute.
        lora_scale (`float`, defaults to 1.0):
            Controls how much to influence the outputs with the LoRA parameters.
        safe_fusing (`bool`, defaults to `False`):
            Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
        adapter_names (`List[str]` or `str`):
            The names of the adapters to use.
    """
    merge_kwargs = {"safe_merge": safe_fusing}

    for module in text_encoder.modules():
        if isinstance(module, BaseTunerLayer):
            if lora_scale != 1.0:
                module.scale_layer(lora_scale)

            # For BC with previous PEFT versions, we need to check the signature
            # of the `merge` method to see if it supports the `adapter_names` argument.
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. "
                    "Please upgrade to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)


def unfuse_text_encoder_lora(text_encoder):
    """
    Unfuses LoRAs for the text encoder.

    Args:
        text_encoder (`torch.nn.Module`):
            The text encoder module to set the adapter layers for. If `None`, it will try to get the `text_encoder`
            attribute.
    """
    for module in text_encoder.modules():
        if isinstance(module, BaseTunerLayer):
            module.unmerge()


def set_adapters_for_text_encoder(
    adapter_names: Union[List[str], str],
    text_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
    text_encoder_weights: Optional[Union[float, List[float], List[None]]] = None,
):
    """
    Sets the adapter layers for the text encoder.

    Args:
        adapter_names (`List[str]` or `str`):
            The names of the adapters to use.
        text_encoder (`torch.nn.Module`, *optional*):
            The text encoder module to set the adapter layers for. If `None`, it will try to get the `text_encoder`
            attribute.
        text_encoder_weights (`List[float]`, *optional*):
            The weights to use for the text encoder. If `None`, the weights are set to `1.0` for all the adapters.
    """
    if text_encoder is None:
        raise ValueError(
            "The pipeline does not have a default `pipe.text_encoder` class. Please make sure to pass a `text_encoder` instead."
        )

    def process_weights(adapter_names, weights):
        # Expand weights into a list, one entry per adapter
        # e.g. for 2 adapters:  7 -> [7,7] ; [3, None] -> [3, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}"
            )

        # Set None values to default of 1.0
        # e.g. [7,7] -> [7,7] ; [3, None] -> [3,1]
        weights = [w if w is not None else 1.0 for w in weights]

        return weights

    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
    text_encoder_weights = process_weights(adapter_names, text_encoder_weights)
    set_weights_and_activate_adapters(text_encoder, adapter_names, text_encoder_weights)


def disable_lora_for_text_encoder(text_encoder: Optional["PreTrainedModel"] = None):
    """
    Disables the LoRA layers for the text encoder.

    Args:
        text_encoder (`torch.nn.Module`, *optional*):
            The text encoder module to disable the LoRA layers for. If `None`, it will try to get the `text_encoder`
            attribute.
    """
    if text_encoder is None:
        raise ValueError("Text Encoder not found.")
    set_adapter_layers(text_encoder, enabled=False)


def enable_lora_for_text_encoder(text_encoder: Optional["PreTrainedModel"] = None):
    """
    Enables the LoRA layers for the text encoder.

    Args:
        text_encoder (`torch.nn.Module`, *optional*):
            The text encoder module to enable the LoRA layers for. If `None`, it will try to get the `text_encoder`
            attribute.
    """
    if text_encoder is None:
        raise ValueError("Text Encoder not found.")
    set_adapter_layers(text_encoder, enabled=True)


def _remove_text_encoder_monkey_patch(text_encoder):
    recurse_remove_peft_layers(text_encoder)
    if getattr(text_encoder, "peft_config", None) is not None:
        del text_encoder.peft_config
        text_encoder._hf_peft_config_loaded = None


def _fetch_state_dict(
    pretrained_model_name_or_path_or_dict,
    weight_name,
    use_safetensors,
    local_files_only,
    cache_dir,
    force_download,
    proxies,
    token,
    revision,
    subfolder,
    user_agent,
    allow_pickle,
    metadata=None,
):
    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        # Let's first try to load .safetensors weights
        if (use_safetensors and weight_name is None) or (
            weight_name is not None and weight_name.endswith(".safetensors")
        ):
            try:
                # Here we're relaxing the loading check to enable more Inference API
                # friendliness where sometimes, it's not at all possible to automatically
                # determine `weight_name`.
                if weight_name is None:
                    weight_name = _best_guess_weight_name(
                        pretrained_model_name_or_path_or_dict,
                        file_extension=".safetensors",
                        local_files_only=local_files_only,
                    )
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = safetensors.torch.load_file(model_file, device="cpu")
                metadata = _load_sft_state_dict_metadata(model_file)

            except (IOError, safetensors.SafetensorError) as e:
                if not allow_pickle:
                    raise e
                # try loading non-safetensors weights
                model_file = None
                metadata = None
                pass

        if model_file is None:
            if weight_name is None:
                weight_name = _best_guess_weight_name(
                    pretrained_model_name_or_path_or_dict, file_extension=".bin", local_files_only=local_files_only
                )
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = load_state_dict(model_file)
            metadata = None
    else:
        state_dict = pretrained_model_name_or_path_or_dict

    return state_dict, metadata


def _best_guess_weight_name(
    pretrained_model_name_or_path_or_dict, file_extension=".safetensors", local_files_only=False
):
    if local_files_only or HF_HUB_OFFLINE:
        raise ValueError("When using the offline mode, you must specify a `weight_name`.")

    targeted_files = []

    if os.path.isfile(pretrained_model_name_or_path_or_dict):
        return
    elif os.path.isdir(pretrained_model_name_or_path_or_dict):
        targeted_files = [f for f in os.listdir(pretrained_model_name_or_path_or_dict) if f.endswith(file_extension)]
    else:
        files_in_repo = model_info(pretrained_model_name_or_path_or_dict).siblings
        targeted_files = [f.rfilename for f in files_in_repo if f.rfilename.endswith(file_extension)]
    if len(targeted_files) == 0:
        return

    # "scheduler" does not correspond to a LoRA checkpoint.
    # "optimizer" does not correspond to a LoRA checkpoint
    # only top-level checkpoints are considered and not the other ones, hence "checkpoint".
    unallowed_substrings = {"scheduler", "optimizer", "checkpoint"}
    targeted_files = list(
        filter(lambda x: all(substring not in x for substring in unallowed_substrings), targeted_files)
    )

    if any(f.endswith(LORA_WEIGHT_NAME) for f in targeted_files):
        targeted_files = list(filter(lambda x: x.endswith(LORA_WEIGHT_NAME), targeted_files))
    elif any(f.endswith(LORA_WEIGHT_NAME_SAFE) for f in targeted_files):
        targeted_files = list(filter(lambda x: x.endswith(LORA_WEIGHT_NAME_SAFE), targeted_files))

    if len(targeted_files) > 1:
        logger.warning(
            f"Provided path contains more than one weights file in the {file_extension} format. `{targeted_files[0]}` is going to be loaded, for precise control, specify a `weight_name` in `load_lora_weights`."
        )
    weight_name = targeted_files[0]
    return weight_name


def _pack_dict_with_prefix(state_dict, prefix):
    sd_with_prefix = {f"{prefix}.{key}": value for key, value in state_dict.items()}
    return sd_with_prefix


def _load_lora_into_text_encoder(
    state_dict,
    network_alphas,
    text_encoder,
    prefix=None,
    lora_scale=1.0,
    text_encoder_name="text_encoder",
    adapter_name=None,
    _pipeline=None,
    low_cpu_mem_usage=False,
    hotswap: bool = False,
    metadata=None,
):
    from ..hooks.group_offloading import _maybe_remove_and_reapply_group_offloading

    if not USE_PEFT_BACKEND:
        raise ValueError("PEFT backend is required for this method.")

    if network_alphas and metadata:
        raise ValueError("`network_alphas` and `metadata` cannot be specified both at the same time.")

    peft_kwargs = {}
    if low_cpu_mem_usage:
        if not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )
        if not is_transformers_version(">", "4.45.2"):
            # Note from sayakpaul: It's not in `transformers` stable yet.
            # https://github.com/huggingface/transformers/pull/33725/
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `transformers` version. Please update it with `pip install -U transformers`."
            )
        peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
    # then the `state_dict` keys should have `unet_name` and/or `text_encoder_name` as
    # their prefixes.
    prefix = text_encoder_name if prefix is None else prefix

    # Safe prefix to check with.
    if hotswap and any(text_encoder_name in key for key in state_dict.keys()):
        raise ValueError("At the moment, hotswapping is not supported for text encoders, please pass `hotswap=False`.")

    # Load the layers corresponding to text encoder and make necessary adjustments.
    if prefix is not None:
        state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
        if metadata is not None:
            metadata = {k.removeprefix(f"{prefix}."): v for k, v in metadata.items() if k.startswith(f"{prefix}.")}

    if len(state_dict) > 0:
        logger.info(f"Loading {prefix}.")
        rank = {}
        state_dict = convert_state_dict_to_diffusers(state_dict)

        # convert state dict
        state_dict = convert_state_dict_to_peft(state_dict)

        for name, _ in text_encoder.named_modules():
            if name.endswith((".q_proj", ".k_proj", ".v_proj", ".out_proj", ".fc1", ".fc2")):
                rank_key = f"{name}.lora_B.weight"
                if rank_key in state_dict:
                    rank[rank_key] = state_dict[rank_key].shape[1]

        if network_alphas is not None:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix]
            network_alphas = {k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys}

        # create `LoraConfig`
        lora_config = _create_lora_config(state_dict, network_alphas, metadata, rank, is_unet=False)

        # adapter_name
        if adapter_name is None:
            adapter_name = get_adapter_name(text_encoder)

        # <Unsafe code
        is_model_cpu_offload, is_sequential_cpu_offload, is_group_offload = _func_optionally_disable_offloading(
            _pipeline
        )
        # inject LoRA layers and load the state dict
        # in transformers we automatically check whether the adapter name is already in use or not
        text_encoder.load_adapter(
            adapter_name=adapter_name,
            adapter_state_dict=state_dict,
            peft_config=lora_config,
            **peft_kwargs,
        )

        # scale LoRA layers with `lora_scale`
        scale_lora_layers(text_encoder, weight=lora_scale)
        text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        elif is_group_offload:
            for component in _pipeline.components.values():
                if isinstance(component, torch.nn.Module):
                    _maybe_remove_and_reapply_group_offloading(component)
        # Unsafe code />

    if prefix is not None and not state_dict:
        model_class_name = text_encoder.__class__.__name__
        logger.warning(
            f"No LoRA keys associated to {model_class_name} found with the {prefix=}. "
            "This is safe to ignore if LoRA state dict didn't originally have any "
            f"{model_class_name} related params. You can also try specifying `prefix=None` "
            "to resolve the warning. Otherwise, open an issue if you think it's unexpected: "
            "https://github.com/huggingface/diffusers/issues/new"
        )


def _func_optionally_disable_offloading(_pipeline):
    """
    Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

    Args:
        _pipeline (`DiffusionPipeline`):
            The pipeline to disable offloading for.

    Returns:
        tuple:
            A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` or `is_group_offload` is True.
    """
    from ..hooks.group_offloading import _is_group_offload_enabled

    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    is_group_offload = False

    if _pipeline is not None and _pipeline.hf_device_map is None:
        for _, component in _pipeline.components.items():
            if not isinstance(component, nn.Module):
                continue
            is_group_offload = is_group_offload or _is_group_offload_enabled(component)
            if not hasattr(component, "_hf_hook"):
                continue
            is_model_cpu_offload = is_model_cpu_offload or isinstance(component._hf_hook, CpuOffload)
            is_sequential_cpu_offload = is_sequential_cpu_offload or (
                isinstance(component._hf_hook, AlignDevicesHook)
                or hasattr(component._hf_hook, "hooks")
                and isinstance(component._hf_hook.hooks[0], AlignDevicesHook)
            )

        if is_sequential_cpu_offload or is_model_cpu_offload:
            logger.info(
                "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
            )
            for _, component in _pipeline.components.items():
                if not isinstance(component, nn.Module) or not hasattr(component, "_hf_hook"):
                    continue
                remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

    return (is_model_cpu_offload, is_sequential_cpu_offload, is_group_offload)


class LoraBaseMixin:
    """Utility class for handling LoRAs."""

    _lora_loadable_modules = []
    _merged_adapters = set()

    @property
    def lora_scale(self) -> float:
        """
        Returns the lora scale which can be set at run time by the pipeline. # if `_lora_scale` has not been set,
        return 1.
        """
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0

    @property
    def num_fused_loras(self):
        """Returns the number of LoRAs that have been fused."""
        return len(self._merged_adapters)

    @property
    def fused_loras(self):
        """Returns names of the LoRAs that have been fused."""
        return self._merged_adapters

    def load_lora_weights(self, **kwargs):
        raise NotImplementedError("`load_lora_weights()` is not implemented.")

    @classmethod
    def save_lora_weights(cls, **kwargs):
        raise NotImplementedError("`save_lora_weights()` not implemented.")

    @classmethod
    def lora_state_dict(cls, **kwargs):
        raise NotImplementedError("`lora_state_dict()` is not implemented.")

    def unload_lora_weights(self):
        """
        Unloads the LoRA parameters.

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
        >>> pipeline.unload_lora_weights()
        >>> ...
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            if model is not None:
                if issubclass(model.__class__, ModelMixin):
                    model.unload_lora()
                elif issubclass(model.__class__, PreTrainedModel):
                    _remove_text_encoder_monkey_patch(model)

    def fuse_lora(
        self,
        components: List[str] = [],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        > [!WARNING] > This is an experimental API.

        Args:
            components: (`List[str]`): List of LoRA-injectable components to fuse the LoRAs into.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
        if "fuse_unet" in kwargs:
            depr_message = "Passing `fuse_unet` to `fuse_lora()` is deprecated and will be ignored. Please use the `components` argument and provide a list of the components whose LoRAs are to be fused. `fuse_unet` will be removed in a future version."
            deprecate(
                "fuse_unet",
                "1.0.0",
                depr_message,
            )
        if "fuse_transformer" in kwargs:
            depr_message = "Passing `fuse_transformer` to `fuse_lora()` is deprecated and will be ignored. Please use the `components` argument and provide a list of the components whose LoRAs are to be fused. `fuse_transformer` will be removed in a future version."
            deprecate(
                "fuse_transformer",
                "1.0.0",
                depr_message,
            )
        if "fuse_text_encoder" in kwargs:
            depr_message = "Passing `fuse_text_encoder` to `fuse_lora()` is deprecated and will be ignored. Please use the `components` argument and provide a list of the components whose LoRAs are to be fused. `fuse_text_encoder` will be removed in a future version."
            deprecate(
                "fuse_text_encoder",
                "1.0.0",
                depr_message,
            )

        if len(components) == 0:
            raise ValueError("`components` cannot be an empty list.")

        # Need to retrieve the names as `adapter_names` can be None. So we cannot directly use it
        # in `self._merged_adapters = self._merged_adapters | merged_adapter_names`.
        merged_adapter_names = set()
        for fuse_component in components:
            if fuse_component not in self._lora_loadable_modules:
                raise ValueError(f"{fuse_component} is not found in {self._lora_loadable_modules=}.")

            model = getattr(self, fuse_component, None)
            if model is not None:
                # check if diffusers model
                if issubclass(model.__class__, ModelMixin):
                    model.fuse_lora(lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names)
                    for module in model.modules():
                        if isinstance(module, BaseTunerLayer):
                            merged_adapter_names.update(set(module.merged_adapters))
                # handle transformers models.
                if issubclass(model.__class__, PreTrainedModel):
                    fuse_text_encoder_lora(
                        model, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names
                    )
                    for module in model.modules():
                        if isinstance(module, BaseTunerLayer):
                            merged_adapter_names.update(set(module.merged_adapters))

        self._merged_adapters = self._merged_adapters | merged_adapter_names

    def unfuse_lora(self, components: List[str] = [], **kwargs):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).

        > [!WARNING] > This is an experimental API.

        Args:
            components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.
            unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
            unfuse_text_encoder (`bool`, defaults to `True`):
                Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
        """
        if "unfuse_unet" in kwargs:
            depr_message = "Passing `unfuse_unet` to `unfuse_lora()` is deprecated and will be ignored. Please use the `components` argument. `unfuse_unet` will be removed in a future version."
            deprecate(
                "unfuse_unet",
                "1.0.0",
                depr_message,
            )
        if "unfuse_transformer" in kwargs:
            depr_message = "Passing `unfuse_transformer` to `unfuse_lora()` is deprecated and will be ignored. Please use the `components` argument. `unfuse_transformer` will be removed in a future version."
            deprecate(
                "unfuse_transformer",
                "1.0.0",
                depr_message,
            )
        if "unfuse_text_encoder" in kwargs:
            depr_message = "Passing `unfuse_text_encoder` to `unfuse_lora()` is deprecated and will be ignored. Please use the `components` argument. `unfuse_text_encoder` will be removed in a future version."
            deprecate(
                "unfuse_text_encoder",
                "1.0.0",
                depr_message,
            )

        if len(components) == 0:
            raise ValueError("`components` cannot be an empty list.")

        for fuse_component in components:
            if fuse_component not in self._lora_loadable_modules:
                raise ValueError(f"{fuse_component} is not found in {self._lora_loadable_modules=}.")

            model = getattr(self, fuse_component, None)
            if model is not None:
                if issubclass(model.__class__, (ModelMixin, PreTrainedModel)):
                    for module in model.modules():
                        if isinstance(module, BaseTunerLayer):
                            for adapter in set(module.merged_adapters):
                                if adapter and adapter in self._merged_adapters:
                                    self._merged_adapters = self._merged_adapters - {adapter}
                            module.unmerge()

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        adapter_weights: Optional[Union[float, Dict, List[float], List[Dict]]] = None,
    ):
        """
        Set the currently active adapters for use in the pipeline.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        if isinstance(adapter_weights, dict):
            components_passed = set(adapter_weights.keys())
            lora_components = set(self._lora_loadable_modules)

            invalid_components = sorted(components_passed - lora_components)
            if invalid_components:
                logger.warning(
                    f"The following components in `adapter_weights` are not part of the pipeline: {invalid_components}. "
                    f"Available components that are LoRA-compatible: {self._lora_loadable_modules}. So, weights belonging "
                    "to the invalid components will be removed and ignored."
                )
                adapter_weights = {k: v for k, v in adapter_weights.items() if k not in invalid_components}

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
        adapter_weights = copy.deepcopy(adapter_weights)

        # Expand weights into a list, one entry per adapter
        if not isinstance(adapter_weights, list):
            adapter_weights = [adapter_weights] * len(adapter_names)

        if len(adapter_names) != len(adapter_weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(adapter_weights)}"
            )

        list_adapters = self.get_list_adapters()  # eg {"unet": ["adapter1", "adapter2"], "text_encoder": ["adapter2"]}
        # eg ["adapter1", "adapter2"]
        all_adapters = {adapter for adapters in list_adapters.values() for adapter in adapters}
        missing_adapters = set(adapter_names) - all_adapters
        if len(missing_adapters) > 0:
            raise ValueError(
                f"Adapter name(s) {missing_adapters} not in the list of present adapters: {all_adapters}."
            )

        # eg {"adapter1": ["unet"], "adapter2": ["unet", "text_encoder"]}
        invert_list_adapters = {
            adapter: [part for part, adapters in list_adapters.items() if adapter in adapters]
            for adapter in all_adapters
        }

        # Decompose weights into weights for denoiser and text encoders.
        _component_adapter_weights = {}
        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            # To guard for cases like Wan. In Wan2.1 and WanVace, we have a single denoiser.
            # Whereas in Wan 2.2, we have two denoisers.
            if model is None:
                continue

            for adapter_name, weights in zip(adapter_names, adapter_weights):
                if isinstance(weights, dict):
                    component_adapter_weights = weights.pop(component, None)
                    if component_adapter_weights is not None and component not in invert_list_adapters[adapter_name]:
                        logger.warning(
                            (
                                f"Lora weight dict for adapter '{adapter_name}' contains {component},"
                                f"but this will be ignored because {adapter_name} does not contain weights for {component}."
                                f"Valid parts for {adapter_name} are: {invert_list_adapters[adapter_name]}."
                            )
                        )

                else:
                    component_adapter_weights = weights

                _component_adapter_weights.setdefault(component, [])
                _component_adapter_weights[component].append(component_adapter_weights)

            if issubclass(model.__class__, ModelMixin):
                model.set_adapters(adapter_names, _component_adapter_weights[component])
            elif issubclass(model.__class__, PreTrainedModel):
                set_adapters_for_text_encoder(adapter_names, model, _component_adapter_weights[component])

    def disable_lora(self):
        """
        Disables the active LoRA layers of the pipeline.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            if model is not None:
                if issubclass(model.__class__, ModelMixin):
                    model.disable_lora()
                elif issubclass(model.__class__, PreTrainedModel):
                    disable_lora_for_text_encoder(model)

    def enable_lora(self):
        """
        Enables the active LoRA layers of the pipeline.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.enable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            if model is not None:
                if issubclass(model.__class__, ModelMixin):
                    model.enable_lora()
                elif issubclass(model.__class__, PreTrainedModel):
                    enable_lora_for_text_encoder(model)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the pipeline.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names of the adapters to delete.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.delete_adapters("cinematic")
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            if model is not None:
                if issubclass(model.__class__, ModelMixin):
                    model.delete_adapters(adapter_names)
                elif issubclass(model.__class__, PreTrainedModel):
                    for adapter_name in adapter_names:
                        delete_adapter_layers(model, adapter_name)

    def get_active_adapters(self) -> List[str]:
        """
        Gets the list of the current active adapters.

        Example:

        ```python
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
        ).to("cuda")
        pipeline.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
        pipeline.get_active_adapters()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )

        active_adapters = []

        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            if model is not None and issubclass(model.__class__, ModelMixin):
                for module in model.modules():
                    if isinstance(module, BaseTunerLayer):
                        active_adapters = module.active_adapters
                        break

        return active_adapters

    def get_list_adapters(self) -> Dict[str, List[str]]:
        """
        Gets the current list of all available adapters in the pipeline.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )

        set_adapters = {}

        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            if (
                model is not None
                and issubclass(model.__class__, (ModelMixin, PreTrainedModel))
                and hasattr(model, "peft_config")
            ):
                set_adapters[component] = list(model.peft_config.keys())

        return set_adapters

    def set_lora_device(self, adapter_names: List[str], device: Union[torch.device, str, int]) -> None:
        """
        Moves the LoRAs listed in `adapter_names` to a target device. Useful for offloading the LoRA to the CPU in case
        you want to load multiple adapters and free some GPU memory.

        After offloading the LoRA adapters to CPU, as long as the rest of the model is still on GPU, the LoRA adapters
        can no longer be used for inference, as that would cause a device mismatch. Remember to set the device back to
        GPU before using those LoRA adapters for inference.

        ```python
        >>> pipe.load_lora_weights(path_1, adapter_name="adapter-1")
        >>> pipe.load_lora_weights(path_2, adapter_name="adapter-2")
        >>> pipe.set_adapters("adapter-1")
        >>> image_1 = pipe(**kwargs)
        >>> # switch to adapter-2, offload adapter-1
        >>> pipeline.set_lora_device(adapter_names=["adapter-1"], device="cpu")
        >>> pipeline.set_lora_device(adapter_names=["adapter-2"], device="cuda:0")
        >>> pipe.set_adapters("adapter-2")
        >>> image_2 = pipe(**kwargs)
        >>> # switch back to adapter-1, offload adapter-2
        >>> pipeline.set_lora_device(adapter_names=["adapter-2"], device="cpu")
        >>> pipeline.set_lora_device(adapter_names=["adapter-1"], device="cuda:0")
        >>> pipe.set_adapters("adapter-1")
        >>> ...
        ```

        Args:
            adapter_names (`List[str]`):
                List of adapters to send device to.
            device (`Union[torch.device, str, int]`):
                Device to send the adapters to. Can be either a torch device, a str or an integer.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        for component in self._lora_loadable_modules:
            model = getattr(self, component, None)
            if model is not None:
                for module in model.modules():
                    if isinstance(module, BaseTunerLayer):
                        for adapter_name in adapter_names:
                            if adapter_name not in module.lora_A:
                                # it is sufficient to check lora_A
                                continue

                            module.lora_A[adapter_name].to(device)
                            module.lora_B[adapter_name].to(device)
                            # this is a param, not a module, so device placement is not in-place -> re-assign
                            if hasattr(module, "lora_magnitude_vector") and module.lora_magnitude_vector is not None:
                                if adapter_name in module.lora_magnitude_vector:
                                    module.lora_magnitude_vector[adapter_name] = module.lora_magnitude_vector[
                                        adapter_name
                                    ].to(device)

    def enable_lora_hotswap(self, **kwargs) -> None:
        """
        Hotswap adapters without triggering recompilation of a model or if the ranks of the loaded adapters are
        different.

        Args:
            target_rank (`int`):
                The highest rank among all the adapters that will be loaded.
            check_compiled (`str`, *optional*, defaults to `"error"`):
                How to handle a model that is already compiled. The check can return the following messages:
                  - "error" (default): raise an error
                  - "warn": issue a warning
                  - "ignore": do nothing
        """
        for key, component in self.components.items():
            if hasattr(component, "enable_lora_hotswap") and (key in self._lora_loadable_modules):
                component.enable_lora_hotswap(**kwargs)

    @staticmethod
    def pack_weights(layers, prefix):
        layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
        return _pack_dict_with_prefix(layers_weights, prefix)

    @staticmethod
    def write_lora_layers(
        state_dict: Dict[str, torch.Tensor],
        save_directory: str,
        is_main_process: bool,
        weight_name: str,
        save_function: Callable,
        safe_serialization: bool,
        lora_adapter_metadata: Optional[dict] = None,
    ):
        """Writes the state dict of the LoRA layers (optionally with metadata) to disk."""
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if lora_adapter_metadata and not safe_serialization:
            raise ValueError("`lora_adapter_metadata` cannot be specified when not using `safe_serialization`.")
        if lora_adapter_metadata and not isinstance(lora_adapter_metadata, dict):
            raise TypeError("`lora_adapter_metadata` must be of type `dict`.")

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    # Inject framework format.
                    metadata = {"format": "pt"}
                    if lora_adapter_metadata:
                        for key, value in lora_adapter_metadata.items():
                            if isinstance(value, set):
                                lora_adapter_metadata[key] = list(value)
                        metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(
                            lora_adapter_metadata, indent=2, sort_keys=True
                        )

                    return safetensors.torch.save_file(weights, filename, metadata=metadata)

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME

        save_path = Path(save_directory, weight_name).as_posix()
        save_function(state_dict, save_path)
        logger.info(f"Model weights saved in {save_path}")

    @classmethod
    def _save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        lora_layers: Dict[str, Dict[str, Union[torch.nn.Module, torch.Tensor]]],
        lora_metadata: Dict[str, Optional[dict]],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        """
        Helper method to pack and save LoRA weights and metadata. This method centralizes the saving logic for all
        pipeline types.
        """
        state_dict = {}
        final_lora_adapter_metadata = {}

        for prefix, layers in lora_layers.items():
            state_dict.update(cls.pack_weights(layers, prefix))

        for prefix, metadata in lora_metadata.items():
            if metadata:
                final_lora_adapter_metadata.update(_pack_dict_with_prefix(metadata, prefix))

        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            lora_adapter_metadata=final_lora_adapter_metadata if final_lora_adapter_metadata else None,
        )

    @classmethod
    def _optionally_disable_offloading(cls, _pipeline):
        return _func_optionally_disable_offloading(_pipeline=_pipeline)
