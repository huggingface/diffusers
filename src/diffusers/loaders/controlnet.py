# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from contextlib import nullcontext
from typing import Dict, List, Optional, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args

from ..loaders.single_file_utils import DIFFUSERS_TO_LDM_MAPPING
from ..models.embeddings import (
    ImageProjection,
    MultiIPAdapterImageProjection,
)
from ..models.modeling_utils import load_model_dict_into_meta
from ..utils import (
    USE_PEFT_BACKEND,
    deprecate,
    get_submodule_by_name,
    is_accelerate_available,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    logging,
)
from .lora_base import (  # noqa
    LORA_WEIGHT_NAME,
    LORA_WEIGHT_NAME_SAFE,
    LoraBaseMixin,
    _fetch_state_dict,
    _load_lora_into_text_encoder,
)
from .lora_conversion_utils import (
    _convert_bfl_flux_control_lora_to_diffusers,
    _convert_hunyuan_video_lora_to_diffusers,
    _convert_kohya_flux_lora_to_diffusers,
    _convert_non_diffusers_lora_to_diffusers,
    _convert_stabilityai_control_lora_to_diffusers,
    _convert_xlabs_flux_lora_to_diffusers,
    _maybe_map_sgm_blocks_to_diffusers,
)


_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True

logger = logging.get_logger(__name__)

CONTROLNET_NAME = "controlnet"


class ControlNetLoadersMixin:
    """
    Load layers into a [`ControlNetModel`].
    """

    _lora_loadable_modules = ["controlnet"]
    controlnet_name = CONTROLNET_NAME
    _control_lora_supported_norm_keys = ["norm1", "norm2", "norm3"]
    
    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        self, 
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        return_alphas: bool = False,
        **kwargs):
        r"""
        """
        
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        state_dict = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        is_stabilityai = "lora_controlnet" in state_dict and "input_blocks.11.0.in_layers.0.weight" not in state_dict
        if is_stabilityai:
            state_dict = _convert_stabilityai_control_lora_to_diffusers(state_dict)
            return (state_dict, None) if return_alphas else state_dict

        raise ValueError

    def load_lora_weights(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], adapter_name=None, **kwargs
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict, return_alphas=True, **kwargs
        )

        has_lora_keys = any("lora" in key for key in state_dict.keys())

        # Control LoRAs also have norm keys
        has_norm_keys = any(
            norm_key in key for key in state_dict.keys() for norm_key in self._control_lora_supported_norm_keys
        )

        if not (has_lora_keys or has_norm_keys):
            raise ValueError("Invalid LoRA checkpoint.")

        controlnet_lora_state_dict = {
            k: state_dict.pop(k) for k in list(state_dict.keys()) if "lora" in k
        }
        controlnet_norm_state_dict = {
            k: state_dict.pop(k)
            for k in list(state_dict.keys())
            if any(norm_key in k for norm_key in self._control_lora_supported_norm_keys)
        }
        controlnet_others_state_dict = {
            k: state_dict.pop(k) for k in list(state_dict.keys())
        }

        controlnet = self

        if len(controlnet_lora_state_dict) > 0:
            self.load_lora_into_controlnet(
                controlnet_lora_state_dict,
                network_alphas=network_alphas,
                controlnet=controlnet,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        
        if len(controlnet_norm_state_dict) > 0:
            self._load_norm_into_controlnet(
                controlnet_norm_state_dict,
                controlnet=controlnet,
                discard_original_layers=False,
            )
        
        if len(controlnet_others_state_dict) > 0:
            self._load_others_into_controlnet(
                controlnet_others_state_dict,
                controlnet=controlnet,
                discard_original_layers=False,
            )
    
    @classmethod
    def load_lora_into_controlnet(
        cls, state_dict, network_alphas, controlnet, adapter_name=None, _pipeline=None, low_cpu_mem_usage=False
    ):
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # Load the layers corresponding to controlnet.
        logger.info(f"Loading {cls}.")
        controlnet.load_lora_adapter(
            state_dict,
            network_alphas=network_alphas,
            adapter_name=adapter_name,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    
    @classmethod
    def _load_norm_into_controlnet(
        cls,
        state_dict,
        controlnet,
        prefix=None,
        discard_original_layers=False,
    ) -> Dict[str, torch.Tensor]:
        # Remove prefix if present
        prefix = prefix or cls.controlnet_name
        for key in list(state_dict.keys()):
            if key.split(".")[0] == prefix:
                state_dict[key[len(f"{prefix}.") :]] = state_dict.pop(key)

        # Find invalid keys
        controlnet_state_dict = controlnet.state_dict()
        controlnet_keys = set(controlnet_state_dict.keys())
        state_dict_keys = set(state_dict.keys())
        extra_keys = list(state_dict_keys - controlnet_keys)

        if extra_keys:
            logger.warning(
                f"Unsupported keys found in state dict when trying to load normalization layers into the controlnet. The following keys will be ignored:\n{extra_keys}."
            )

        for key in extra_keys:
            state_dict.pop(key)

        # Save the layers that are going to be overwritten so that unload_lora_weights can work as expected
        overwritten_layers_state_dict = {}
        if not discard_original_layers:
            for key in state_dict.keys():
                overwritten_layers_state_dict[key] = controlnet_state_dict[key].clone()

        logger.info(
            "The provided state dict contains normalization layers in addition to LoRA layers. The normalization layers will directly update the state_dict of the controlnet "
            'as opposed to the LoRA layers that will co-exist separately until the "fuse_lora()" method is called. That is to say, the normalization layers will always be directly '
            "fused into the controlnet and can only be unfused if `discard_original_layers=True` is passed. This might also have implications when dealing with multiple LoRAs. "
            "If you notice something unexpected, please open an issue: https://github.com/huggingface/diffusers/issues."
        )

        # We can't load with strict=True because the current state_dict does not contain all the controlnet keys
        incompatible_keys = controlnet.load_state_dict(state_dict, strict=False)
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)

        # We shouldn't expect to see the supported norm keys here being present in the unexpected keys.
        if unexpected_keys:
            if any(norm_key in k for k in unexpected_keys for norm_key in cls._control_lora_supported_norm_keys):
                raise ValueError(
                    f"Found {unexpected_keys} as unexpected keys while trying to load norm layers into the controlnet."
                )

        return overwritten_layers_state_dict
    
    @classmethod
    def _load_others_into_controlnet(
        cls,
        state_dict,
        controlnet,
        prefix=None,
        discard_original_layers=False,
    ) -> Dict[str, torch.Tensor]:
        # Remove prefix if present
        prefix = prefix or cls.controlnet_name
        for key in list(state_dict.keys()):
            if key.split(".")[0] == prefix:
                state_dict[key[len(f"{prefix}.") :]] = state_dict.pop(key)

        # Find invalid keys
        controlnet_state_dict = controlnet.state_dict()
        controlnet_keys = set(controlnet_state_dict.keys())
        state_dict_keys = set(state_dict.keys())
        extra_keys = list(state_dict_keys - controlnet_keys)

        if extra_keys:
            logger.warning(
                f"Unsupported keys found in state dict when trying to load normalization layers into the controlnet. The following keys will be ignored:\n{extra_keys}."
            )

        for key in extra_keys:
            state_dict.pop(key)

        # Save the layers that are going to be overwritten so that unload_lora_weights can work as expected
        overwritten_layers_state_dict = {}
        if not discard_original_layers:
            for key in state_dict.keys():
                overwritten_layers_state_dict[key] = controlnet_state_dict[key].clone()

        logger.info(
            "The provided state dict contains normalization layers in addition to LoRA layers. The normalization layers will directly update the state_dict of the transformer "
            'as opposed to the LoRA layers that will co-exist separately until the "fuse_lora()" method is called. That is to say, the normalization layers will always be directly '
            "fused into the controlnet and can only be unfused if `discard_original_layers=True` is passed. This might also have implications when dealing with multiple LoRAs. "
            "If you notice something unexpected, please open an issue: https://github.com/huggingface/diffusers/issues."
        )

        # We can't load with strict=True because the current state_dict does not contain all the transformer keys
        incompatible_keys = controlnet.load_state_dict(state_dict, strict=False)
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)

        # We shouldn't expect to see the supported norm keys here being present in the unexpected keys.
        if unexpected_keys:
            if any(norm_key in k for k in unexpected_keys for norm_key in cls._control_lora_supported_norm_keys):
                raise ValueError(
                    f"Found {unexpected_keys} as unexpected keys while trying to load norm layers into the transformer."
                )

        return overwritten_layers_state_dict

    def fuse_lora(
        self,
        components: List[str] = ["controlnet"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().fuse_lora(
            components=components, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names
        )
    
    def unfuse_lora(self, components: List[str] = ["controlnet"], **kwargs):
        super().unfuse_lora(components=components)
