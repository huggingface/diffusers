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

import copy
import inspect
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch
import torch.nn as nn

from ..utils import (
    USE_PEFT_BACKEND,
    delete_adapter_layers,
    is_accelerate_available,
    is_transformers_available,
    logging,
    recurse_remove_peft_layers,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)


if is_transformers_available():
    from transformers import PreTrainedModel

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)


class LoraUtilsMixin:
    """Utility class for handling LoRAs."""

    is_unet_denoiser = False
    is_transformer_denoiser = False
    num_fused_loras = 0

    def load_lora_weights(self, **kwargs):
        raise NotImplementedError("`load_lora_weights()` is not implemented.")

    @classmethod
    def save_lora_weights(cls, **kwargs):
        raise NotImplementedError("`save_lora_weights()` not implemented.")

    def _remove_text_encoder_monkey_patch(self):
        if hasattr(self, "text_encoder"):
            recurse_remove_peft_layers(self.text_encoder)
            # TODO: @younesbelkada handle this in transformers side
            if getattr(self.text_encoder, "peft_config", None) is not None:
                del self.text_encoder.peft_config
                self.text_encoder._hf_peft_config_loaded = None

        if hasattr(self, "text_encoder_2"):
            recurse_remove_peft_layers(self.text_encoder_2)
            if getattr(self.text_encoder_2, "peft_config", None) is not None:
                del self.text_encoder_2.peft_config
                self.text_encoder_2._hf_peft_config_loaded = None

    @classmethod
    def _optionally_disable_offloading(cls, _pipeline):
        """
        Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

        Args:
            _pipeline (`DiffusionPipeline`):
                The pipeline to disable offloading for.

        Returns:
            tuple:
                A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
        """
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        if _pipeline is not None and _pipeline.hf_device_map is None:
            for _, component in _pipeline.components.items():
                if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                    if not is_model_cpu_offload:
                        is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                    if not is_sequential_cpu_offload:
                        is_sequential_cpu_offload = (
                            isinstance(component._hf_hook, AlignDevicesHook)
                            or hasattr(component._hf_hook, "hooks")
                            and isinstance(component._hf_hook.hooks[0], AlignDevicesHook)
                        )

                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

        return (is_model_cpu_offload, is_sequential_cpu_offload)

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

        if self.is_unet_denoiser:
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            unet.unload_lora()
        elif self.is_transformer_denoiser:
            transformer = (
                getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            )
            transformer.unload_lora()
        else:
            raise ValueError("No valid denoiser found in the network.")

        # Safe to call the following regardless of LoRA.
        self._remove_text_encoder_monkey_patch()

    def fuse_lora(
        self,
        fuse_denoiser: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
    ):
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            fuse_denoiser (`bool`, defaults to `True`):
                Whether to fuse the denoiser (UNet, Transformer, etc.) LoRA parameters.
            fuse_text_encoder (`bool`, defaults to `True`):
                Whether to fuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
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
        from peft.tuners.tuners_utils import BaseTunerLayer

        fuse_unet = True if fuse_denoiser and self.is_unet_denoiser else False
        fuse_transformer = True if fuse_denoiser and self.is_transformer_denoiser else False

        if fuse_unet:
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            unet.fuse_lora(lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names)
        elif fuse_transformer:
            transformer = (
                getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            )
            transformer.fuse_lora(lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names)

        def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False, adapter_names=None):
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

        if fuse_text_encoder:
            if hasattr(self, "text_encoder"):
                fuse_text_encoder_lora(self.text_encoder, lora_scale, safe_fusing, adapter_names=adapter_names)
            if hasattr(self, "text_encoder_2"):
                fuse_text_encoder_lora(self.text_encoder_2, lora_scale, safe_fusing, adapter_names=adapter_names)

        if fuse_denoiser or fuse_text_encoder:
            self.num_fused_loras += 1

    def unfuse_lora(self, unfuse_denoiser: bool = True, unfuse_text_encoder: bool = True):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
            unfuse_text_encoder (`bool`, defaults to `True`):
                Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        unfuse_unet = True if unfuse_denoiser and self.is_unet_denoiser else False
        unfuse_transformer = True if unfuse_denoiser and self.is_transformer_denoiser else False

        if unfuse_unet:
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            for module in unet.modules():
                if isinstance(module, BaseTunerLayer):
                    module.unmerge()
        elif unfuse_transformer:
            transformer = (
                getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            )
            for module in transformer.modules():
                if isinstance(module, BaseTunerLayer):
                    module.unmerge()

        def unfuse_text_encoder_lora(text_encoder):
            for module in text_encoder.modules():
                if isinstance(module, BaseTunerLayer):
                    module.unmerge()

        if unfuse_text_encoder:
            if hasattr(self, "text_encoder"):
                unfuse_text_encoder_lora(self.text_encoder)
            if hasattr(self, "text_encoder_2"):
                unfuse_text_encoder_lora(self.text_encoder_2)

        self.num_fused_loras -= 1

    def set_adapters_for_text_encoder(
        self,
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
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

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
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError(
                "The pipeline does not have a default `pipe.text_encoder` class. Please make sure to pass a `text_encoder` instead."
            )
        set_weights_and_activate_adapters(text_encoder, adapter_names, text_encoder_weights)

    def disable_lora_for_text_encoder(self, text_encoder: Optional["PreTrainedModel"] = None):
        """
        Disables the LoRA layers for the text encoder.

        Args:
            text_encoder (`torch.nn.Module`, *optional*):
                The text encoder module to disable the LoRA layers for. If `None`, it will try to get the
                `text_encoder` attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError("Text Encoder not found.")
        set_adapter_layers(text_encoder, enabled=False)

    def enable_lora_for_text_encoder(self, text_encoder: Optional["PreTrainedModel"] = None):
        """
        Enables the LoRA layers for the text encoder.

        Args:
            text_encoder (`torch.nn.Module`, *optional*):
                The text encoder module to enable the LoRA layers for. If `None`, it will try to get the `text_encoder`
                attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError("Text Encoder not found.")
        set_adapter_layers(self.text_encoder, enabled=True)

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        adapter_weights: Optional[Union[float, Dict, List[float], List[Dict]]] = None,
    ):
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        adapter_weights = copy.deepcopy(adapter_weights)

        # Expand weights into a list, one entry per adapter
        if not isinstance(adapter_weights, list):
            adapter_weights = [adapter_weights] * len(adapter_names)

        if len(adapter_names) != len(adapter_weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(adapter_weights)}"
            )

        # Decompose weights into weights for unet, text_encoder and text_encoder_2
        denoiser_lora_weights, text_encoder_lora_weights, text_encoder_2_lora_weights = [], [], []

        list_adapters = self.get_list_adapters()  # eg {"unet": ["adapter1", "adapter2"], "text_encoder": ["adapter2"]}
        all_adapters = {
            adapter for adapters in list_adapters.values() for adapter in adapters
        }  # eg ["adapter1", "adapter2"]
        invert_list_adapters = {
            adapter: [part for part, adapters in list_adapters.items() if adapter in adapters]
            for adapter in all_adapters
        }  # eg {"adapter1": ["unet"], "adapter2": ["unet", "text_encoder"]}

        denoiser_name = "unet" if self.is_unet_denoiser else "transformer"
        for adapter_name, weights in zip(adapter_names, adapter_weights):
            if isinstance(weights, dict):
                denoiser_lora_weight = weights.pop(denoiser_name, None)
                text_encoder_lora_weight = weights.pop("text_encoder", None)
                text_encoder_2_lora_weight = weights.pop("text_encoder_2", None)

                if len(weights) > 0:
                    raise ValueError(
                        f"Got invalid key '{weights.keys()}' in lora weight dict for adapter {adapter_name}."
                    )

                if text_encoder_2_lora_weight is not None and not hasattr(self, "text_encoder_2"):
                    logger.warning(
                        "Lora weight dict contains text_encoder_2 weights but will be ignored because pipeline does not have text_encoder_2."
                    )

                # warn if adapter doesn't have parts specified by adapter_weights
                for part_weight, part_name in zip(
                    [denoiser_lora_weight, text_encoder_lora_weight, text_encoder_2_lora_weight],
                    [denoiser_name, "text_encoder", "text_encoder_2"],
                ):
                    if part_weight is not None and part_name not in invert_list_adapters[adapter_name]:
                        logger.warning(
                            f"Lora weight dict for adapter '{adapter_name}' contains {part_name}, but this will be ignored because {adapter_name} does not contain weights for {part_name}. Valid parts for {adapter_name} are: {invert_list_adapters[adapter_name]}."
                        )

            else:
                denoiser_lora_weight = weights
                text_encoder_lora_weight = weights
                text_encoder_2_lora_weight = weights

            denoiser_lora_weights.append(denoiser_lora_weight)
            text_encoder_lora_weights.append(text_encoder_lora_weight)
            text_encoder_2_lora_weights.append(text_encoder_2_lora_weight)

        if denoiser_name == "unet":
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            # Handle the UNET
            unet.set_adapters(adapter_names, denoiser_lora_weights)
        else:
            transformer = (
                getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            )
            # Handle the UNET
            transformer.set_adapters(adapter_names, denoiser_lora_weights)

        # Handle the Text Encoder
        if hasattr(self, "text_encoder"):
            self.set_adapters_for_text_encoder(adapter_names, self.text_encoder, text_encoder_lora_weights)
        if hasattr(self, "text_encoder_2"):
            self.set_adapters_for_text_encoder(adapter_names, self.text_encoder_2, text_encoder_2_lora_weights)

    def disable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Disable denoiser adapters
        if self.is_unet_denoiser:
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            unet.disable_lora()
        else:
            transformer = (
                getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            )
            transformer.disable_lora()

        # Disable text encoder adapters
        if hasattr(self, "text_encoder"):
            self.disable_lora_for_text_encoder(self.text_encoder)
        if hasattr(self, "text_encoder_2"):
            self.disable_lora_for_text_encoder(self.text_encoder_2)

    def enable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Enable unet adapters
        if self.is_unet_denoiser:
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            unet.enable_lora()
        else:
            transformer = (
                getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            )
            transformer.enable_lora()

        # Enable text encoder adapters
        if hasattr(self, "text_encoder"):
            self.enable_lora_for_text_encoder(self.text_encoder)
        if hasattr(self, "text_encoder_2"):
            self.enable_lora_for_text_encoder(self.text_encoder_2)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Args:
        Deletes the LoRA layers of `adapter_name` for the unet and text-encoder(s).
            adapter_names (`Union[List[str], str]`):
                The names of the adapter to delete. Can be a single string or a list of strings
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Delete unet adapters
        if self.is_unet_denoiser:
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            unet.delete_adapters(adapter_names)
        else:
            transformer = (
                getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            )
            transformer.delete_adapters(adapter_names)

        for adapter_name in adapter_names:
            # Delete text encoder adapters
            if hasattr(self, "text_encoder"):
                delete_adapter_layers(self.text_encoder, adapter_name)
            if hasattr(self, "text_encoder_2"):
                delete_adapter_layers(self.text_encoder_2, adapter_name)

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

        from peft.tuners.tuners_utils import BaseTunerLayer

        active_adapters = []
        if self.is_unet_denoiser:
            denoiser = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        else:
            denoiser = getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer

        for module in denoiser.modules():
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

        if hasattr(self, "text_encoder") and hasattr(self.text_encoder, "peft_config"):
            set_adapters["text_encoder"] = list(self.text_encoder.peft_config.keys())

        if hasattr(self, "text_encoder_2") and hasattr(self.text_encoder_2, "peft_config"):
            set_adapters["text_encoder_2"] = list(self.text_encoder_2.peft_config.keys())

        if self.is_unet_denoiser:
            denoiser = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            denoiser_name = self.unet_name
        else:
            denoiser = getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
            denoiser_name = self.transformer_name

        if hasattr(self, denoiser_name) and hasattr(denoiser, "peft_config"):
            set_adapters[denoiser_name] = (
                list(self.unet.peft_config.keys())
                if self.is_unet_denoiser
                else list(self.transformer.peft_config.keys())
            )

        return set_adapters

    def set_lora_device(self, adapter_names: List[str], device: Union[torch.device, str, int]) -> None:
        """
        Moves the LoRAs listed in `adapter_names` to a target device. Useful for offloading the LoRA to the CPU in case
        you want to load multiple adapters and free some GPU memory.

        Args:
            adapter_names (`List[str]`):
                List of adapters to send device to.
            device (`Union[torch.device, str, int]`):
                Device to send the adapters to. Can be either a torch device, a str or an integer.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        # Handle the denoiser
        if self.is_unet_denoiser:
            denoiser = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        else:
            denoiser = getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer

        for denoiser_module in denoiser.modules():
            if isinstance(denoiser_module, BaseTunerLayer):
                for adapter_name in adapter_names:
                    denoiser_module.lora_A[adapter_name].to(device)
                    denoiser_module.lora_B[adapter_name].to(device)
                    # this is a param, not a module, so device placement is not in-place -> re-assign
                    if (
                        hasattr(denoiser_module, "lora_magnitude_vector")
                        and denoiser_module.lora_magnitude_vector is not None
                    ):
                        denoiser_module.lora_magnitude_vector[adapter_name] = denoiser_module.lora_magnitude_vector[
                            adapter_name
                        ].to(device)

        # Handle the text encoder
        modules_to_process = []
        if hasattr(self, "text_encoder"):
            modules_to_process.append(self.text_encoder)

        if hasattr(self, "text_encoder_2"):
            modules_to_process.append(self.text_encoder_2)

        for text_encoder in modules_to_process:
            # loop over submodules
            for text_encoder_module in text_encoder.modules():
                if isinstance(text_encoder_module, BaseTunerLayer):
                    for adapter_name in adapter_names:
                        text_encoder_module.lora_A[adapter_name].to(device)
                        text_encoder_module.lora_B[adapter_name].to(device)
                        # this is a param, not a module, so device placement is not in-place -> re-assign
                        if (
                            hasattr(text_encoder_module, "lora_magnitude_vector")
                            and text_encoder_module.lora_magnitude_vector is not None
                        ):
                            text_encoder_module.lora_magnitude_vector[
                                adapter_name
                            ] = text_encoder_module.lora_magnitude_vector[adapter_name].to(device)

    @staticmethod
    def pack_weights(layers, prefix):
        layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
        layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
        return layers_state_dict

    @staticmethod
    def write_lora_layers(
        state_dict: Dict[str, torch.Tensor],
        save_directory: str,
        is_main_process: bool,
        weight_name: str,
        save_function: Callable,
        safe_serialization: bool,
    ):
        from .lora import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

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
