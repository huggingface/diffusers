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

import math
from typing import List, Optional, Tuple, Union

import torch

from ..hooks import HookRegistry, LayerSkipConfig
from ..hooks.layer_skip import _apply_layer_skip_hook
from .guider_utils import BaseGuidance, rescale_noise_cfg, _default_prepare_inputs


class AutoGuidance(BaseGuidance):
    """
    AutoGuidance: https://huggingface.co/papers/2406.02507
    
    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        auto_guidance_layers (`int` or `List[int]`, *optional*):
            The layer indices to apply skip layer guidance to. Can be a single integer or a list of integers. If not
            provided, `skip_layer_config` must be provided.
        auto_guidance_config (`LayerSkipConfig` or `List[LayerSkipConfig]`, *optional*):
            The configuration for the skip layer guidance. Can be a single `LayerSkipConfig` or a list of
            `LayerSkipConfig`. If not provided, `skip_layer_guidance_layers` must be provided.
        dropout (`float`, *optional*):
            The dropout probability for autoguidance on the enabled skip layers (either with `auto_guidance_layers` or
            `auto_guidance_config`). If not provided, the dropout probability will be set to 1.0.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
        start (`float`, defaults to `0.0`):
            The fraction of the total number of denoising steps after which guidance starts.
        stop (`float`, defaults to `1.0`):
            The fraction of the total number of denoising steps after which guidance stops.
    """

    _input_predictions = ["pred_cond", "pred_uncond"]

    def __init__(
        self,
        guidance_scale: float = 7.5,
        auto_guidance_layers: Optional[Union[int, List[int]]] = None,
        auto_guidance_config: Union[LayerSkipConfig, List[LayerSkipConfig]] = None,
        dropout: Optional[float] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__(start, stop)

        self.guidance_scale = guidance_scale
        self.auto_guidance_layers = auto_guidance_layers
        self.auto_guidance_config = auto_guidance_config
        self.dropout = dropout
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        if auto_guidance_layers is None and auto_guidance_config is None:
            raise ValueError(
                "Either `auto_guidance_layers` or `auto_guidance_config` must be provided to enable Skip Layer Guidance."
            )
        if auto_guidance_layers is not None and auto_guidance_config is not None:
            raise ValueError("Only one of `auto_guidance_layers` or `auto_guidance_config` can be provided.")
        if (dropout is None and auto_guidance_layers is not None) or (dropout is not None and auto_guidance_layers is None):
            raise ValueError("`dropout` must be provided if `auto_guidance_layers` is provided.")

        if auto_guidance_layers is not None:
            if isinstance(auto_guidance_layers, int):
                auto_guidance_layers = [auto_guidance_layers]
            if not isinstance(auto_guidance_layers, list):
                raise ValueError(
                    f"Expected `auto_guidance_layers` to be an int or a list of ints, but got {type(auto_guidance_layers)}."
                )
            auto_guidance_config = [LayerSkipConfig(layer, fqn="auto", dropout=dropout) for layer in auto_guidance_layers]

        if isinstance(auto_guidance_config, LayerSkipConfig):
            auto_guidance_config = [auto_guidance_config]

        if not isinstance(auto_guidance_config, list):
            raise ValueError(
                f"Expected `auto_guidance_config` to be a LayerSkipConfig or a list of LayerSkipConfig, but got {type(auto_guidance_config)}."
            )

        self.auto_guidance_config = auto_guidance_config
        self._auto_guidance_hook_names = [f"AutoGuidance_{i}" for i in range(len(self.auto_guidance_config))]

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        if self._is_ag_enabled() and self.is_unconditional:
            for name, config in zip(self._auto_guidance_hook_names, self.auto_guidance_config):
                _apply_layer_skip_hook(denoiser, config, name=name)
    
    def prepare_inputs(self, denoiser: torch.nn.Module, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
        return _default_prepare_inputs(denoiser, self.num_conditions, *args)

    def prepare_outputs(self, denoiser: torch.nn.Module, pred: torch.Tensor) -> None:
        self._num_outputs_prepared += 1
        if self._num_outputs_prepared > self.num_conditions:
            raise ValueError(f"Expected {self.num_conditions} outputs, but prepare_outputs called more times.")
        key = self._input_predictions[self._num_outputs_prepared - 1]
        self._preds[key] = pred

        if key == "pred_uncond":
            # If we are in AutoGuidance unconditional inference mode, we need to remove the hooks after inference
            registry = HookRegistry.check_if_exists_or_initialize(denoiser)
            # Remove the hooks after inference
            for hook_name in self._auto_guidance_hook_names:
                registry.remove_hook(hook_name, recurse=True)

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = None

        if not self._is_ag_enabled():
            pred = pred_cond
        else:
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred
    
    @property
    def is_conditional(self) -> bool:
        return self._num_outputs_prepared == 0

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_ag_enabled():
            num_conditions += 1
        return num_conditions

    def _is_ag_enabled(self) -> bool:
        if not self._enabled:
            return False
        
        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self._start * self._num_inference_steps)
            skip_stop_step = int(self._stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step
        
        is_close = False
        if self.use_original_formulation:
            is_close = math.isclose(self.guidance_scale, 0.0)
        else:
            is_close = math.isclose(self.guidance_scale, 1.0)
        
        return is_within_range and not is_close
