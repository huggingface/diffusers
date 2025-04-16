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

from ..hooks import HookRegistry
from ..hooks.smoothed_energy_guidance_utils import SmoothedEnergyGuidanceConfig, _apply_smoothed_energy_guidance_hook
from .guider_utils import BaseGuidance, rescale_noise_cfg


class SmoothedEnergyGuidance(BaseGuidance):
    """
    Smoothed Energy Guidance (SEG): https://huggingface.co/papers/2408.00760

    SEG is only supported as an experimental prototype feature for now, so the implementation may be modified
    in the future without warning or guarantee of reproducibility. This implementation assumes:
    - Generated images are square (height == width)
    - The model does not combine different modalities together (e.g., text and image latent streams are
      not combined together such as Flux)
    
    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        seg_guidance_scale (`float`, defaults to `3.0`):
            The scale parameter for smoothed energy guidance. Anatomy and structure coherence may improve with higher
            values, but it may also lead to overexposure and saturation.
        seg_blur_sigma (`float`, defaults to `9999999.0`):
            The amount by which we blur the attention weights. Setting this value greater than 9999.0 results in
            infinite blur, which means uniform queries. Controlling it exponentially is empirically effective.
        seg_blur_threshold_inf (`float`, defaults to `9999.0`):
            The threshold above which the blur is considered infinite.
        seg_guidance_start (`float`, defaults to `0.0`):
            The fraction of the total number of denoising steps after which smoothed energy guidance starts.
        seg_guidance_stop (`float`, defaults to `1.0`):
            The fraction of the total number of denoising steps after which smoothed energy guidance stops.
        seg_guidance_layers (`int` or `List[int]`, *optional*):
            The layer indices to apply smoothed energy guidance to. Can be a single integer or a list of integers. If not
            provided, `seg_guidance_config` must be provided. The recommended values are `[7, 8, 9]` for Stable Diffusion
            3.5 Medium.
        seg_guidance_config (`SmoothedEnergyGuidanceConfig` or `List[SmoothedEnergyGuidanceConfig]`, *optional*):
            The configuration for the smoothed energy layer guidance. Can be a single `SmoothedEnergyGuidanceConfig` or a list of
            `SmoothedEnergyGuidanceConfig`. If not provided, `seg_guidance_layers` must be provided.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
        start (`float`, defaults to `0.01`):
            The fraction of the total number of denoising steps after which guidance starts.
        stop (`float`, defaults to `0.2`):
            The fraction of the total number of denoising steps after which guidance stops.
    """

    _input_predictions = ["pred_cond", "pred_uncond", "pred_cond_seg"]

    def __init__(
        self,
        guidance_scale: float = 7.5,
        seg_guidance_scale: float = 2.8,
        seg_blur_sigma: float = 9999999.0,
        seg_blur_threshold_inf: float = 9999.0,
        seg_guidance_start: float = 0.0,
        seg_guidance_stop: float = 1.0,
        seg_guidance_layers: Optional[Union[int, List[int]]] = None,
        seg_guidance_config: Union[SmoothedEnergyGuidanceConfig, List[SmoothedEnergyGuidanceConfig]] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__(start, stop)

        self.guidance_scale = guidance_scale
        self.seg_guidance_scale = seg_guidance_scale
        self.seg_blur_sigma = seg_blur_sigma
        self.seg_blur_threshold_inf = seg_blur_threshold_inf
        self.seg_guidance_start = seg_guidance_start
        self.seg_guidance_stop = seg_guidance_stop
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        if not (0.0 <= seg_guidance_start < 1.0):
            raise ValueError(
                f"Expected `seg_guidance_start` to be between 0.0 and 1.0, but got {seg_guidance_start}."
            )
        if not (seg_guidance_start <= seg_guidance_stop <= 1.0):
            raise ValueError(
                f"Expected `seg_guidance_stop` to be between 0.0 and 1.0, but got {seg_guidance_stop}."
            )

        if seg_guidance_layers is None and seg_guidance_config is None:
            raise ValueError(
                "Either `seg_guidance_layers` or `seg_guidance_config` must be provided to enable Smoothed Energy Guidance."
            )
        if seg_guidance_layers is not None and seg_guidance_config is not None:
            raise ValueError("Only one of `seg_guidance_layers` or `seg_guidance_config` can be provided.")

        if seg_guidance_layers is not None:
            if isinstance(seg_guidance_layers, int):
                seg_guidance_layers = [seg_guidance_layers]
            if not isinstance(seg_guidance_layers, list):
                raise ValueError(
                    f"Expected `seg_guidance_layers` to be an int or a list of ints, but got {type(seg_guidance_layers)}."
                )
            seg_guidance_config = [SmoothedEnergyGuidanceConfig(layer, fqn="auto") for layer in seg_guidance_layers]

        if isinstance(seg_guidance_config, SmoothedEnergyGuidanceConfig):
            seg_guidance_config = [seg_guidance_config]

        if not isinstance(seg_guidance_config, list):
            raise ValueError(
                f"Expected `seg_guidance_config` to be a SmoothedEnergyGuidanceConfig or a list of SmoothedEnergyGuidanceConfig, but got {type(seg_guidance_config)}."
            )

        self.seg_guidance_config = seg_guidance_config
        self._seg_layer_hook_names = [f"SmoothedEnergyGuidance_{i}" for i in range(len(self.seg_guidance_config))]

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        if self._is_seg_enabled() and self.is_conditional and self._num_outputs_prepared > 0:
            for name, config in zip(self._seg_layer_hook_names, self.seg_guidance_config):
                _apply_smoothed_energy_guidance_hook(denoiser, config, self.seg_blur_sigma, name=name)
    
    def prepare_inputs(self, denoiser: torch.nn.Module, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
        num_conditions = self.num_conditions
        list_of_inputs = []
        for arg in args:
            if arg is None or isinstance(arg, torch.Tensor):
                list_of_inputs.append([arg] * num_conditions)
            elif isinstance(arg, (tuple, list)):
                if len(arg) != 2:
                    raise ValueError(
                        f"Expected a tuple or list of length 2, but got {len(arg)} for argument {arg}. Please provide a tuple/list of length 2 "
                        f"with the first element being the conditional input and the second element being the unconditional input or None."
                    )
                if arg[1] is None:
                    # Only conditioning inputs for all batches
                    list_of_inputs.append([arg[0]] * num_conditions)
                else:
                    list_of_inputs.append([arg[0], arg[1], arg[0]])
            else:
                raise ValueError(
                    f"Expected a tensor, tuple, or list, but got {type(arg)} for argument {arg}. Please provide a tensor, tuple, or list."
                )
        return tuple(list_of_inputs)

    def prepare_outputs(self, denoiser: torch.nn.Module, pred: torch.Tensor) -> None:
        self._num_outputs_prepared += 1
        if self._num_outputs_prepared > self.num_conditions:
            raise ValueError(f"Expected {self.num_conditions} outputs, but prepare_outputs called more times.")
        key = self._input_predictions[self._num_outputs_prepared - 1]
        if not self._is_cfg_enabled() and self._is_seg_enabled():
            # If we're predicting pred_cond and pred_cond_seg only, we need to set the key to pred_cond_seg
            # to avoid writing into pred_uncond which is not used
            if self._num_outputs_prepared == 2:
                key = "pred_cond_seg"
        self._preds[key] = pred

        if key == "pred_cond_seg":
            # If we are in SLG mode, we need to remove the hooks after inference
            registry = HookRegistry.check_if_exists_or_initialize(denoiser)
            # Remove the hooks after inference
            for hook_name in self._seg_layer_hook_names:
                registry.remove_hook(hook_name, recurse=True)

    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: Optional[torch.Tensor] = None,
        pred_cond_seg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = None

        if not self._is_cfg_enabled() and not self._is_seg_enabled():
            pred = pred_cond
        elif not self._is_cfg_enabled():
            shift = pred_cond - pred_cond_seg
            pred = pred_cond if self.use_original_formulation else pred_cond_seg
            pred = pred + self.seg_guidance_scale * shift
        elif not self._is_seg_enabled():
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift
        else:
            shift = pred_cond - pred_uncond
            shift_seg = pred_cond - pred_cond_seg
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift + self.seg_guidance_scale * shift_seg

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred
    
    @property
    def is_conditional(self) -> bool:
        return self._num_outputs_prepared == 0 or self._num_outputs_prepared == 2

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        if self._is_seg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_cfg_enabled(self) -> bool:
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

    def _is_seg_enabled(self) -> bool:
        if not self._enabled:
            return False
        
        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self.seg_guidance_start * self._num_inference_steps)
            skip_stop_step = int(self.seg_guidance_stop * self._num_inference_steps)
            is_within_range = skip_start_step < self._step < skip_stop_step
        
        is_zero = math.isclose(self.seg_guidance_scale, 0.0)
        
        return is_within_range and not is_zero
