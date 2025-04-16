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

from typing import Dict, Optional, Union, Tuple, List

import torch

from .guider_utils import BaseGuidance, rescale_noise_cfg, _default_prepare_inputs


class CFGPlusPlusGuidance(BaseGuidance):
    """
    CFG++: https://huggingface.co/papers/2406.08070
    
    Args:
        guidance_scale (`float`, defaults to `0.7`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
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
        guidance_scale: float = 0.7,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__(start, stop)

        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def prepare_inputs(self, denoiser: torch.nn.Module, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
        return _default_prepare_inputs(denoiser, self.num_conditions, *args)

    def prepare_outputs(self, denoiser: torch.nn.Module, pred: torch.Tensor) -> None:
        self._num_outputs_prepared += 1
        if self._num_outputs_prepared > self.num_conditions:
            raise ValueError(f"Expected {self.num_conditions} outputs, but prepare_outputs called more times.")
        key = self._input_predictions[self._num_outputs_prepared - 1]
        self._preds[key] = pred

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = None

        if not self._is_cfgpp_enabled():
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
        if self._is_cfgpp_enabled():
            num_conditions += 1
        return num_conditions

    @property
    def outputs(self) -> Dict[str, torch.Tensor]:
        scheduler_step_kwargs = {}
        if self._is_cfgpp_enabled():
            scheduler_step_kwargs["_use_cfgpp"] = True
            scheduler_step_kwargs["_model_output_uncond"] = self._preds.get("pred_uncond")
        return self._preds, scheduler_step_kwargs

    def _is_cfgpp_enabled(self) -> bool:
        if not self._enabled:
            return False
        
        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self._start * self._num_inference_steps)
            skip_stop_step = int(self._stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step
        
        return is_within_range
