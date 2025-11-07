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

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


class ClassifierFreeGuidance(BaseGuidance):
    """
    Implements Classifier-Free Guidance (CFG) for diffusion models.

    Reference: https://huggingface.co/papers/2207.12598

    CFG improves generation quality and prompt adherence by jointly training models on both conditional and
    unconditional data, then combining predictions during inference. This allows trading off between quality (high
    guidance) and diversity (low guidance).

    **Two CFG Formulations:**

    1. **Original formulation** (from paper):
       ```
       x_pred = x_cond + guidance_scale * (x_cond - x_uncond)
       ```
       Moves conditional predictions further from unconditional ones.

    2. **Diffusers-native formulation** (default, from Imagen paper):
       ```
       x_pred = x_uncond + guidance_scale * (x_cond - x_uncond)
       ```
       Moves unconditional predictions toward conditional ones, effectively suppressing negative features (e.g., "bad
       quality", "watermarks"). Equivalent in theory but more intuitive.

    Use `use_original_formulation=True` to switch to the original formulation.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            CFG scale applied by this guider during post-processing. Higher values = stronger prompt conditioning but
            may reduce quality. Typical range: 1.0-20.0.
        guidance_rescale (`float`, defaults to `0.0`):
            Rescaling factor to prevent overexposure from high guidance scales. Based on [Common Diffusion Noise
            Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Range: 0.0 (no rescaling)
            to 1.0 (full rescaling).
        use_original_formulation (`bool`, defaults to `False`):
            If `True`, uses the original CFG formulation from the paper. If `False` (default), uses the
            diffusers-native formulation from the Imagen paper.
        start (`float`, defaults to `0.0`):
            Fraction of denoising steps (0.0-1.0) after which CFG starts. Use > 0.0 to disable CFG in early denoising
            steps.
        stop (`float`, defaults to `1.0`):
            Fraction of denoising steps (0.0-1.0) after which CFG stops. Use < 1.0 to disable CFG in late denoising
            steps.
        enabled (`bool`, defaults to `True`):
            Whether CFG is enabled. Set to `False` to disable CFG entirely (uses only conditional predictions).
    """

    _input_predictions = ["pred_cond", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> GuiderOutput:
        pred = None

        if not self._is_cfg_enabled():
            pred = pred_cond
        else:
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
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
