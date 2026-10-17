# Copyright 2026 The HuggingFace Team. All rights reserved.
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


import torch

from ..configuration_utils import register_to_config
from .classifier_free_guidance import ClassifierFreeGuidance
from .guider_utils import GuiderOutput, rescale_noise_cfg


class TripoSplatClassifierFreeGuidance(ClassifierFreeGuidance):
    """Classifier-free guidance with TripoSplat's prediction arithmetic and conditional-only scales below one."""

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(guidance_scale, guidance_rescale, use_original_formulation, start, stop, enabled)

    def _is_cfg_enabled(self) -> bool:
        if not self.use_original_formulation and self.guidance_scale <= 1:
            return False
        return super()._is_cfg_enabled()

    def forward(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor | None = None) -> GuiderOutput:
        if not self._is_cfg_enabled():
            pred = pred_cond
        else:
            scale = self.guidance_scale + int(self.use_original_formulation)
            pred = scale * pred_cond - (scale - 1) * pred_uncond
        if self.guidance_rescale > 0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)
        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)
