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
from typing import Optional

import torch

from .guider_utils import GuidanceMixin, rescale_noise_cfg


class ClassifierFreeZeroStarGuidance(GuidanceMixin):
    """
    Classifier-free Zero* (CFG-Zero*): https://huggingface.co/papers/2503.18886

    This is an implementation of the Classifier-Free Zero* guidance technique, which is a variant of classifier-free
    guidance. It proposes zero initialization of the noise predictions for the first few steps of the diffusion
    process, and also introduces an optimal rescaling factor for the noise predictions, which can help in improving the
    quality of generated images.

    The authors of the paper suggest setting zero initialization in the first 4% of the inference steps.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        zero_init_steps (`int`, defaults to `1`):
            The number of inference steps for which the noise predictions are zeroed out (see Section 4.2).
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
    """

    _input_predictions = ["pred_cond", "pred_uncond"]

    def __init__(
        self,
        guidance_scale: float = 7.5,
        zero_init_steps: int = 1,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
    ):
        super().__init__()

        self.guidance_scale = guidance_scale
        self.zero_init_steps = zero_init_steps
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = None

        if self._step < self.zero_init_steps:
            pred = torch.zeros_like(pred_cond)
        elif self._is_cfg_enabled():
            pred = pred_cond
        else:
            shift = pred_cond - pred_uncond
            pred_cond_flat = pred_cond.flatten(1)
            pred_uncond_flat = pred_uncond.flatten(1)
            alpha = cfg_zero_star_scale(pred_cond_flat, pred_uncond_flat)
            alpha = alpha.view(-1, *(1,) * (len(pred_cond.shape) - 1))
            pred_uncond = pred_uncond * alpha
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_cfg_enabled(self) -> bool:
        if self.use_original_formulation:
            return not math.isclose(self.guidance_scale, 0.0)
        else:
            return not math.isclose(self.guidance_scale, 1.0)


def cfg_zero_star_scale(cond: torch.Tensor, uncond: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cond = cond.float()
    uncond = uncond.float()
    dot_product = torch.sum(cond * uncond, dim=1, keepdim=True)
    squared_norm = torch.sum(uncond**2, dim=1, keepdim=True) + eps
    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    scale = dot_product / squared_norm
    return scale.type_as(cond)
