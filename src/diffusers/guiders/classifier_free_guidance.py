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


class ClassifierFreeGuidance(GuidanceMixin):
    """
    Classifier-free guidance (CFG): https://huggingface.co/papers/2207.12598

    CFG is a technique used to improve generation quality and condition-following in diffusion models. It works by
    jointly training a model on both conditional and unconditional data, and using a weighted sum of the two during
    inference. This allows the model to tradeoff between generation quality and sample diversity.

    The original paper proposes scaling and shifting the conditional distribution based on the difference between
    conditional and unconditional predictions. [x_pred = x_cond + scale * (x_cond - x_uncond)]

    Diffusers implemented the scaling and shifting on the unconditional prediction instead, which is equivalent to what
    the original paper proposed in theory. [x_pred = x_uncond + scale * (x_cond - x_uncond)]

    The intution behind the original formulation can be thought of as moving the conditional distribution estimates
    further away from the unconditional distribution estimates, while the diffusers-native implementation can be
    thought of as moving the unconditional distribution towards the conditional distribution estimates to get rid of
    the unconditional predictions (usually negative features like "bad quality, bad anotomy, watermarks", etc.)

    The `use_original_formulation` argument can be set to `True` to use the original CFG formulation mentioned in the
    paper. By default, we use the diffusers-native implementation that has been in the codebase for a long time.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time.
    """

    def __init__(
        self, guidance_scale: float = 7.5, guidance_rescale: float = 0.0, use_original_formulation: bool = False
    ):
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = None

        if math.isclose(self.guidance_scale, 1.0):
            pred = pred_cond
        else:
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if not math.isclose(self.guidance_scale, 1.0):
            num_conditions += 1
        return num_conditions
