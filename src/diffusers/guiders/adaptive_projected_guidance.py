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


class AdaptiveProjectedGuidance(GuidanceMixin):
    """
    Adaptive Projected Guidance (APG): https://huggingface.co/papers/2410.02416

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        adaptive_projected_guidance_momentum (`float`, defaults to `None`):
            The momentum parameter for the adaptive projected guidance. Disabled if set to `None`.
        adaptive_projected_guidance_rescale (`float`, defaults to `15.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time.
    """

    def __init__(
        self,
        guidance_scale: float = 7.5,
        adaptive_projected_guidance_momentum: Optional[float] = None,
        adaptive_projected_guidance_rescale: float = 15.0,
        eta: float = 1.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
    ):
        self.guidance_scale = guidance_scale
        self.adaptive_projected_guidance_momentum = adaptive_projected_guidance_momentum
        self.adaptive_projected_guidance_rescale = adaptive_projected_guidance_rescale
        self.eta = eta
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation
        self.momentum_buffer = None

    def prepare_inputs(self, *args):
        if self._step == 0:
            if self.adaptive_projected_guidance_momentum is not None:
                self.momentum_buffer = MomentumBuffer(self.adaptive_projected_guidance_momentum)
        return super().prepare_inputs(*args)

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = None

        if math.isclose(self.guidance_scale, 1.0):
            pred = pred_cond
        else:
            pred = normalized_guidance(
                pred_cond,
                pred_uncond,
                self.guidance_scale,
                self.momentum_buffer,
                self.eta,
                self.adaptive_projected_guidance_rescale,
                self.use_original_formulation,
            )

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if not math.isclose(self.guidance_scale, 1.0):
            num_conditions += 1
        return num_conditions


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: Optional[MomentumBuffer] = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
    use_original_formulation: bool = False,
):
    diff = pred_cond - pred_uncond
    dim = [-i for i in range(1, len(diff.shape))]
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dim, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    v0, v1 = diff.double(), pred_cond.double()
    v1 = torch.nn.functional.normalize(v1, dim=dim)
    v0_parallel = (v0 * v1).sum(dim=dim, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel, diff_orthogonal = v0_parallel.type_as(diff), v0_orthogonal.type_as(diff)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred = pred_cond if use_original_formulation else pred_uncond
    pred = pred + (guidance_scale - 1) * normalized_update
    return pred
