# Copyright 2026 HuggingFace Inc.
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

import pytest
import torch

from diffusers.guiders import MagnitudeAwareGuidance
from diffusers.guiders.magnitude_aware_guidance import mambo_guidance


@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (2, 4, 3, 8, 8)])
@pytest.mark.parametrize("use_original_formulation", [False, True])
def test_mambo_guidance_matches_per_sample_norm_reference(shape, use_original_formulation):
    generator = torch.Generator().manual_seed(0)
    pred_cond = torch.randn(shape, generator=generator)
    pred_uncond = torch.randn(shape, generator=generator)
    guidance_scale, alpha = 5.0, 8.0

    out = mambo_guidance(pred_cond, pred_uncond, guidance_scale, alpha, use_original_formulation)

    flat_diff = (pred_cond - pred_uncond).flatten(1)
    ratio = (flat_diff.norm(dim=1) / pred_uncond.flatten(1).norm(dim=1)).view(-1, *([1] * (len(shape) - 1)))
    if use_original_formulation:
        scale = guidance_scale * torch.exp(-alpha * ratio)
        expected = pred_cond + scale * (pred_cond - pred_uncond)
    else:
        scale = 1.0 + (guidance_scale - 1.0) * torch.exp(-alpha * ratio)
        expected = pred_uncond + scale * (pred_cond - pred_uncond)

    assert torch.allclose(out, expected, atol=1e-5)


def test_magnitude_aware_guidance_forward_on_image_latents():
    guider = MagnitudeAwareGuidance(guidance_scale=5.0)
    guider.set_state(step=0, num_inference_steps=10, timestep=torch.tensor(500))
    pred_cond = torch.randn(2, 4, 8, 8)
    pred_uncond = torch.randn(2, 4, 8, 8)

    output = guider.forward(pred_cond, pred_uncond)

    assert output.pred.shape == pred_cond.shape
