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

"""Tests for LTX2VideoUpBlock3d conv_in width wiring (#14307)."""

import torch

from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoUpBlock3d


def test_conv_in_projects_to_upsampler_input_width():
    """conv_in must project onto out_channels * upscale_factor, not out_channels.

    When in_channels != out_channels * upscale_factor the block needs a conv_in
    whose output matches the upsampler's expected input width.
    """
    block = LTX2VideoUpBlock3d(
        in_channels=48,
        out_channels=32,
        num_layers=1,
        spatio_temporal_scale=True,
        upscale_factor=2,
    )
    # 48 != 32*2=64, so conv_in must exist
    assert block.conv_in is not None

    # Forward pass must succeed without shape errors
    x = torch.randn(1, 48, 4, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 7, 16, 16)


def test_no_conv_in_when_widths_match():
    """When in_channels == out_channels * upscale_factor, conv_in is not needed."""
    block = LTX2VideoUpBlock3d(
        in_channels=64,
        out_channels=32,
        num_layers=1,
        spatio_temporal_scale=True,
        upscale_factor=2,
    )
    # 64 == 32*2, so conv_in should be None
    assert block.conv_in is None

    x = torch.randn(1, 64, 4, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 7, 16, 16)


def test_conv_in_with_no_upsampler():
    """When spatio_temporal_scale is False, upscale_factor is effectively 1."""
    block = LTX2VideoUpBlock3d(
        in_channels=48,
        out_channels=32,
        num_layers=1,
        spatio_temporal_scale=False,
        upscale_factor=1,
    )
    # 48 != 32*1=32, so conv_in must exist and project to 32
    assert block.conv_in is not None

    x = torch.randn(1, 48, 4, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 4, 8, 8)
