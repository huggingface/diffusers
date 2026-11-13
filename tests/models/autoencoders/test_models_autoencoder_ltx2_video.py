# coding=utf-8
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

from diffusers import AutoencoderKLLTX2Video
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLLTX2VideoTesterConfig(BaseModelTesterConfig):
    @property
    def main_input_name(self):
        return "sample"

    @property
    def model_class(self):
        return AutoencoderKLLTX2Video

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "decoder_block_out_channels": (16, 32, 64),
            "layers_per_block": (1, 1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, True, True),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (False, False, False, False),
            "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": False,
            "patch_size": 1,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
            "encoder_spatial_padding_mode": "zeros",
            # Full model uses `reflect` but this does not have deterministic backward implementation, so use `zeros`
            "decoder_spatial_padding_mode": "zeros",
        }

    def get_dummy_inputs(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLLTX2Video(AutoencoderKLLTX2VideoTesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    def test_outputs_equivalence(self):
        pytest.skip("Unsupported test.")


class TestAutoencoderKLLTX2VideoTraining(AutoencoderKLLTX2VideoTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLLTX2Video."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "LTX2VideoEncoder3d",
            "LTX2VideoDecoder3d",
            "LTX2VideoDownBlock3D",
            "LTX2VideoMidBlock3d",
            "LTX2VideoUpBlock3d",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAutoencoderKLLTX2VideoMemory(AutoencoderKLLTX2VideoTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLLTX2Video."""


class TestAutoencoderKLLTX2VideoSlicingTiling(AutoencoderKLLTX2VideoTesterConfig, AutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLLTX2Video."""


class _MockTemporalExpansionDecoder(torch.nn.Module):
    """Deterministic decoder stub with the causal temporal expansion of the real decoders.

    Latent frame 0 maps to video frame 0 and every subsequent latent frame maps to
    `temporal_compression_ratio` video frames. Because each output frame depends only on its
    latent frame, temporally tiled decoding must reproduce full decoding exactly, which makes
    frame misalignment detectable without pretrained weights.
    """

    def __init__(self, temporal_compression_ratio: int):
        super().__init__()
        self.temporal_compression_ratio = temporal_compression_ratio

    def forward(self, x: torch.Tensor, temb=None, causal=None) -> torch.Tensor:
        out_num_frames = (x.shape[2] - 1) * self.temporal_compression_ratio + 1
        x = x[:, :3]
        out = x.new_zeros((x.shape[0], 3, out_num_frames, *x.shape[-2:]))
        out[:, :, 0] = x[:, :, 0]
        out[:, :, 1:] = x[:, :, 1:].repeat_interleave(self.temporal_compression_ratio, dim=2)
        return out


class TestAutoencoderKLLTX2VideoFramewiseDecode:
    """Tests for temporally tiled decoding alignment."""

    def test_framewise_decode_matches_full_decode(self):
        torch.manual_seed(0)
        vae = AutoencoderKLLTX2Video(
            in_channels=3,
            out_channels=3,
            latent_channels=3,
            block_out_channels=(8, 8, 8, 8),
            decoder_block_out_channels=(8, 16, 32),
            layers_per_block=(1, 1, 1, 1, 1),
            decoder_layers_per_block=(1, 1, 1, 1),
            spatio_temporal_scaling=(True, True, False, False),
            decoder_spatio_temporal_scaling=(True, True, False),
            decoder_inject_noise=(False, False, False, False),
            downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            upsample_residual=(False, False, False),
            upsample_factor=(1, 1, 1),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
            encoder_spatial_padding_mode="zeros",
            decoder_spatial_padding_mode="zeros",
        )
        vae.decoder = _MockTemporalExpansionDecoder(vae.temporal_compression_ratio)

        # Enough latent frames so that several temporal tiles overlap.
        z = torch.randn(1, vae.config.latent_channels, 7, 8, 8)

        with torch.no_grad():
            full = vae.decode(z).sample
            vae.use_framewise_decoding = True
            tiled = vae.decode(z).sample

        assert tiled.shape == full.shape
        torch.testing.assert_close(tiled, full, atol=1e-6, rtol=1e-6)
