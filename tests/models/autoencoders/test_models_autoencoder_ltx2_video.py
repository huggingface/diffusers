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

    @pytest.mark.parametrize("height,width,tile_height,tile_width", [(22, 40, 9, 16), (40, 22, 16, 9)])
    def test_tiled_decode_skips_overlap_only_tail(self, height, width, tile_height, tile_width):
        config = self.get_init_dict()
        config.update(spatial_compression_ratio=8, decoder_spatial_padding_mode="reflect")
        torch.manual_seed(0)
        model = self.model_class(**config).to(torch_device).eval()
        model.enable_tiling(
            tile_sample_min_height=tile_height * 8,
            tile_sample_min_width=tile_width * 8,
            tile_sample_stride_height=(tile_height - 2) * 8,
            tile_sample_stride_width=(tile_width - 2) * 8,
        )
        latents = randn_tensor(
            (1, config["latent_channels"], 1, height, width), generator=self.generator, device=torch_device
        )
        with torch.no_grad():
            video = model.decode(latents).sample
        assert video.shape == (1, 3, 1, height * 8, width * 8)
        assert torch.isfinite(video).all()
