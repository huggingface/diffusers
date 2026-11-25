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

import torch

from diffusers import AutoencoderKLQwenImage21
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLQwenImage21TesterConfig(BaseModelTesterConfig):
    # Four spatial downsampling stages, so the latents are 16x smaller per axis. The condition and target images
    # carry an alpha channel, hence four input channels.
    num_channels = 4
    spatial_compression_ratio = 16
    sizes = (96, 96)

    @property
    def model_class(self):
        return AutoencoderKLQwenImage21

    @property
    def output_shape(self):
        return (self.num_channels, 1, *self.sizes)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "base_dim": 4,
            "decoder_base_dim": 4,
            "z_dim": 4,
            "dim_mult": [1, 1, 1, 1, 1],
            "num_res_blocks": 1,
            "attn_scales": [],
            "temperal_downsample": [False, True, True, True],
            "latents_mean": [0.0] * 4,
            "latents_std": [1.0] * 4,
        }

    def get_dummy_inputs(self):
        image = randn_tensor((1, self.num_channels, 1, *self.sizes), generator=self.generator, device=torch_device)
        return {"sample": image}


class TestAutoencoderKLQwenImage21(AutoencoderKLQwenImage21TesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    def test_spatial_compression_ratio_matches_architecture(self):
        """
        `scale_factor_spatial` drives every tile-to-latent conversion, so it has to be the ratio the encoder
        actually applies — one downsample per `dim_mult` stage after the first.
        """
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device).eval()
        with torch.no_grad():
            latent = model.encode(self.get_dummy_inputs()["sample"]).latent_dist.mode()

        expected = 2 ** (len(init_dict["dim_mult"]) - 1)
        assert model.spatial_compression_ratio == expected
        assert latent.shape[-1] == self.sizes[-1] // expected
        assert latent.shape[-2] == self.sizes[-2] // expected

    def test_tiled_encode_keeps_the_latent_shape(self):
        """
        Every tile-to-latent conversion divides by `scale_factor_spatial`, so a wrong ratio silently changes the
        shape of a tiled encode: 2048x2048 came out as a 168x168 latent instead of 128x128. The tile sizes are
        lowered here so the tiled path actually runs on an input this small.

        Only the shape is asserted. Tile values differ from a single pass because the causal convolutions carry a
        feature cache that each tile starts fresh, which is a property of the tiling implementation itself.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = self.get_dummy_inputs()["sample"]

        with torch.no_grad():
            untiled = model.encode(sample).latent_dist.mode()
            model.enable_tiling(
                tile_sample_min_height=48,
                tile_sample_min_width=48,
                tile_sample_stride_height=32,
                tile_sample_stride_width=32,
            )
            tiled = model.encode(sample).latent_dist.mode()

        assert tiled.shape == untiled.shape


class TestAutoencoderKLQwenImage21SlicingTiling(AutoencoderKLQwenImage21TesterConfig, AutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLQwenImage21."""
