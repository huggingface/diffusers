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

from diffusers import AutoencoderKLLTX2VideoDiffusionDecoder
from diffusers.models.autoencoders import autoencoder_kl_ltx2_diffusion_decoder
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
)
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLLTX2VideoDiffusionDecoderTesterConfig(BaseModelTesterConfig):
    """Tiny config for the LTX-2.4 diffusion-decoder VAE.

    The decoder's neighborhood attention needs every stage to be at least its kernel size in T/H/W, which
    sets the floor on the dummy input: with a kernel of 3, a compression of 16x spatial / 8x temporal and
    the production stride pattern, the smallest usable latent is 2x3x3, i.e. a 9x48x48 video.
    """

    @property
    def main_input_name(self):
        return "sample"

    @property
    def model_class(self):
        return AutoencoderKLLTX2VideoDiffusionDecoder

    @property
    def output_shape(self):
        return (3, 9, 48, 48)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "layers_per_block": (1, 1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, True, True),
            "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            "patch_size": 2,
            "patch_size_t": 1,
            "encoder_causal": True,
            "encoder_spatial_padding_mode": "zeros",
            "decoder_head_dim": 16,
            "decoder_stage_channels": (64, 32, 16, 16, 16),
            "decoder_stage_depths": (1, 1, 1, 1, 2),
            "decoder_stage_kernels": ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
            "decoder_upsample_strides": ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
            "decoder_upsample_channel_reductions": (2, 2, 1, 1),
            "decoder_stage5_kernel": (3, 3, 3),
            "decoder_t_emb_dim": 32,
            "spatial_compression_ratio": 16,
            "temporal_compression_ratio": 8,
        }

    def get_dummy_inputs(self):
        video = randn_tensor((2, 3, 9, 48, 48), generator=self.generator, device=torch_device)
        # The decoder denoises, so it draws noise on every call: without a seeded generator no two forward
        # passes agree and every output comparison below would be meaningless.
        return {"sample": video, "generator": self.generator}


class TestAutoencoderKLLTX2VideoDiffusionDecoder(AutoencoderKLLTX2VideoDiffusionDecoderTesterConfig, ModelTesterMixin):
    base_precision = 1e-2


class TestAutoencoderKLLTX2VideoDiffusionDecoderSwiGLUTiling(AutoencoderKLLTX2VideoDiffusionDecoderTesterConfig):
    """The SwiGLU evaluates in token tiles to bound decode memory; that must not change the result."""

    def test_token_tiled_swiglu_matches_untiled(self):
        """Force the tiled path at test scale and require bit-identical output.

        The dummy video is 9x48x48, so its stage-5 grid is 5184 tokens -- an order of magnitude under the
        16384-token tile size, which means every other test in this file exercises only the untiled
        branch. Shrinking the tile size is what actually covers the loop.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        latent = model.encode(inputs["sample"]).latent_dist.mode()

        def decode():
            # Re-seed per call: the decoder samples the noise it denoises, so a shared generator would
            # hand the second call different noise and the comparison would be vacuous.
            generator = torch.Generator(device=torch_device).manual_seed(0)
            with torch.no_grad():
                return model.decode(latent, generator=generator, return_dict=False)[0]

        original = autoencoder_kl_ltx2_diffusion_decoder._SWIGLU_TILE_SIZE
        try:
            autoencoder_kl_ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = 10**9  # larger than the volume
            untiled = decode()
            autoencoder_kl_ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = 128  # ~41 tiles at this size
            tiled = decode()
        finally:
            autoencoder_kl_ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = original

        assert tiled.shape == untiled.shape
        # Exact, not approximate: the MLP is pointwise across tokens, so tiling changes only how many
        # hidden-width elements are live, never a value.
        assert torch.equal(tiled, untiled), (
            f"tiled SwiGLU diverged from untiled by {(tiled - untiled).abs().max().item():.3e}"
        )


class TestAutoencoderKLLTX2VideoDiffusionDecoderMemory(
    AutoencoderKLLTX2VideoDiffusionDecoderTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests for AutoencoderKLLTX2VideoDiffusionDecoder."""


class TestAutoencoderKLLTX2VideoDiffusionDecoderAttention(
    AutoencoderKLLTX2VideoDiffusionDecoderTesterConfig, AttentionTesterMixin
):
    """Attention processor tests for AutoencoderKLLTX2VideoDiffusionDecoder."""


class TestAutoencoderKLLTX2VideoDiffusionDecoderSlicingTiling(
    AutoencoderKLLTX2VideoDiffusionDecoderTesterConfig, NewAutoencoderTesterMixin
):
    """Slicing tests for AutoencoderKLLTX2VideoDiffusionDecoder; tiling is not supported."""

    def test_enable_disable_tiling(self):
        model = self.model_class(**self.get_init_dict())
        with pytest.raises(NotImplementedError, match="Tiled decoding is not supported"):
            model.enable_tiling()
