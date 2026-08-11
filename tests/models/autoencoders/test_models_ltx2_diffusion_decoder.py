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

from diffusers import LTX2VideoDiffusionDecoderModel
from diffusers.models.autoencoders import ltx2_diffusion_decoder
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor
from diffusers.utils import is_kernels_available
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, require_torch_gpu, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
)


enable_full_determinism()


class LTX2VideoDiffusionDecoderModelTesterConfig(BaseModelTesterConfig):
    """Tiny config for the LTX-2.5 diffusion decoder.

    The decoder's neighborhood attention needs every stage to be at least its kernel size in T/H/W, which
    sets the floor on the dummy input: with a kernel of 3, a compression of 16x spatial / 8x temporal and
    the production stride pattern, the smallest usable latent is 2x3x3, i.e. a 9x48x48 video.
    """

    @property
    def main_input_name(self):
        return "z"

    @property
    def model_class(self):
        return LTX2VideoDiffusionDecoderModel

    @property
    def output_shape(self):
        return (3, 9, 48, 48)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "out_channels": 3,
            "latent_channels": 8,
            "patch_size": 2,
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
        # The decoder takes latents directly now: 2 latent frames decode to 9 pixel frames.
        latents = randn_tensor((2, 8, 2, 3, 3), generator=self.generator, device=torch_device)
        # The decoder denoises, so it draws noise on every call: without a seeded generator no two forward
        # passes agree and every output comparison below would be meaningless.
        return {"z": latents, "generator": self.generator}


class TestLTX2VideoDiffusionDecoderModel(LTX2VideoDiffusionDecoderModelTesterConfig, ModelTesterMixin):
    base_precision = 1e-2


class TestLTX2VideoDiffusionDecoderModelSwiGLUTiling(LTX2VideoDiffusionDecoderModelTesterConfig):
    """The SwiGLU evaluates in token tiles to bound decode memory; that must not change the result."""

    def test_token_tiled_swiglu_matches_untiled(self):
        """Force the tiled path at test scale and require bit-identical output.

        The dummy video is 9x48x48, so its stage-5 grid is 5184 tokens -- an order of magnitude under the
        16384-token tile size, which means every other test in this file exercises only the untiled
        branch. Shrinking the tile size is what actually covers the loop.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        latent = inputs["z"]

        def decode():
            # Re-seed per call: the decoder samples the noise it denoises, so a shared generator would
            # hand the second call different noise and the comparison would be vacuous.
            generator = torch.Generator(device=torch_device).manual_seed(0)
            with torch.no_grad():
                return model.decode(latent, generator=generator, return_dict=False)[0]

        original = ltx2_diffusion_decoder._SWIGLU_TILE_SIZE
        try:
            ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = 10**9  # larger than the volume
            untiled = decode()
            ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = 128  # ~41 tiles at this size
            tiled = decode()
        finally:
            ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = original

        assert tiled.shape == untiled.shape
        # Exact, not approximate: the MLP is pointwise across tokens, so tiling changes only how many
        # hidden-width elements are live, never a value.
        assert torch.equal(tiled, untiled), (
            f"tiled SwiGLU diverged from untiled by {(tiled - untiled).abs().max().item():.3e}"
        )


class TestLTX2VideoDiffusionDecoderModelMemory(LTX2VideoDiffusionDecoderModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for LTX2VideoDiffusionDecoderModel."""


class TestLTX2VideoDiffusionDecoderModelAttention(LTX2VideoDiffusionDecoderModelTesterConfig, AttentionTesterMixin):
    """Attention processor tests for LTX2VideoDiffusionDecoderModel."""


@require_torch_gpu
@pytest.mark.skipif(not is_kernels_available(), reason="Fetching NATTEN from the Hub requires the `kernels` package.")
class TestLTX2VideoDiffusionDecoderModelNattenProcessor(LTX2VideoDiffusionDecoderModelTesterConfig):
    """The NATTEN processor is the reference decoder's attention path; it must agree with the default flex path.

    CUDA-only twice over: NATTEN has no CPU kernels, and the processor fetches its build from the Hub
    (`shi-labs/natten`) through `kernels`, which resolves a variant for the running torch/CUDA.
    """

    def test_natten_processor_decodes(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()

        # One shared instance swaps every attention module: the decoder's attention is homogeneous, with
        # per-stage differences (the kernel size) living on the module rather than the processor.
        model.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
        processors = model.attn_processors
        assert processors and all(
            isinstance(processor, LTX2VideoVaeNeighborhoodNattenProcessor) for processor in processors.values()
        )

        with torch.no_grad():
            output = model.decode(inputs["z"], generator=inputs["generator"], return_dict=False)[0]

        assert output.shape == (inputs["z"].shape[0], *self.output_shape)
        assert torch.isfinite(output).all(), "NATTEN decode produced NaN/inf values"
