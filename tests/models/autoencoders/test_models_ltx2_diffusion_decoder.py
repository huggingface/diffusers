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
        return "hidden_states"

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
        """One denoising step's worth of input: noised pixels, the context conditioning them, and a noise level.

        The latent this corresponds to is `(2, 8, 2, 3, 3)`, which decodes to 9 pixel frames of 48x48. The context
        shares the diffusion stage's token grid, so it is that canvas divided by `patch_size` with the last stage's
        channel count. Building it directly rather than by running the context stages keeps this a pure input
        fixture -- `LTX2VideoDiffusionDecodePipeline` is where the two are wired together.
        """
        hidden_states = randn_tensor((2, 3, 9, 48, 48), generator=self.generator, device=torch_device)
        latent_context = randn_tensor((2, 9, 24, 24, 16), generator=self.generator, device=torch_device)
        timestep = torch.full((2,), 0.5, device=torch_device)
        return {"hidden_states": hidden_states, "latent_context": latent_context, "timestep": timestep}

    def get_dummy_latents(self):
        """A latent to feed the context stages, matching the canvas `get_dummy_inputs` describes."""
        return randn_tensor((2, 8, 2, 3, 3), generator=self.generator, device=torch_device)

    def encode_context(self, model, latents):
        """The two deterministic halves, as `LTX2VideoDiffusionDecodePipeline` runs them."""
        return model.encode_context_stage_4(model.encode_context_stages_1_to_3(latents))


class TestLTX2VideoDiffusionDecoderModel(LTX2VideoDiffusionDecoderModelTesterConfig, ModelTesterMixin):
    base_precision = 1e-2


class TestLTX2VideoDiffusionDecoderModelSwiGLUTiling(LTX2VideoDiffusionDecoderModelTesterConfig):
    """The SwiGLU evaluates in token tiles to bound decode memory; that must not change the result."""

    def test_token_tiled_swiglu_matches_untiled(self):
        """Force the tiled path at test scale and require near-identical output.

        The dummy video is 9x48x48, so its stage-5 grid is 5184 tokens -- an order of magnitude under the
        16384-token tile size, which means every other test in this file exercises only the untiled
        branch. Shrinking the tile size is what actually covers the loop.

        The comparison is a tight `allclose`, not `torch.equal`: the MLP is pointwise across tokens, so
        tiling cannot change what is computed, but a matmul over a 128-token slice may reduce in a
        different order than the same rows inside the full-tensor call.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()

        def decode():
            # A fixed input rather than sampled noise: the point is that tiling the MLP changes nothing, so
            # both calls have to see the same tensors.
            with torch.no_grad():
                return model(**inputs, return_dict=False)[0]

        original = ltx2_diffusion_decoder._SWIGLU_TILE_SIZE
        try:
            ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = 10**9  # larger than the volume
            untiled = decode()
            ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = 128  # ~41 tiles at this size
            tiled = decode()
        finally:
            ltx2_diffusion_decoder._SWIGLU_TILE_SIZE = original

        assert tiled.shape == untiled.shape
        assert torch.allclose(tiled, untiled, rtol=1e-5, atol=1e-5), (
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

        # Through the context stages as well as the denoising step: the deterministic stages are where most
        # of the neighborhood attention lives, and they use a different kernel size per stage.
        latents = self.get_dummy_latents()
        with torch.no_grad():
            latent_context = self.encode_context(model, latents)
            output = model(inputs["hidden_states"], latent_context, inputs["timestep"], return_dict=False)[0]

        assert output.shape == (inputs["hidden_states"].shape[0], *self.output_shape)
        assert torch.isfinite(output).all(), "NATTEN decode produced NaN/inf values"
