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


class TestLTX2VideoDiffusionDecoderModelTileSchedule(LTX2VideoDiffusionDecoderModelTesterConfig):
    """Where a tiled decode cuts, independently of anything decoding it.

    The pipeline walks this schedule; the arithmetic in it is the decoder's own -- cell-to-pixel scales, the
    causal frame mapping, the ghost frames NATTEN's border shift leaves on the end. Pinning it here rather than
    only through a decode means a mistake reads as a wrong number instead of a wrong picture.
    """

    TILES = {
        "tile_sample_min_num_frames": 8,
        "tile_sample_stride_num_frames": 6,
        "tile_sample_min_height": 32,
        "tile_sample_stride_height": 24,
        "tile_sample_min_width": 32,
        "tile_sample_stride_width": 24,
    }

    def get_schedule(self, feature_shape=(11, 16, 20), **tiles):
        model = self.model_class(**self.get_init_dict())
        model.enable_tiling(**{**self.TILES, **tiles})
        return model.get_tile_schedule(feature_shape)

    def test_cells_map_to_pixels_by_the_last_upsample_and_the_patch_size(self):
        """A cell is the last upsample's stride times the diffusion stage's patch size: (2, 2, 2) x 2 here."""
        schedule = self.get_schedule()
        assert schedule.scales == (2, 4, 4)
        # 8 frames / 32 px tiles over those scales, with 6 / 24 strides, so the overlap is 1 cell each way.
        assert schedule.cell_strides == (3, 6, 6)
        assert schedule.blend == (2, 8, 8)

    def test_ghost_frames_are_excluded_from_the_cut_but_kept_for_the_last_tile(self):
        """The border-shift padding is real signal for the final tile's attention and nothing else.

        With a kernel of 3 the decoder pads 2 latent frames, and the earlier temporal upsamples (strides 1, 2, 2)
        carry them to 8 cells -- so an 11-cell feature volume holds 3 cells of video.
        """
        schedule = self.get_schedule(feature_shape=(11, 16, 20))
        assert (schedule.total_frames, schedule.num_frames) == (11, 3)
        # Only the tile ending at the last real cell reaches past it, and it reaches all the way.
        assert schedule.feature_end(schedule.num_frames) == 11
        assert schedule.feature_end(2) == 2

    def test_the_causal_frame_mapping_places_tiles_without_gaps_or_overlap(self):
        """The origin cell decodes to one frame, every later cell to `scale_t`, so tile starts are offset by one.

        This is the arithmetic a tiled decode is most easily wrong about: an off-by-one here still produces a
        full-sized video, just one sampled from the wrong slice of the noise canvas.
        """
        schedule = self.get_schedule(feature_shape=(20, 16, 20), tile_sample_min_num_frames=8)
        assert len(schedule.temporal) > 1, "need a real temporal split for this to mean anything"
        scale_t = schedule.scales[0]

        assert schedule.pixel_origin(0) == 0
        for t0, _ in schedule.temporal[1:]:
            # A non-origin tile keeps the upsample's duplicate leading frame, so it starts one frame early.
            assert schedule.pixel_origin(t0) == t0 * scale_t - 1
        # Each group contributes exactly the frames the next one starts after.
        for index, (t0, _) in enumerate(schedule.temporal[:-1]):
            kept = schedule.pixel_frames(schedule.cell_strides[0], is_origin=index == 0)
            assert schedule.pixel_origin(t0) + kept == schedule.pixel_origin(schedule.temporal[index + 1][0])

    def test_a_short_trailing_remnant_is_merged_into_its_neighbour(self):
        """Neighborhood attention rejects a grid smaller than its kernel, so a stub tile cannot stand alone."""
        # A stride of 6 cells over 20 would start a final tile at 18, leaving 2 cells -- under the kernel of 3.
        schedule = self.get_schedule(feature_shape=(28, 16, 20), tile_sample_min_num_frames=8)
        assert all(end - start >= 3 for start, end in schedule.temporal), schedule.temporal
        assert schedule.temporal[-1][1] == schedule.num_frames, "the tiles must still cover the whole video"

    def test_tiles_cover_every_axis_end_to_end(self):
        schedule = self.get_schedule(feature_shape=(20, 30, 40))
        for axis, tiles, length in (
            ("t", schedule.temporal, schedule.num_frames),
            ("h", schedule.height, 30),
            ("w", schedule.width, 40),
        ):
            assert tiles[0][0] == 0 and tiles[-1][1] == length, (axis, tiles)
            for (_, prev_end), (next_start, _) in zip(tiles, tiles[1:]):
                assert next_start < prev_end, f"{axis} tiles leave a gap: {tiles}"


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
