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

from ...testing_utils import enable_full_determinism, require_accelerator, require_torch_gpu, torch_device
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

    @pytest.mark.skip(
        "`forward` runs through the `apply_forward_hook`-decorated `decode`, and that decorator's "
        "`pre_forward` call clears the input device accelerate's `AlignDevicesHook` recorded for the caller, so the "
        "output comes back on the last device of the split rather than the input device and the comparison raises. "
        "`test_cpu_offload` covers split placement instead — there every submodule executes on the same device."
    )
    def test_model_parallelism(self, base_model_output, tmp_path, atol=1e-5, rtol=0):
        pass


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
        assert torch.allclose(tiled, untiled, rtol=1e-5, atol=1e-5), (
            f"tiled SwiGLU diverged from untiled by {(tiled - untiled).abs().max().item():.3e}"
        )


class TestLTX2VideoDiffusionDecoderModelTiling(LTX2VideoDiffusionDecoderModelTesterConfig):
    """Tiled decoding: the early stages run on the full latent, stages 4-5 run per tile with blending.

    The latent is 3x4x5 (17x64x80 pixels) so every axis is large enough to split: the tiling grid — the
    stage-4 input grid — is 9x16x20, and the tile sizes below cut it into three temporal and two/three
    spatial tiles.
    """

    def get_latent(self):
        return randn_tensor((1, 8, 3, 4, 5), generator=self.generator, device=torch_device)

    def decode(self, model, latent, num_inference_steps=None):
        # Re-seed per call: the decoder samples the noise it denoises, so outputs are only comparable
        # across calls that drew from the same generator state.
        generator = torch.Generator("cpu").manual_seed(0)
        with torch.no_grad():
            return model.decode(latent, generator=generator, num_inference_steps=num_inference_steps)[0]

    @require_accelerator
    def test_tiles_covering_the_video_match_untiled_exactly(self):
        """A tile schedule with a single covering tile must reproduce the untiled decode bit for bit.

        This pins the per-tile plumbing — the ghost-frame carry/crop, the leading-frame drop, and the
        stitching — because any offset in them shifts the single tile's output relative to the untiled path.
        The default tile sizes are larger than the test video, so `tiled_decode` builds exactly one tile.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        latent = self.get_latent()

        for num_inference_steps in (None, 3):  # None: the single-step x0 shortcut; 3: the Euler loop
            untiled = self.decode(model, latent, num_inference_steps)
            generator = torch.Generator("cpu").manual_seed(0)
            with torch.no_grad():
                tiled = model.tiled_decode(latent, generator=generator, num_inference_steps=num_inference_steps)
            assert torch.equal(tiled, untiled), (
                f"single-tile tiled decode diverged from untiled by {(tiled - untiled).abs().max().item():.3e} "
                f"with num_inference_steps={num_inference_steps}"
            )

    def test_tiled_decode_with_splits(self):
        """Actually-split tiles must reassemble to the untiled output shape, on both noise paths.

        Values legitimately differ from the untiled decode (each tile sees a truncated attention context at
        its borders), so this asserts geometry, not closeness. The multi-step run additionally covers the
        shared noise canvas that overlapping tiles slice from.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        latent = self.get_latent()
        untiled = self.decode(model, latent)

        model.enable_tiling(
            # Tiling-grid cells are 2 frames x 4 px x 4 px here (last upsample stride (2, 2, 2), patch 2), so
            # this is a 4-cell tile with a 3-cell stride temporally and 8x8-cell tiles with 6-cell strides
            # spatially: tiles (0, 4), (3, 7), (6, 9) over T and (0, 8), (6, 16|20) over H/W.
            tile_sample_min_num_frames=8,
            tile_sample_stride_num_frames=6,
            tile_sample_min_height=32,
            tile_sample_stride_height=24,
            tile_sample_min_width=32,
            tile_sample_stride_width=24,
        )
        for num_inference_steps in (None, 3):
            tiled = self.decode(model, latent, num_inference_steps)
            assert tiled.shape == untiled.shape
            assert torch.isfinite(tiled).all()

        model.disable_tiling()
        assert torch.equal(self.decode(model, latent), untiled)


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


def _dense_joint_attention(query, key, value, keyframe_query, keyframe_key, keyframe_value, keyframe_times, kernel):
    """Joint attention written out as one dense masked softmax over every video and plane token.

    An independent statement of the visibility rule `_joint_neighborhood_attention` implements with query bricks: a
    centered window clamped (not shifted) at the volume border, the same spatial window on the two planes nearest a
    frame, and the same spatial window on the two frames nearest a plane.
    """
    batch_size, num_frames, height, width, heads, head_dim = query.shape
    num_planes = keyframe_query.shape[1]
    lo = [k // 2 for k in kernel]
    hi = [k - l - 1 for k, l in zip(kernel, lo)]
    frame_times = torch.arange(num_frames, dtype=torch.float32)
    video_slots = torch.argsort((frame_times[:, None] - keyframe_times[None]).abs(), dim=-1, stable=True)[:, :2]
    plane_slots = torch.argsort((keyframe_times[:, None] - frame_times[None]).abs(), dim=-1, stable=True)[:, :2]

    def grid(length):
        t, h, w = torch.meshgrid(torch.arange(length), torch.arange(height), torch.arange(width), indexing="ij")
        return t.reshape(-1), h.reshape(-1), w.reshape(-1)

    vt, vh, vw = grid(num_frames)
    pp, ph, pw = grid(num_planes)
    is_plane = torch.cat([torch.zeros_like(vt, dtype=torch.bool), torch.ones_like(pp, dtype=torch.bool)])
    axis0, hh, ww = torch.cat([vt, pp]), torch.cat([vh, ph]), torch.cat([vw, pw])
    dh, dw = hh[None] - hh[:, None], ww[None] - ww[:, None]
    spatial = (dh >= -lo[1]) & (dh <= hi[1]) & (dw >= -lo[2]) & (dw <= hi[2])
    dt = axis0[None] - axis0[:, None]
    video_video = ~is_plane[:, None] & ~is_plane[None] & (dt >= -lo[0]) & (dt <= hi[0])
    query_slots = torch.where(is_plane, 0, axis0.clamp(max=num_frames - 1))
    video_plane = (
        ~is_plane[:, None] & is_plane[None] & (video_slots[query_slots][:, None, :] == axis0[None, :, None]).any(-1)
    )
    plane_rows = torch.where(is_plane, axis0.clamp(max=num_planes - 1), 0)
    plane_video = (
        is_plane[:, None] & ~is_plane[None] & (plane_slots[plane_rows][:, None, :] == axis0[None, :, None]).any(-1)
    )
    plane_plane = is_plane[:, None] & is_plane[None] & (dt == 0)
    visible = spatial & (video_video | video_plane | plane_video | plane_plane)

    def tokens(video, planes):
        flat = torch.cat(
            [video.reshape(batch_size, -1, heads, head_dim), planes.reshape(batch_size, -1, heads, head_dim)], 1
        )
        return flat.transpose(1, 2)

    scores = tokens(query, keyframe_query) @ tokens(key, keyframe_key).transpose(-1, -2)
    attended = scores.masked_fill(~visible, float("-inf")).softmax(-1) @ tokens(value, keyframe_value)
    attended = attended.transpose(1, 2)
    num_video = num_frames * height * width
    return (
        attended[:, :num_video].reshape(query.shape),
        attended[:, num_video:].reshape(keyframe_query.shape),
    )


class TestLTX2VideoDiffusionDecoderModelKeyframes(LTX2VideoDiffusionDecoderModelTesterConfig):
    """Keyframe-aware decoding: a second stream of single-frame planes, attended jointly with the video."""

    def get_model(self, keyframe_type_embedding=True):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict(), decoder_keyframe_type_embedding=keyframe_type_embedding)
        with torch.no_grad():
            # Random, not default-initialized: the zero `type_emb` and `scale_shift_table` would hide wiring errors.
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=self.generator) * 0.3)
        return model.to(torch_device).eval()

    def get_latents(self, num_planes=2, latent_frames=3):
        generator = self.generator
        latents = randn_tensor((1, 8, latent_frames, 3, 4), generator=generator, device=torch_device)
        keyframes = randn_tensor((1, 8, num_planes, 3, 4), generator=generator, device=torch_device)
        return latents, keyframes

    def decode(self, model, latents, **kwargs):
        generator = torch.Generator("cpu").manual_seed(0)
        with torch.no_grad():
            return model.decode(latents, generator=generator, return_dict=False, **kwargs)[0]

    def test_type_emb_is_opt_in(self):
        default = self.model_class(**self.get_init_dict())
        assert default.decoder.type_emb is None
        assert "decoder.type_emb" not in default.state_dict()

        model = self.model_class(**self.get_init_dict(), decoder_keyframe_type_embedding=True)
        assert model.state_dict()["decoder.type_emb"].shape == (self.get_init_dict()["latent_channels"],)
        assert torch.equal(model.decoder.type_emb, torch.zeros_like(model.decoder.type_emb))

    def test_type_emb_save_load_round_trip(self, tmp_path):
        model = self.get_model()
        model.save_pretrained(tmp_path / "keyframes")
        loaded, info = self.model_class.from_pretrained(tmp_path / "keyframes", output_loading_info=True)
        assert loaded.config.decoder_keyframe_type_embedding
        assert not info["missing_keys"] and not info["unexpected_keys"]
        assert torch.equal(loaded.decoder.type_emb.to(torch_device), model.decoder.type_emb)

        # A checkpoint without the tag keeps loading cleanly with the default config.
        self.get_model(keyframe_type_embedding=False).save_pretrained(tmp_path / "plain")
        loaded, info = self.model_class.from_pretrained(tmp_path / "plain", output_loading_info=True)
        assert loaded.decoder.type_emb is None
        assert not info["missing_keys"] and not info["unexpected_keys"]

    def test_plain_decode_is_unchanged(self):
        """Without planes the decode must not depend on whether the model carries a tag, nor on the new arguments."""
        latents, _ = self.get_latents()
        model = self.get_model()
        plain = self.decode(model, latents)
        assert torch.equal(self.decode(model, latents, keyframe_latents=None, keyframe_frame_indices=None), plain)

        untagged = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        state_dict = {k: v for k, v in model.state_dict().items() if k != "decoder.type_emb"}
        untagged.load_state_dict(state_dict, strict=True)
        assert torch.equal(self.decode(untagged, latents), plain)

    def test_keyframe_decode_depends_on_planes_and_tag(self):
        latents, keyframes = self.get_latents()
        model = self.get_model()
        indices = torch.tensor([8, 16])
        plain = self.decode(model, latents)
        with_planes = self.decode(model, latents, keyframe_latents=keyframes, keyframe_frame_indices=indices)
        assert with_planes.shape == plain.shape
        assert torch.isfinite(with_planes).all()
        assert not torch.allclose(with_planes, plain)
        assert torch.equal(
            self.decode(model, latents, keyframe_latents=keyframes, keyframe_frame_indices=indices), with_planes
        )

        other = self.decode(model, latents, keyframe_latents=keyframes.flip(2), keyframe_frame_indices=indices)
        assert not torch.allclose(other, with_planes)

        with torch.no_grad():
            model.decoder.type_emb.add_(1.0)
        retagged = self.decode(model, latents, keyframe_latents=keyframes, keyframe_frame_indices=indices)
        assert not torch.allclose(retagged, with_planes)

    def test_planes_outside_the_two_nearest_are_invisible(self):
        """Every video position attends to its two nearest planes only, so a third, farther plane cannot matter."""
        latents, keyframes = self.get_latents(num_planes=3)
        model = self.get_model()
        indices = torch.tensor([8, 16, 400])
        reference = self.decode(model, latents, keyframe_latents=keyframes, keyframe_frame_indices=indices)

        far = keyframes.clone()
        far[:, :, 2] = torch.randn_like(far[:, :, 2])
        assert torch.equal(
            self.decode(model, latents, keyframe_latents=far, keyframe_frame_indices=indices), reference
        )

        near = keyframes.clone()
        near[:, :, 0] = torch.randn_like(near[:, :, 0])
        assert not torch.allclose(
            self.decode(model, latents, keyframe_latents=near, keyframe_frame_indices=indices), reference
        )

    @pytest.mark.parametrize(
        "num_frames, num_planes, kernel",
        [(5, 2, (3, 3, 3)), (3, 1, (3, 5, 5)), (2, 3, (5, 3, 7))],
    )
    def test_joint_attention_matches_dense_softmax(self, num_frames, num_planes, kernel):
        generator = torch.Generator().manual_seed(0)
        shape = (2, num_frames, 6, 7, 2, 16)
        query, key, value = (torch.randn(shape, generator=generator) for _ in range(3))
        plane_shape = (2, num_planes, 6, 7, 2, 16)
        keyframe_query, keyframe_key, keyframe_value = (
            torch.randn(plane_shape, generator=generator) for _ in range(3)
        )
        keyframe_times = torch.rand(num_planes, generator=generator) * (num_frames + 2) - 1

        ours = ltx2_diffusion_decoder._joint_neighborhood_attention(
            query, key, value, keyframe_query, keyframe_key, keyframe_value, keyframe_times, kernel
        )
        dense = _dense_joint_attention(
            query, key, value, keyframe_query, keyframe_key, keyframe_value, keyframe_times, kernel
        )
        for got, expected in zip(ours, dense):
            assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), (got - expected).abs().max()

    def test_keyframe_geometry(self):
        indices = torch.tensor([0, 8, 16])
        # t(0) = 0 and t(f) = (f + (r - 1) / 2) / r: the center of the cell holding frame f.
        assert ltx2_diffusion_decoder._keyframe_stage_times(indices, 8).tolist() == [0.0, 1.4375, 2.4375]
        assert ltx2_diffusion_decoder._keyframe_stage_times(indices, 2).tolist() == [0.0, 4.25, 8.25]
        assert ltx2_diffusion_decoder._keyframe_stage_times(indices, 1).tolist() == [0.0, 8.0, 16.0]

        # Nearest two by |dt|, ties to the lower index, `-1` when there are fewer candidates.
        slots = ltx2_diffusion_decoder._nearest_slots(torch.arange(4.0), torch.tensor([1.0, 3.0]), 2)
        assert slots.tolist() == [[0, 1], [0, 1], [0, 1], [1, 0]]
        slots = ltx2_diffusion_decoder._nearest_slots(torch.arange(2.0), torch.tensor([5.0]), 2)
        assert slots.tolist() == [[0, -1], [0, -1]]

        # A tile keeps its planes plus the nearest one on each side: [56, 64] has none inside, keeps 48 and 96.
        planes = torch.tensor([0, 48, 96, 144])
        assert ltx2_diffusion_decoder._keyframe_planes_for_tile(planes, 56, 64).tolist() == [False, True, True, False]
        assert ltx2_diffusion_decoder._keyframe_planes_for_tile(planes, 0, 100).tolist() == [True, True, True, True]
        assert ltx2_diffusion_decoder._keyframe_planes_for_tile(planes, 150, 200).tolist() == [
            False,
            False,
            False,
            True,
        ]

    def test_single_covering_tile_matches_untiled_with_keyframes(self):
        """One tile spanning the video must reproduce the untiled keyframe decode bit for bit."""
        latents, keyframes = self.get_latents(num_planes=3)
        model = self.get_model()
        indices = torch.tensor([8, 16, 40])  # 40 lies past the 17-frame clip: kept as the nearest plane after it
        for num_inference_steps in (None, 3):
            untiled = self.decode(
                model,
                latents,
                num_inference_steps=num_inference_steps,
                keyframe_latents=keyframes,
                keyframe_frame_indices=indices,
            )
            generator = torch.Generator("cpu").manual_seed(0)
            with torch.no_grad():
                tiled = model.tiled_decode(
                    latents,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    keyframe_latents=keyframes,
                    keyframe_frame_indices=indices,
                )
            assert torch.equal(tiled, untiled), (tiled - untiled).abs().max()

    def test_tiled_decode_with_splits_and_keyframes(self):
        latents, keyframes = self.get_latents(num_planes=2, latent_frames=4)
        model = self.get_model()
        indices = torch.tensor([8, 24])
        untiled = self.decode(model, latents, keyframe_latents=keyframes, keyframe_frame_indices=indices)
        # Cells are 2 frames x 4 px here: temporal tiles (0, 4), (3, 7), (6, 13) over the 13-cell grid, and two tiles
        # over each spatial axis.
        model.enable_tiling(
            tile_sample_min_num_frames=8,
            tile_sample_stride_num_frames=6,
            tile_sample_min_height=32,
            tile_sample_stride_height=24,
            tile_sample_min_width=40,
            tile_sample_stride_width=32,
        )
        for num_inference_steps in (None, 3):
            tiled = self.decode(
                model,
                latents,
                num_inference_steps=num_inference_steps,
                keyframe_latents=keyframes,
                keyframe_frame_indices=indices,
            )
            assert tiled.shape == untiled.shape
            assert torch.isfinite(tiled).all()
        model.disable_tiling()
        assert torch.equal(
            self.decode(model, latents, keyframe_latents=keyframes, keyframe_frame_indices=indices), untiled
        )

    def test_keyframe_inputs_are_validated(self):
        latents, keyframes = self.get_latents()
        model = self.get_model()
        with pytest.raises(ValueError, match="keyframe_frame_indices"):
            model.decode(latents, keyframe_latents=keyframes)
        with pytest.raises(ValueError, match="one pixel frame per plane"):
            model.decode(latents, keyframe_latents=keyframes, keyframe_frame_indices=[8])
        with pytest.raises(ValueError, match="non-negative"):
            model.decode(latents, keyframe_latents=keyframes, keyframe_frame_indices=[-8, 8])
        with pytest.raises(ValueError, match="must match"):
            model.decode(latents, keyframe_latents=keyframes[..., :2], keyframe_frame_indices=[8, 16])
        with pytest.raises(ValueError, match="at least one plane"):
            model.decode(latents, keyframe_latents=keyframes[:, :, :0], keyframe_frame_indices=[])
