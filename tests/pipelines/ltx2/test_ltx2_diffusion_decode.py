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

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    LTX2VideoDiffusionDecodePipeline,
    LTX2VideoDiffusionDecoderModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, require_accelerator, torch_device
from .testing_utils import get_dummy_vae


enable_full_determinism()


DECODER_CONFIG = {
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


def _build(with_vae: bool = False, **config_overrides):
    torch.manual_seed(0)
    decoder = LTX2VideoDiffusionDecoderModel(**{**DECODER_CONFIG, **config_overrides}).to(torch_device).eval()
    # Non-trivial statistics, so a run that skipped denormalization would not accidentally match.
    with torch.no_grad():
        decoder.latents_mean.copy_(torch.linspace(-0.1, 0.1, DECODER_CONFIG["latent_channels"]))
        decoder.latents_std.copy_(torch.linspace(0.5, 1.5, DECODER_CONFIG["latent_channels"]))

    vae = None
    if with_vae:
        torch.manual_seed(0)
        # Wider latents than the shared default, to match the decoder, and the causal decoder this pipeline
        # is exercised with.
        vae = (
            get_dummy_vae(latent_channels=DECODER_CONFIG["latent_channels"], decoder_causal=True)
            .to(torch_device)
            .eval()
        )

    return LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=_scheduler(), vae=vae)


def _scheduler():
    """The decoder's scheduler, as `convert_ltx2_to_diffusers.py` saves it: a plain uniform sigma walk."""
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
        shift_terminal=None,
        stochastic_sampling=False,
    )


def _latents():
    return torch.randn(1, 8, 2, 3, 3, generator=torch.Generator().manual_seed(1)).to(torch_device)


def test_decode_without_vae():
    """`vae` is optional: the pipeline must fall back to the decoder's own latent statistics."""
    pipe = _build(with_vae=False)
    assert pipe.vae is None
    frames = pipe(_latents(), generator=torch.Generator(torch_device).manual_seed(0), output_type="np").frames[0]
    assert frames.shape == (9, 48, 48, 3)


def test_decode_with_vae_uses_its_statistics():
    """When a `vae` is supplied its statistics are used instead of the decoder's."""
    latents = _latents()
    without = _build(with_vae=False)(
        latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt"
    ).frames
    with_vae = _build(with_vae=True)(
        latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt"
    ).frames
    # The dummy VAE's stats are mean 0 / std 1, the decoder's are not, so the two must disagree.
    assert not torch.equal(without, with_vae)


def test_decode_is_reproducible_with_a_generator():
    """The decoder samples the noise it denoises, so only a seeded generator makes it deterministic."""
    pipe, latents = _build(), _latents()
    first = pipe(latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt").frames
    same = pipe(latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt").frames
    other = pipe(latents, generator=torch.Generator(torch_device).manual_seed(7), output_type="pt").frames
    assert torch.equal(first, same)
    assert not torch.equal(first, other)


def test_denormalize_can_be_skipped():
    """`denormalize=False` must leave the latents alone for callers that already denormalized."""
    pipe, latents = _build(), _latents()
    normalized = pipe(latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt").frames
    raw = pipe(
        latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt", denormalize=False
    ).frames
    assert not torch.equal(normalized, raw)


def test_sigma_schedule_is_uniform():
    """The decoder walks `linspace(1, 1/n, n)`, not the scheduler's default `linspace(sigma_max, sigma_min, n)`.

    Nothing downstream would raise if the scheduler's default schedule were used instead -- it is the same length
    and the same shape -- so the schedule itself is what has to be pinned.
    """
    pipe = _build()
    assert pipe.get_sigmas(1) == [1.0]
    assert pipe.get_sigmas(4) == [1.0, 0.75, 0.5, 0.25]
    # The default comes from the checkpoint, i.e. what the decoder was distilled for.
    assert len(pipe.get_sigmas()) == pipe.diffusion_decoder.config.decoder_num_inference_steps


def test_num_inference_steps_and_sigmas_are_exclusive():
    pipe = _build()
    with pytest.raises(ValueError, match="Only one of"):
        pipe(_latents(), num_inference_steps=2, sigmas=[1.0, 0.5])


def test_multi_step_decode_runs_the_scheduler_loop():
    """More than one step must actually integrate: the extra steps have to change the result."""
    pipe, latents = _build(), _latents()
    one = pipe(
        latents, num_inference_steps=1, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt"
    ).frames
    three = pipe(
        latents, num_inference_steps=3, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt"
    ).frames
    assert one.shape == three.shape
    assert not torch.equal(one, three)
    assert torch.isfinite(three).all()


class TestTiling:
    """Tiled decoding: the early stages run on the full latent, the last stage and the diffusion loop run per tile.

    The latent is 3x4x5 (17x64x80 pixels) so every axis is large enough to split: the tiling grid -- the grid
    entering the last deterministic stage -- is 9x16x20, and the tile sizes below cut it into three temporal and
    two/three spatial tiles.
    """

    SPLIT_TILES = {
        # Tiling-grid cells are 2 frames x 4 px x 4 px here (last upsample stride (2, 2, 2), patch 2), so this is
        # a 4-cell tile with a 3-cell stride temporally and 8x8-cell tiles with 6-cell strides spatially: tiles
        # (0, 4), (3, 7), (6, 9) over T and (0, 8), (6, 16|20) over H/W.
        "tile_sample_min_num_frames": 8,
        "tile_sample_stride_num_frames": 6,
        "tile_sample_min_height": 32,
        "tile_sample_stride_height": 24,
        "tile_sample_min_width": 32,
        "tile_sample_stride_width": 24,
    }

    def latent(self):
        return torch.randn(1, 8, 3, 4, 5, generator=torch.Generator().manual_seed(2)).to(torch_device)

    def decode(self, pipe, latent, num_inference_steps=None):
        # Re-seed per call: the decoder samples the noise it denoises, so outputs are only comparable across
        # calls that drew from the same generator state.
        generator = torch.Generator("cpu").manual_seed(0)
        with torch.no_grad():
            return pipe.decode(latent, generator=generator, sigmas=pipe.get_sigmas(num_inference_steps))

    def test_tiles_covering_the_video_match_untiled_exactly(self):
        """A tile schedule with a single covering tile must reproduce the untiled decode bit for bit.

        This pins the per-tile plumbing -- the ghost-frame carry/crop, the leading-frame drop, and the stitching --
        because any offset in them shifts the single tile's output relative to the untiled path. The default tile
        sizes are larger than the test video, so `tiled_decode` builds exactly one tile.
        """
        pipe, latent = _build(), self.latent()

        for num_inference_steps in (None, 3):  # None: the single-step x0 shortcut; 3: the Euler loop
            untiled = self.decode(pipe, latent, num_inference_steps)
            generator = torch.Generator("cpu").manual_seed(0)
            with torch.no_grad():
                tiled = pipe.tiled_decode(latent, generator=generator, sigmas=pipe.get_sigmas(num_inference_steps))
            assert torch.equal(tiled, untiled), (
                f"single-tile tiled decode diverged from untiled by {(tiled - untiled).abs().max().item():.3e} "
                f"with num_inference_steps={num_inference_steps}"
            )

    def test_tiled_decode_with_splits(self):
        """Actually-split tiles must reassemble to the untiled output shape, on both noise paths.

        Values legitimately differ from the untiled decode (each tile sees a truncated attention context at its
        borders), so this asserts geometry, not closeness. The multi-step run additionally covers the shared noise
        canvas that overlapping tiles slice from.
        """
        pipe, latent = _build(), self.latent()
        untiled = self.decode(pipe, latent)

        pipe.diffusion_decoder.enable_tiling(**self.SPLIT_TILES)
        for num_inference_steps in (None, 3):
            tiled = self.decode(pipe, latent, num_inference_steps)
            assert tiled.shape == untiled.shape
            assert torch.isfinite(tiled).all()

        pipe.diffusion_decoder.disable_tiling()
        assert torch.equal(self.decode(pipe, latent), untiled)

    def test_tiled_decode_tiles_even_when_tiling_is_disabled(self):
        """`tiled_decode` tiles on its own terms; `use_tiling` only gates whether `decode` routes to it.

        `disable_tiling` flips the routing flag and leaves the configured tile sizes alone, so a direct
        `tiled_decode` call still has a split schedule to honor. This counts last-stage invocations rather than
        comparing outputs because output comparison cannot see the failure: a `tiled_decode` that quietly fell back
        to one full-grid tile would reproduce the untiled decode exactly and pass every other test in this class.
        """
        pipe, latent = _build(), self.latent()
        decoder = pipe.diffusion_decoder
        decoder.enable_tiling(**self.SPLIT_TILES)
        decoder.disable_tiling()
        assert not decoder.use_tiling

        # The last deterministic stage runs once per tile, so its call count is the tile count.
        stage_4_calls = []
        original_stage_4 = decoder.encode_context_stage_4

        def counting_stage_4(hidden_states, *args, **kwargs):
            stage_4_calls.append(tuple(hidden_states.shape[1:4]))
            return original_stage_4(hidden_states, *args, **kwargs)

        decoder.encode_context_stage_4 = counting_stage_4
        try:
            generator = torch.Generator("cpu").manual_seed(0)
            with torch.no_grad():
                pipe.tiled_decode(latent, generator=generator)
            tiled_call_count = len(stage_4_calls)

            stage_4_calls.clear()
            self.decode(pipe, latent)
            untiled_call_count = len(stage_4_calls)
        finally:
            del decoder.encode_context_stage_4

        assert tiled_call_count > 1, (
            f"tiled_decode ran the last stage {tiled_call_count} time(s) with use_tiling=False; it must tile "
            "regardless of the flag"
        )
        assert untiled_call_count == 1, (
            f"decode ran the last stage {untiled_call_count} times with use_tiling=False; it must not tile"
        )

    def test_decode_skips_tiling_for_a_video_that_fits_in_one_tile(self):
        """`decode` sizes the latent up before routing, so tiling only engages when it would split.

        The two outcomes are indistinguishable from the output alone: a video below the tile size that reaches
        `tiled_decode` anyway gets a single-tile schedule, which decodes to the same pixels. So this asserts the
        routing directly -- `tiled_decode` is never reached -- and separately pins the contract that matters to
        callers, that turning tiling on cannot change a small video's output.
        """
        pipe, latent = _build(), self.latent()
        decoder = pipe.diffusion_decoder
        untiled = self.decode(pipe, latent)

        calls = []
        original_tiled_decode = pipe.tiled_decode

        def counting_tiled_decode(*args, **kwargs):
            calls.append(1)
            return original_tiled_decode(*args, **kwargs)

        pipe.tiled_decode = counting_tiled_decode
        try:
            # Default tile sizes are far larger than this 17x64x80 video, so the gate declines to tile.
            decoder.enable_tiling()
            fits_in_one_tile = self.decode(pipe, latent)
            assert not calls, "decode routed to tiled_decode for a video that fits in a single tile"
            assert torch.equal(fits_in_one_tile, untiled), (
                "enabling tiling changed the output of a video below the tile size by "
                f"{(fits_in_one_tile - untiled).abs().max().item():.3e}"
            )

            # Shrink the tiles below the video and the same latent must now route.
            decoder.enable_tiling(**self.SPLIT_TILES)
            self.decode(pipe, latent)
            assert calls, "decode did not route to tiled_decode for a video larger than the tile size"
        finally:
            del pipe.tiled_decode
            decoder.disable_tiling()


@require_accelerator
def test_model_cpu_offload_decodes():
    """Offloading must survive the pipeline reaching into the decoder for its context stages.

    Accelerate's offload hook fires on `forward`, and this pipeline calls `encode_context_stages_1_to_3` and
    `encode_context_stage_4` before it ever calls one -- so those carry `@apply_forward_hook`. Without it the
    weights stay on the CPU and the first matmul raises a device mismatch, on both the tiled and untiled paths.
    """
    for tiled in (False, True):
        pipe = _build()
        if tiled:
            pipe.diffusion_decoder.enable_tiling(**TestTiling.SPLIT_TILES)
        pipe.enable_model_cpu_offload(device=torch_device)
        latents = torch.randn(1, 8, 3, 4, 5, generator=torch.Generator().manual_seed(2))
        frames = pipe(latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt").frames
        assert frames.shape == (1, 17, 3, 64, 80)
        assert torch.isfinite(frames).all()


@pytest.mark.parametrize("model_output_type", ["v", "x0"])
def test_scheduler_step_matches_the_closed_form_euler_update(model_output_type):
    """The scheduler must integrate exactly what the decoder's own solver did, on both prediction types.

    This is the regression the move to a scheduler is most exposed to: `step` would still return a plausible
    tensor if the sign of `dt` flipped, if the x0-to-velocity conversion used the wrong sigma, or if the sigma
    handed to the model were the scheduler's `sigma * num_train_timesteps` timestep instead. So the loop is
    recomputed here in closed form -- `x - (sigma - sigma_next) * v` -- and compared bit for bit.
    """
    steps = 3
    pipe = _build(decoder_model_output_type=model_output_type, decoder_num_inference_steps=steps)
    decoder, latents = pipe.diffusion_decoder, _latents()
    sigmas = pipe.get_sigmas()
    assert len(sigmas) == steps

    with torch.no_grad():
        context = decoder.encode_context_stage_4(decoder.encode_context_stages_1_to_3(latents))
        pixel_shape = (
            latents.shape[0],
            DECODER_CONFIG["out_channels"],
            context.shape[1],
            context.shape[2] * DECODER_CONFIG["patch_size"],
            context.shape[3] * DECODER_CONFIG["patch_size"],
        )
        # Same draw the pipeline makes, so both loops start from the same canvas.
        x_t = randn_tensor(
            pixel_shape,
            generator=torch.Generator(torch_device).manual_seed(0),
            device=latents.device,
            dtype=latents.dtype,
        )

        # float32 scalars, matching the dtype the scheduler holds its sigmas in: a Python float would divide
        # and subtract in double and leave a few ulps of difference that say nothing about the update rule.
        sigma_values = torch.tensor(sigmas + [0.0], dtype=torch.float32, device=torch_device)
        for i in range(steps):
            sigma, sigma_next = sigma_values[i], sigma_values[i + 1]
            prediction = decoder(x_t, context, sigma.expand(latents.shape[0]), return_dict=False)[0]
            if model_output_type == "x0":
                if i == steps - 1:
                    # The x0 shortcut: `x - sigma * (x - x0) / sigma` is the prediction itself, and taking it
                    # directly is what keeps the common one-step decode off a full-canvas float32 round trip.
                    expected = prediction
                    break
                velocity = (x_t.float() - prediction.float()) / sigma
            else:
                velocity = prediction.float()
            x_t = (x_t.float() - (sigma - sigma_next) * velocity).to(x_t.dtype)
        else:
            expected = x_t

        actual = pipe.decode(latents, generator=torch.Generator(torch_device).manual_seed(0), sigmas=sigmas)

    assert torch.equal(actual, expected), (
        f"scheduler-driven decode diverged from the closed-form Euler update by "
        f"{(actual - expected).abs().max().item():.3e} for model_output_type={model_output_type!r}"
    )


@pytest.mark.parametrize(
    "bad_config",
    [{"shift": 3.0}, {"shift_terminal": 0.1}, {"stochastic_sampling": True}],
)
def test_a_scheduler_that_reshapes_the_schedule_warns(bad_config, caplog):
    """Three scheduler settings change the decode with no error of their own -- they must not pass in silence.

    `use_dynamic_shifting` is left out: `set_timesteps` already raises on it, so it cannot be the silent case.
    """
    pipe = _build()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(**{**_scheduler().config, **bad_config})
    with caplog.at_level("WARNING", logger="diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode"):
        pipe(_latents(), generator=torch.Generator(torch_device).manual_seed(0), output_type="pt")
    assert any(key in caplog.text for key in bad_config), (
        f"decoding with {bad_config} produced no warning naming it; log was: {caplog.text!r}"
    )


def test_the_decoders_own_scheduler_does_not_warn(caplog):
    """The shipped config must be silent, or the warning above is noise every user learns to ignore."""
    pipe = _build()
    with caplog.at_level("WARNING", logger="diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode"):
        pipe(_latents(), generator=torch.Generator(torch_device).manual_seed(0), output_type="pt")
    assert not caplog.text, f"the decoder's own scheduler warned: {caplog.text!r}"
