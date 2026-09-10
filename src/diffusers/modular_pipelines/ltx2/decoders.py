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

from contextlib import nullcontext
from typing import Any

import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, LTX2VideoDiffusionDecoderModel

# NOTE (modular.md gotcha #1): `LTX2Vocoder` / `LTX2VocoderWithBWE` currently live under
# `diffusers.pipelines.ltx2.vocoder`, and modular blocks must not import from `diffusers.pipelines.*`.
# They are already `ModelMixin` / `ConfigMixin` model classes, so the clean fix is to relocate them to
# `src/diffusers/models/` and re-export from `diffusers.models` before this lands. Imported from the
# pipelines path here only so the draft is runnable; switch to the models path once moved.
from ...pipelines.ltx2.vocoder import LTX2Vocoder
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


logger = logging.get_logger(__name__)


# The pack/unpack/denormalize helpers below mirror the static methods on
# `diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline`. They are redefined here (rather than imported)
# because modular blocks must not import from `diffusers.pipelines.*` (modular.md gotcha #1); this follows
# the same redefinition pattern as `modular_pipelines/ltx/decoders.py`.
def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Denormalize video latents across the channel dimension [B, C, F, H, W].
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    # Packed video latents of shape [B, S, D] are unpacked into a video tensor of shape [B, C, F, H, W].
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _denormalize_audio_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


def _unpack_audio_latents(
    latents: torch.Tensor,
    latent_length: int,
    num_mel_bins: int,
    patch_size: int | None = None,
    patch_size_t: int | None = None,
) -> torch.Tensor:
    # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor [B, C, L, M].
    if patch_size is not None and patch_size_t is not None:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
    else:
        # Assume [B, S, D] = [B, L, C * M], i.e. a (mel) patch_size of M and a patch_size_t of 1.
        latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
    return latents


# The diffusion decoder's denoising loop and its mid-network tiling live on
# `LTX2VideoDiffusionDecodePipeline`. Modular blocks must not import from `diffusers.pipelines.*`
# (modular.md gotcha #1), so they are copied here and kept in sync by `make fix-copies`.
# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._blend_v
def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blend `b`'s top edge into `a`'s bottom edge with a linear ramp. See `AutoencoderKLLTX2Video.blend_v`."""
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
            y / blend_extent
        )
    return b


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._blend_h
def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blend `b`'s left edge into `a`'s right edge with a linear ramp. See `AutoencoderKLLTX2Video.blend_h`."""
    blend_extent = min(a.shape[4], b.shape[4], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
            x / blend_extent
        )
    return b


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._blend_t
def _blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blend `b`'s first frames into `a`'s last frames with a linear ramp. See `AutoencoderKLLTX2Video.blend_t`."""
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    for x in range(blend_extent):
        b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
            x / blend_extent
        )
    return b


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._check_scheduler
def _check_scheduler(scheduler: FlowMatchEulerDiscreteScheduler) -> None:
    """Reject only what this pipeline cannot drive, which is resolution-dependent shifting.

    Nothing else about the scheduler is checked. The sigmas the decoder was distilled on are a default
    ([`get_sigmas`]), not a requirement: a shift, a terminal shift or a stochastic update are all legitimate choices,
    and a finetune may well want them — moving the loop onto a scheduler is what makes them possible.
    """
    if scheduler.config.use_dynamic_shifting:
        raise ValueError(
            f"{scheduler.__class__.__name__} has `use_dynamic_shifting=True`, which needs a resolution-derived "
            "`mu` that this pipeline does not compute. Use a scheduler with `use_dynamic_shifting=False` — the "
            "converted checkpoints ship one in a `diffusion_decoder_scheduler` subfolder. Note that a "
            "transformer's scheduler usually has it on, so it cannot be reused here as-is."
        )


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._progress_bar
def _progress_bar(progress_bar, total: int):
    """`progress_bar(total=total)` when the caller has one, an inert context otherwise."""
    return progress_bar(total=total) if progress_bar is not None else nullcontext()


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._decoder_sigmas
def _decoder_sigmas(decoder: LTX2VideoDiffusionDecoderModel, num_inference_steps: int | None = None) -> list[float]:
    """The decoder's sigma schedule: `linspace(1, 1 / num_inference_steps, num_inference_steps)`.

    Uniform, rather than the scheduler's own default `linspace(sigma_max, sigma_min, n)`, so it has to be handed to
    `set_timesteps` explicitly. `num_inference_steps` defaults to what the decoder was distilled for.
    """
    if num_inference_steps is None:
        num_inference_steps = decoder.config.decoder_num_inference_steps
    return torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=torch.float32).tolist()


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._denoise
def _denoise(
    decoder: LTX2VideoDiffusionDecoderModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    latent_context: torch.Tensor,
    x_t: torch.Tensor,
    sigmas: list[float],
    progress_bar=None,
) -> torch.Tensor:
    """Denoise `x_t` `(B, C, F, H, W)` through the decoder's diffusion stage, conditioned on `latent_context`."""
    model_output_type = decoder.config.decoder_model_output_type
    batch_size, dtype = latent_context.shape[0], x_t.dtype

    # Once per call, not once per decode: a tiled decode runs this loop from the top for every tile, and
    # `set_timesteps` is what rewinds the scheduler's step index between them.
    scheduler.set_timesteps(sigmas=sigmas, device=x_t.device)
    num_inference_steps = len(scheduler.timesteps)

    for i, t in enumerate(scheduler.timesteps):
        # The decoder takes the noise level in [0, 1] and scales it itself, so hand it the sigma rather than the
        # scheduler's `sigma * num_train_timesteps` timestep.
        sigma = scheduler.sigmas[i]
        model_output = decoder(x_t, latent_context, sigma.expand(batch_size), return_dict=False)[0]

        # An x0 prediction at the last step *is* the sample: the Euler update to t=0 reduces to
        # `x_t - t * (x_t - prediction) / t`. Returning it skips a full-canvas float32 round trip.
        if model_output_type == "x0" and i == num_inference_steps - 1:
            if progress_bar is not None:
                progress_bar.update()
            return model_output

        model_output = model_output.float()
        if model_output_type == "x0":
            # The scheduler integrates a velocity, so turn the sample prediction into one.
            model_output = (x_t.float() - model_output) / sigma
        # `step` returns in `model_output`'s dtype, float32 above to keep the update off the canvas dtype; the
        # canvas itself stays in the dtype its noise was drawn in.
        x_t = scheduler.step(model_output, t, x_t, return_dict=False)[0].to(dtype)
        if progress_bar is not None:
            progress_bar.update()
    return x_t


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._tiled_decode
def _tiled_decode(
    decoder: LTX2VideoDiffusionDecoderModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    z: torch.Tensor,
    generator: torch.Generator | None,
    sigmas: list[float],
    progress_bar=None,
) -> torch.Tensor:
    """Decode with the last deterministic stage and the diffusion stage running per tile. See [`tiled_decode`].

    The cut itself comes from [`LTX2VideoDiffusionDecoderModel.get_tile_schedule`] — it is a fact about the decoder's
    grid, not about sampling. What is here is the part that has to be: each tile runs its own denoising loop, so the
    loop over tiles necessarily wraps the loop over steps.
    """
    config = decoder.config
    batch_size = z.shape[0]
    patch_size = config.patch_size

    features = decoder.encode_context_stages_1_to_3(z)
    schedule = decoder.get_tile_schedule(features.shape[1:4])
    scale_t, scale_h, scale_w = schedule.scales
    stride_t, stride_h, stride_w = schedule.cell_strides
    blend_frames, blend_height, blend_width = schedule.blend

    # A single-step x0 decode predicts pixels from pure noise, so each tile draws its own; a multi-step decode
    # integrates its noise across steps, so overlapping tiles must start from the same canvas.
    single_step_x0 = len(sigmas) == 1 and config.decoder_model_output_type == "x0"
    x_t_full = None
    if not single_step_x0:
        x_t_full = randn_tensor(
            (batch_size, config.out_channels, *schedule.pixel_shape),
            generator=generator,
            device=z.device,
            dtype=z.dtype,
        )

    frame_groups = []
    with _progress_bar(progress_bar, schedule.num_tiles * len(sigmas)) as bar:
        for t0, t1 in schedule.temporal:
            rows = []
            for h0, h1 in schedule.height:
                row = []
                for w0, w1 in schedule.width:
                    context = decoder.encode_context_stage_4(
                        features[:, t0 : schedule.feature_end(t1), h0:h1, w0:w1],
                        drop_leading_frame=t0 == 0,
                        crop_trailing_ghost=t1 == schedule.num_frames,
                    )
                    tile_pixel_shape = (
                        batch_size,
                        config.out_channels,
                        context.shape[1],
                        context.shape[2] * patch_size,
                        context.shape[3] * patch_size,
                    )
                    if single_step_x0:
                        x_t = randn_tensor(tile_pixel_shape, generator=generator, device=z.device, dtype=z.dtype)
                    else:
                        pixel_t0 = schedule.pixel_origin(t0)
                        x_t = x_t_full[
                            :,
                            :,
                            pixel_t0 : pixel_t0 + tile_pixel_shape[2],
                            h0 * scale_h : h0 * scale_h + tile_pixel_shape[3],
                            w0 * scale_w : w0 * scale_w + tile_pixel_shape[4],
                        ]
                    row.append(_denoise(decoder, scheduler, context, x_t, sigmas, progress_bar=bar))
                rows.append(row)

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile to the current tile and add the current tile to the
                    # result row
                    if i > 0:
                        tile = _blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = _blend_h(row[j - 1], tile, blend_width)
                    # The last tile can extend past the stride grid (a short remnant is merged into it), so it
                    # keeps its full extent instead of being cropped to the stride.
                    keep_height = stride_h * scale_h if i < len(rows) - 1 else tile.shape[3]
                    keep_width = stride_w * scale_w if j < len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_height, :keep_width])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

    result = []
    for k, group in enumerate(frame_groups):
        if k > 0:
            group = _blend_t(frame_groups[k - 1], group, blend_frames)
        if k < len(frame_groups) - 1:
            group = group[:, :, : schedule.pixel_frames(stride_t, is_origin=k == 0)]
        result.append(group)
    return torch.cat(result, dim=2)


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._should_tile
def _should_tile(decoder: LTX2VideoDiffusionDecoderModel, z: torch.Tensor) -> bool:
    """Whether tiling is on *and* the video is big enough for the schedule to actually split it."""
    config = decoder.config
    return decoder.use_tiling and (
        z.shape[2] > decoder.tile_sample_min_num_frames // config.temporal_compression_ratio
        or z.shape[3] > decoder.tile_sample_min_height // config.spatial_compression_ratio
        or z.shape[4] > decoder.tile_sample_min_width // config.spatial_compression_ratio
    )


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode._untiled_decode
def _untiled_decode(
    decoder: LTX2VideoDiffusionDecoderModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    z: torch.Tensor,
    generator: torch.Generator | None,
    sigmas: list[float],
    progress_bar=None,
) -> torch.Tensor:
    """Decode denormalized latents in one pass: every stage sees the whole volume."""
    config = decoder.config
    latent_context = decoder.encode_context_stage_4(decoder.encode_context_stages_1_to_3(z))
    # The context grid is the diffusion stage's token grid, so the pixel canvas is its shape times the patch size —
    # temporally that is the causal (T - 1) * ratio + 1 mapping of the LTX-2 latent space.
    pixel_shape = (
        z.shape[0],
        config.out_channels,
        latent_context.shape[1],
        latent_context.shape[2] * config.patch_size,
        latent_context.shape[3] * config.patch_size,
    )
    x_t = randn_tensor(pixel_shape, generator=generator, device=z.device, dtype=z.dtype)
    with _progress_bar(progress_bar, len(sigmas)) as bar:
        return _denoise(decoder, scheduler, latent_context, x_t, sigmas, progress_bar=bar)


class LTX2TrimConditionTokensStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Drops the appended keyframe-condition tokens from the denoised latents, leaving only the "
            "generated-video tokens for the decoders."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam(
                "base_token_count",
                type_hint=int,
                required=True,
                description="Number of generated-video tokens, i.e. the sequence length before appended tokens.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Denoised latents for the generated video, with condition tokens removed.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.latents = block_state.latents[:, : block_state.base_token_count]
        self.set_block_state(state, block_state)
        return components, state


class LTX2DiffusionVaeDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Step that unpacks and decodes the denoised video latents with the LTX-2 diffusion decoder (or returns "
            "latents). Swap this in for `LTX2VaeDecoderStep` on checkpoints that ship the diffusion decoder, which "
            "from LTX-2.5 on is the native default. The decoder denoises rather than deterministically decoding, so "
            "it draws its own noise from `generator`, runs its own denoising loop over "
            "`decode_num_inference_steps`, and needs its own scheduler."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("diffusion_decoder", LTX2VideoDiffusionDecoderModel),
            # Not `scheduler`: that name is the transformer's, and its config has `use_dynamic_shifting` on, which
            # the decoder cannot satisfy — it walks a plain uniform sigma schedule. Created from config by default
            # so a repo whose `modular_model_index.json` predates this component still loads.
            ComponentSpec(
                "diffusion_decoder_scheduler",
                FlowMatchEulerDiscreteScheduler,
                config=FrozenDict(
                    {
                        "num_train_timesteps": 1000,
                        "shift": 1.0,
                        "use_dynamic_shifting": False,
                        "shift_terminal": None,
                        "stochastic_sampling": False,
                    }
                ),
                default_creation_method="from_config",
            ),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam.template("latents", required=True),
            InputParam.template("output_type", default="pil"),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames", type_hint=int, default=121, description="The number of frames in the generated video."
            ),
            InputParam.template("generator"),
            InputParam.template("dtype", required=True),
            InputParam(
                "decode_num_inference_steps",
                type_hint=int,
                description=(
                    "Number of denoising steps the diffusion decoder takes. Separate from `num_inference_steps`, "
                    "which belongs to the transformer's loop. Defaults to what the decoder was distilled for."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        decoder = components.diffusion_decoder

        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio

        latents = _unpack_latents(
            block_state.latents,
            latent_num_frames,
            latent_height,
            latent_width,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        )

        if block_state.output_type == "latent":
            block_state.videos = _denormalize_latents(
                latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
            )
            self.set_block_state(state, block_state)
            return components, state

        latents = latents.to(block_state.dtype)
        latents = _denormalize_latents(
            latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
        )
        latents = latents.to(decoder.dtype)
        # The decoder's `forward` is a single denoising step, so the loop (and the tiling around it) is driven
        # from here. It samples the noise it denoises, so pass the generator to keep decoding reproducible.
        scheduler = components.diffusion_decoder_scheduler
        _check_scheduler(scheduler)
        sigmas = _decoder_sigmas(decoder, block_state.decode_num_inference_steps)
        decode = _tiled_decode if _should_tile(decoder, latents) else _untiled_decode
        video = decode(decoder, scheduler, latents, block_state.generator, sigmas)
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class LTX2VaeDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return "Step that unpacks and decodes the denoised video latents into videos (or returns latents)."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLLTX2Video),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 32}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam.template("latents", required=True),
            InputParam.template("output_type", default="pil"),
            InputParam.template("height", default=512),
            InputParam.template("width", default=704),
            InputParam(
                "num_frames",
                type_hint=int,
                default=None,
                description=(
                    "The number of frames in the generated video. Omit to auto-predict via the `duration_head` "
                    "(see `LTX2AutoDurationStep`)."
                ),
            ),
            InputParam(
                "decode_timestep", default=0.0, description="The timestep at which the VAE decodes the final latents."
            ),
            InputParam(
                "decode_noise_scale",
                default=None,
                description="Noise interpolation factor applied to the latents at the decode timestep.",
            ),
            InputParam.template("generator"),
            InputParam.template("batch_size"),
            InputParam.template("dtype", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae = components.vae

        latents = block_state.latents
        height = block_state.height
        width = block_state.width
        num_frames = block_state.num_frames

        latent_num_frames = (num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = height // components.vae_spatial_compression_ratio
        latent_width = width // components.vae_spatial_compression_ratio

        latents = _unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        )

        if block_state.output_type == "latent":
            block_state.videos = _denormalize_latents(
                latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
            )
            self.set_block_state(state, block_state)
            return components, state

        # LTX-2 applies the optional decode-time noise on the *normalized* latents, then denormalizes
        # (the reverse of LTX-1's decoder order).
        latents = latents.to(block_state.dtype)
        if not vae.config.timestep_conditioning:
            timestep = None
        else:
            device = latents.device
            batch_size = block_state.batch_size
            decode_timestep = block_state.decode_timestep
            decode_noise_scale = block_state.decode_noise_scale

            noise = randn_tensor(latents.shape, generator=block_state.generator, device=device, dtype=latents.dtype)
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size

            timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                :, None, None, None, None
            ]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents = _denormalize_latents(
            latents, components.latents_mean, components.latents_std, components.vae_scaling_factor
        )
        latents = latents.to(vae.dtype)
        video = vae.decode(latents, timestep, return_dict=False)[0]
        block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class LTX2AudioDecoderStep(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def description(self) -> str:
        return (
            "Step that unpacks and decodes the denoised audio latents into a waveform via the audio VAE and vocoder "
            "(or returns the unpacked audio latents when `output_type='latent'`)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        # The checkpoint may ship either `LTX2Vocoder` or `LTX2VocoderWithBWE`; the concrete class is resolved from
        # the vocoder subfolder's config at load time. `LTX2Vocoder` is declared here as the representative type.
        return [
            ComponentSpec("audio_vae", AutoencoderKLLTX2Audio),
            ComponentSpec("vocoder", LTX2Vocoder),
        ]

    @property
    def inputs(self) -> list[tuple[str, Any]]:
        return [
            InputParam("audio_latents", type_hint=torch.Tensor, required=True, description="Denoised audio latents."),
            InputParam(
                "audio_num_frames",
                type_hint=int,
                required=True,
                description="Number of audio latent frames, used to unpack the audio latent sequence.",
            ),
            InputParam.template("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        audio_vae = components.audio_vae

        num_mel_bins = audio_vae.config.mel_bins
        latent_mel_bins = num_mel_bins // components.audio_vae_mel_compression_ratio

        audio_latents = _denormalize_audio_latents(
            block_state.audio_latents, components.audio_latents_mean, components.audio_latents_std
        )
        audio_latents = _unpack_audio_latents(
            audio_latents, block_state.audio_num_frames, num_mel_bins=latent_mel_bins
        )

        if block_state.output_type == "latent":
            block_state.audio = audio_latents
        else:
            audio_latents = audio_latents.to(audio_vae.dtype)
            generated_mel_spectrograms = audio_vae.decode(audio_latents, return_dict=False)[0]
            block_state.audio = components.vocoder(generated_mel_spectrograms)

        self.set_block_state(state, block_state)
        return components, state
