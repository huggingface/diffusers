# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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

import math
from contextlib import nullcontext

import torch

from ...models.autoencoders import AutoencoderKLLTX2Video, LTX2VideoDiffusionDecoderModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LTX2VideoDecodeOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _tile_intervals(length: int, tile_size: int, stride: int, min_size: int) -> list[tuple[int, int]]:
    """Overlapping `[start, end)` tiles covering `[0, length)`, with starts spaced `stride` apart.

    A trailing remnant shorter than `min_size` is merged into the previous tile instead of decoded on its own:
    neighborhood attention rejects any grid smaller than its kernel, so a remnant tile cannot always stand alone.
    """
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [(starts[-1], length)]


def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blend `b`'s top edge into `a`'s bottom edge with a linear ramp. See `AutoencoderKLLTX2Video.blend_v`."""
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
            y / blend_extent
        )
    return b


def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blend `b`'s left edge into `a`'s right edge with a linear ramp. See `AutoencoderKLLTX2Video.blend_h`."""
    blend_extent = min(a.shape[4], b.shape[4], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
            x / blend_extent
        )
    return b


def _blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blend `b`'s first frames into `a`'s last frames with a linear ramp. See `AutoencoderKLLTX2Video.blend_t`."""
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    for x in range(blend_extent):
        b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
            x / blend_extent
        )
    return b


def _progress_bar(progress_bar, total: int):
    """`progress_bar(total=total)` when the caller has one, an inert context otherwise."""
    return progress_bar(total=total) if progress_bar is not None else nullcontext()


def _decoder_sigmas(decoder: LTX2VideoDiffusionDecoderModel, num_inference_steps: int | None = None) -> list[float]:
    """The decoder's sigma schedule: `linspace(1, 1 / num_inference_steps, num_inference_steps)`.

    Uniform, rather than the scheduler's own default `linspace(sigma_max, sigma_min, n)`, so it has to be handed to
    `set_timesteps` explicitly. `num_inference_steps` defaults to what the decoder was distilled for.
    """
    if num_inference_steps is None:
        num_inference_steps = decoder.config.decoder_num_inference_steps
    return torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=torch.float32).tolist()


def _tile_geometry(config) -> tuple[tuple[int, int, int], list[int]]:
    """Pixels per cell of the tiling grid, and the smallest tile each remaining kernel will accept."""
    patch_size = config.patch_size
    # One cell of the grid entering the last deterministic stage covers this many output pixels: that stage's
    # upsample stride times the diffusion stage's patch size.
    upsample_stride = config.decoder_upsample_strides[-1]
    scales = (upsample_stride[0], upsample_stride[1] * patch_size, upsample_stride[2] * patch_size)
    # Every tile must satisfy both remaining neighborhood-attention kernels: the last deterministic stage sees the
    # tile as-is, the diffusion stage sees it scaled by the upsample stride.
    min_sizes = [
        max(kernel_4, -(-kernel_5 // stride))
        for kernel_4, kernel_5, stride in zip(
            config.decoder_stage_kernels[-1], config.decoder_stage5_kernel, upsample_stride
        )
    ]
    return scales, min_sizes


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


def _tiled_decode(
    decoder: LTX2VideoDiffusionDecoderModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    z: torch.Tensor,
    generator: torch.Generator | None,
    sigmas: list[float],
    progress_bar=None,
) -> torch.Tensor:
    """Decode with the last deterministic stage and the diffusion stage running per tile. See [`tiled_decode`]."""
    config = decoder.config
    batch_size = z.shape[0]
    patch_size = config.patch_size
    (scale_t, scale_h, scale_w), min_sizes = _tile_geometry(config)

    features = decoder.encode_context_stages_1_to_3(z)
    # The trailing ghost frames replicate through the earlier stages' temporal upsamples, whose composed mapping
    # is affine with slope equal to the product of their strides.
    trailing_pad_latent_frames = (config.decoder_stage_kernels[0][0] // 2) * 2
    ghost_frames = trailing_pad_latent_frames * math.prod(stride[0] for stride in config.decoder_upsample_strides[:-1])
    num_frames = features.shape[1] - ghost_frames
    height, width = features.shape[2], features.shape[3]

    tile_t = decoder.tile_sample_min_num_frames // scale_t
    stride_t = decoder.tile_sample_stride_num_frames // scale_t
    tile_h, stride_h = decoder.tile_sample_min_height // scale_h, decoder.tile_sample_stride_height // scale_h
    tile_w, stride_w = decoder.tile_sample_min_width // scale_w, decoder.tile_sample_stride_width // scale_w

    temporal_tiles = _tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])
    height_tiles = _tile_intervals(height, tile_h, stride_h, min_sizes[1])
    width_tiles = _tile_intervals(width, tile_w, stride_w, min_sizes[2])
    blend_frames = (tile_t - stride_t) * scale_t
    blend_height = (tile_h - stride_h) * scale_h
    blend_width = (tile_w - stride_w) * scale_w

    # A single-step x0 decode predicts pixels from pure noise, so each tile draws its own; a multi-step decode
    # integrates its noise across steps, so overlapping tiles must start from the same canvas.
    single_step_x0 = len(sigmas) == 1 and config.decoder_model_output_type == "x0"
    x_t_full = None
    if not single_step_x0:
        pixel_frames = num_frames * scale_t - (1 if scale_t == 2 else 0)
        x_t_full = randn_tensor(
            (batch_size, config.out_channels, pixel_frames, height * scale_h, width * scale_w),
            generator=generator,
            device=z.device,
            dtype=z.dtype,
        )

    num_tiles = len(temporal_tiles) * len(height_tiles) * len(width_tiles)
    frame_groups = []
    with _progress_bar(progress_bar, num_tiles * len(sigmas)) as bar:
        for t0, t1 in temporal_tiles:
            is_origin = t0 == 0
            is_trailing = t1 == num_frames
            # The tile containing the video end takes the ghost frames with it into the last stage.
            feature_t1 = features.shape[1] if is_trailing else t1
            rows = []
            for h0, h1 in height_tiles:
                row = []
                for w0, w1 in width_tiles:
                    context = decoder.encode_context_stage_4(
                        features[:, t0:feature_t1, h0:h1, w0:w1],
                        drop_leading_frame=is_origin,
                        crop_trailing_ghost=is_trailing,
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
                        # A non-origin tile keeps the duplicate leading frame, placing its first cell one pixel
                        # frame earlier than `t0 * scale_t` — the causal 1-then-`scale_t` frame mapping.
                        pixel_t0 = t0 * scale_t - (1 if not is_origin and scale_t == 2 else 0)
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
            # The origin group is one frame short of `stride * scale`: its first cell decodes to a single pixel
            # frame under the causal mapping.
            keep_frames = stride_t * scale_t - (1 if k == 0 and scale_t == 2 else 0)
            group = group[:, :, :keep_frames]
        result.append(group)
    return torch.cat(result, dim=2)


def _should_tile(decoder: LTX2VideoDiffusionDecoderModel, z: torch.Tensor) -> bool:
    """Whether tiling is on *and* the video is big enough for the schedule to actually split it."""
    config = decoder.config
    return decoder.use_tiling and (
        z.shape[2] > decoder.tile_sample_min_num_frames // config.temporal_compression_ratio
        or z.shape[3] > decoder.tile_sample_min_height // config.spatial_compression_ratio
        or z.shape[4] > decoder.tile_sample_min_width // config.spatial_compression_ratio
    )


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


class LTX2VideoDiffusionDecodePipeline(DiffusionPipeline):
    r"""
    Decode LTX-2 video latents with the diffusion decoder introduced in LTX-2.5.

    Unlike a convolutional decoder this one is itself a small diffusion model: it denoises pixels conditioned on a
    context volume built from the latents, so it needs a scheduler and a generator. Pair it with any LTX-2 pipeline run
    with `output_type="latent"`, passing `denormalize=False` since that path already applied the latent statistics.

    Because the decoder denoises, the tiling lives here rather than on the model: tiles are cut in the *middle* of the
    decoder, on the grid entering its last deterministic stage, and each tile runs its own denoising loop before the
    results are blended. Turn it on with `pipe.diffusion_decoder.enable_tiling()`, which sets the tile sizes this
    pipeline reads.

    Args:
        diffusion_decoder ([`LTX2VideoDiffusionDecoderModel`]):
            The diffusion video decoder. Its `forward` is a single denoising step; this pipeline owns the loop.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler driving the decoder's denoising steps. This is not the transformer's scheduler: the decoder walks
            a plain uniform sigma schedule, so it needs one with `use_dynamic_shifting=False` and
            `shift_terminal=None`. Checkpoints converted by `convert_ltx2_to_diffusers.py` ship it in a
            `diffusion_decoder_scheduler` subfolder.
        vae ([`AutoencoderKLLTX2Video`], *optional*):
            Only consulted for the latent statistics used to denormalize. When omitted the pipeline falls back to the
            LTX-2 defaults, so a decode-only workflow does not have to load a second autoencoder.
    """

    model_cpu_offload_seq = "diffusion_decoder"
    _optional_components = ["vae"]

    def __init__(
        self,
        diffusion_decoder: LTX2VideoDiffusionDecoderModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTX2Video = None,
    ):
        super().__init__()
        self.register_modules(diffusion_decoder=diffusion_decoder, scheduler=scheduler, vae=vae)
        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def _latent_stats(self, device: torch.device, dtype: torch.dtype):
        """Latent mean/std/scaling, from `vae` when it is loaded and from the LTX-2 defaults when it is not."""
        if self.vae is not None:
            return self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        return (
            self.diffusion_decoder.latents_mean.to(device=device, dtype=dtype),
            self.diffusion_decoder.latents_std.to(device=device, dtype=dtype),
            self.diffusion_decoder.config.scaling_factor,
        )

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample.LTX2LatentUpsamplePipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def get_sigmas(self, num_inference_steps: int | None = None) -> list[float]:
        """The decoder's sigma schedule, `linspace(1, 1 / num_inference_steps, num_inference_steps)`."""
        return _decoder_sigmas(self.diffusion_decoder, num_inference_steps)

    def tiled_decode(
        self, z: torch.Tensor, generator: torch.Generator | None = None, sigmas: list[float] | None = None
    ) -> torch.Tensor:
        r"""Decode a batch of latents with the last deterministic stage and the diffusion stage running per tile.

        Tiles live on the grid entering the last deterministic stage, where one cell maps to a fixed block of output
        pixels; the decoder's `tile_sample_*` sizes are converted to that grid, so they should be multiples of the cell
        size (8 px spatially and 2 frames temporally for the production config). Temporal tiles follow the causal frame
        mapping: the tile containing t=0 drops the temporal upsample's duplicate leading frame and only the tile
        containing the video end carries the NATTEN border padding.

        This tiles whatever `diffusion_decoder.use_tiling` says — [`decode`] is the entry point that honors it.
        """
        sigmas = sigmas if sigmas is not None else self.get_sigmas()
        return _tiled_decode(self.diffusion_decoder, self.scheduler, z, generator, sigmas, self.progress_bar)

    def decode(
        self, z: torch.Tensor, generator: torch.Generator | None = None, sigmas: list[float] | None = None
    ) -> torch.Tensor:
        """Decode a batch of latents, routing to [`tiled_decode`] when tiling is on and the video needs it.

        `z` is expected to be denormalized already, matching [`AutoencoderKLLTX2Video`].
        """
        sigmas = sigmas if sigmas is not None else self.get_sigmas()
        if _should_tile(self.diffusion_decoder, z):
            return self.tiled_decode(z, generator=generator, sigmas=sigmas)
        return _untiled_decode(self.diffusion_decoder, self.scheduler, z, generator, sigmas, self.progress_bar)

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.Tensor,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        denormalize: bool = True,
    ):
        r"""
        Args:
            latents (`torch.Tensor`):
                Latents of shape `(B, C, F, H, W)`. Note that an LTX-2 pipeline run with `output_type="latent"` returns
                latents that are *already* denormalized, so pass `denormalize=False` for those.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Defaults to the decoder's `decoder_num_inference_steps` config value, which
                is what the checkpoint was distilled for — 1 for LTX-2.5.
            sigmas (`list[float]`, *optional*):
                Custom sigma schedule, overriding `num_inference_steps`. The default is the uniform `linspace(1, 1 /
                num_inference_steps, num_inference_steps)` the decoder was trained on.
            generator (`torch.Generator`, *optional*):
                The decoder samples the noise it denoises, so pass a generator to make decoding reproducible.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the decoded video. Choose between `"pil"`, `"np"`, `"pt"` and `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.ltx2.pipeline_output.LTX2VideoDecodeOutput`] instead of a plain tuple.
            denormalize (`bool`, *optional*, defaults to `True`):
                Whether to apply the latent statistics before decoding. Set to `False` if the latents are already
                denormalized.

        Returns:
            [`~pipelines.ltx2.pipeline_output.LTX2VideoDecodeOutput`] or `tuple`
        """
        if sigmas is not None and num_inference_steps is not None:
            raise ValueError("Only one of `num_inference_steps` or `sigmas` can be passed, not both.")

        device = self._execution_device
        latents = latents.to(device)

        if denormalize:
            latents_mean, latents_std, scaling_factor = self._latent_stats(device, latents.dtype)
            latents = self._denormalize_latents(latents, latents_mean, latents_std, scaling_factor)

        latents = latents.to(self.diffusion_decoder.dtype)
        sigmas = sigmas if sigmas is not None else self.get_sigmas(num_inference_steps)
        video = self.decode(latents, generator=generator, sigmas=sigmas)
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return LTX2VideoDecodeOutput(frames=video)
