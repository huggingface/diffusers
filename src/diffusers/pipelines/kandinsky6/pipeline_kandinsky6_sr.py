# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import inspect
import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from ...models import Kandinsky6SRLatentUpscalerBank, Kandinsky6SRTransformer3DModel, Kandinsky6SRVAE
from ...schedulers import FlowMatchEulerDiscreteScheduler, PiflowScheduler
from ...utils import replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import Kandinsky6SRPipelineOutput


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import Kandinsky6SRPipeline, Kandinsky6TI2VAPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = Kandinsky6TI2VAPipeline.from_pretrained(
        ...     "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()
        >>> video = pipe(
        ...     prompt="A cat and a dog baking a cake together in a kitchen.",
        ...     height=480,
        ...     width=864,
        ...     num_inference_steps=16,
        ...     guidance_scale=1.0,
        ...     sample_audio=False,
        ... ).frames[0]

        >>> sr_pipe = Kandinsky6SRPipeline.from_pretrained(
        ...     "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", torch_dtype=torch.bfloat16
        ... )
        >>> # The transformer always runs attention through the `flex` backend; compiling avoids the eager
        >>> # fallback's much higher memory use at video resolutions.
        >>> sr_pipe.transformer.compile_repeated_blocks(fullgraph=True)
        >>> sr_pipe.enable_model_cpu_offload()
        >>> output = sr_pipe(video=video, resolution_scale=2.25, num_inference_steps=2)
        >>> export_to_video(output.frames[0], "output_sr.mp4", fps=24)
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _tile_positions(length: int, tile: int, min_overlap: float, snap: int) -> list[int]:
    """Evenly spread, `snap`-aligned tile start positions covering `[0, length - tile]` with at least `min_overlap`
    (a fraction of `tile`) shared between neighbours."""
    if tile >= length:
        return [0]
    span = (length - tile) // snap
    max_stride = max(1, math.floor(tile * (1.0 - min_overlap) / snap))
    count = math.ceil(span / max_stride) + 1
    return [round(index * span / (count - 1)) * snap for index in range(count)]


def _hann_window_2d(height: int, width: int, device: torch.device) -> torch.Tensor:
    """2D Hann window with non-zero borders, so a region covered by a single tile keeps a positive weight."""
    window_y = torch.hann_window(height + 2, device=device)[1:-1]
    window_x = torch.hann_window(width + 2, device=device)[1:-1]
    return window_y[:, None] * window_x[None, :]


class Kandinsky6SRPipeline(DiffusionPipeline):
    r"""
    Pipeline for video super-resolution with Kandinsky 6.

    The video is split into overlapping spatial tiles, every tile is refined by the SR transformer at one of the tile
    sizes the model was trained on (`transformer.config.tile_sizes`), and the refined tiles are blended back with Hann
    windows. When the pipeline has a `latent_upscaler`, the tiles are cut from the K-VAE latents of the whole video and
    upscaled in latent space; otherwise the pixel tiles are bilinearly upscaled and encoded.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`Kandinsky6SRTransformer3DModel`]):
            Transformer that refines the latent tiles.
        vae ([`Kandinsky6SRVAE`]):
            Causal video K-VAE used to encode the input video and decode the refined tiles.
        scheduler ([`FlowMatchEulerDiscreteScheduler`] or [`PiflowScheduler`]):
            Scheduler used with `transformer` to denoise the tiles. Distilled checkpoints ship with a
            [`PiflowScheduler`].
        latent_upscaler ([`Kandinsky6SRLatentUpscalerBank`], *optional*):
            Latent upscalers for the supported scales.
    """

    model_cpu_offload_seq = "latent_upscaler->transformer->vae"
    _optional_components = ["latent_upscaler"]

    def __init__(
        self,
        transformer: Kandinsky6SRTransformer3DModel,
        vae: Kandinsky6SRVAE,
        scheduler: FlowMatchEulerDiscreteScheduler | PiflowScheduler,
        latent_upscaler: Kandinsky6SRLatentUpscalerBank | None = None,
    ) -> None:
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler, latent_upscaler=latent_upscaler)

        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 16
        )
        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 4
        )
        self.transformer_tile_sizes = (
            tuple(tuple(size) for size in self.transformer.config.tile_sizes)
            if getattr(self, "transformer", None) is not None
            else ((512, 512), (512, 768), (768, 512))
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def check_inputs(
        self, video, resolution_scale, num_inference_steps, lq_noise_scale, min_overlap, tiles_batch_size
    ):
        if resolution_scale not in (2, 2.25, 4):
            raise ValueError(f"`resolution_scale` must be 2, 2.25 or 4 but is {resolution_scale}.")
        if num_inference_steps < 1:
            raise ValueError(f"`num_inference_steps` has to be positive but is {num_inference_steps}.")
        if not 0.0 < lq_noise_scale <= 1.0:
            raise ValueError(f"`lq_noise_scale` has to be in (0, 1] but is {lq_noise_scale}.")
        if not 0.0 <= min_overlap < 1.0:
            raise ValueError(f"`min_overlap` has to be in [0, 1) but is {min_overlap}.")
        if tiles_batch_size < 1:
            raise ValueError(f"`tiles_batch_size` has to be positive but is {tiles_batch_size}.")

        num_frames = video.shape[2]
        if (num_frames - 1) % self.vae_scale_factor_temporal != 0:
            raise ValueError(
                f"`video` must have `1 + k * {self.vae_scale_factor_temporal}` frames but has {num_frames}."
            )

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        r"""
        Encodes a `(batch_size, channels, num_frames, height, width)` video in `[-1, 1]` into K-VAE latents scaled by
        the VAE `scaling_factor`.
        """
        # The K-VAE was trained on `pixels / 128 - 1` rather than the `pixels / 127.5 - 1` of `VideoProcessor`.
        video = (video.to(self.vae.dtype) + 1) * (127.5 / 128) - 1
        latents = self.vae.encode(video, return_dict=False)[0].mode()
        return latents * self.vae.config.scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        r"""Decodes scaled K-VAE latents into a `(batch_size, channels, num_frames, height, width)` video in `[-1, 1]`."""
        video = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        return ((video.float() + 1) * (128 / 127.5) - 1).clamp(-1, 1)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: list[PIL.Image.Image] | list[list[PIL.Image.Image]] | np.ndarray | torch.Tensor,
        resolution_scale: float = 2.25,
        num_inference_steps: int = 4,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        lq_noise_scale: float = 0.7,
        min_overlap: float = 0.2,
        tiles_batch_size: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Kandinsky6SRPipelineOutput | tuple:
        r"""
        The call function to the pipeline for super-resolution.

        Args:
            video (`list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`):
                The low-resolution video(s), in any format [`~video_processor.VideoProcessor.preprocess_video`]
                accepts, with `1 + k * 4` frames. Sizes are rounded down to a multiple of the VAE spatial factor.
            resolution_scale (`float`, defaults to `2.25`):
                Total spatial upscale: `2`, `4`, or `2.25` (a 1.125x bilinear pre-upscale followed by the 2x path).
            num_inference_steps (`int`, defaults to `4`):
                The number of denoising steps per tile. Use `2` with the distilled checkpoints.
            timesteps (`list[int]`, *optional*):
                Custom timesteps for schedulers that support them.
            sigmas (`list[float]`, *optional*):
                Custom sigmas for schedulers that support them.
            lq_noise_scale (`float`, defaults to `0.7`):
                Amount of Gaussian noise mixed into the low-resolution latents (variance preserving) before denoising.
            min_overlap (`float`, defaults to `0.2`):
                Minimum overlap between neighbouring tiles as a fraction of the tile size.
            tiles_batch_size (`int`, defaults to `1`):
                Number of tiles denoised per transformer call.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) used for the noise mixed into the tiles.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generated video: `"pil"`, `"np"` or `"pt"`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`Kandinsky6SRPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`Kandinsky6SRPipelineOutput`] or `tuple`:
                The super-resolved video; a one-element tuple when `return_dict=False`.
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        # 1. Preprocess the video and check inputs
        video = self.video_processor.preprocess_video(video).to(device)
        self.check_inputs(video, resolution_scale, num_inference_steps, lq_noise_scale, min_overlap, tiles_batch_size)

        # 2. The 2.25x route bilinearly pre-upscales the pixels by 1.125x before the 2x path
        tiling_scale, pre_upscale = (2, 1.125) if resolution_scale == 2.25 else (int(resolution_scale), 1.0)
        if pre_upscale != 1.0:
            batch_size, channels, num_frames, height, width = video.shape
            snap = self.vae_scale_factor_spatial
            target = (round(height * pre_upscale / snap) * snap, round(width * pre_upscale / snap) * snap)
            frames = video.permute(0, 2, 1, 3, 4).flatten(0, 1)
            frames = F.interpolate(frames, size=target, mode="bilinear", align_corners=False)
            video = frames.unflatten(0, (batch_size, num_frames)).permute(0, 2, 1, 3, 4)
        batch_size, _, num_frames, height, width = video.shape

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 3. Tile grid at the input resolution; every tile is refined at the closest trained tile resolution
        base_height, base_width = min(self.transformer_tile_sizes, key=lambda hw: abs(hw[1] / hw[0] - width / height))
        snap = self.vae_scale_factor_spatial
        if base_height % (snap * tiling_scale) or base_width % (snap * tiling_scale):
            raise ValueError(
                f"The tile size {(base_height, base_width)} must be divisible by the VAE spatial factor times the"
                f" upscale factor ({snap * tiling_scale}) so that tiles align with the latent grid."
            )
        tile_height, tile_width = base_height // tiling_scale, base_width // tiling_scale
        if height < tile_height or width < tile_width:
            raise ValueError(
                f"`video` must be at least {tile_height}x{tile_width} pixels for `resolution_scale={resolution_scale}`,"
                f" got {height}x{width}."
            )
        tops = _tile_positions(height, tile_height, min_overlap, snap)
        lefts = _tile_positions(width, tile_width, min_overlap, snap)
        tile_grid = [(top, left) for top in tops for left in lefts]

        # 4. Low-resolution latent tiles at the trained tile resolution
        use_latent_upscaler = self.latent_upscaler is not None and tiling_scale in self.latent_upscaler.config.scales
        if use_latent_upscaler:
            latents = self.encode_video(video)
            lq_tiles = [
                self.latent_upscaler(
                    latents[
                        :, :, :, top // snap : (top + tile_height) // snap, left // snap : (left + tile_width) // snap
                    ],
                    scale=tiling_scale,
                    return_dict=False,
                )[0]
                for top, left in tile_grid
            ]
        else:
            lq_tiles = []
            for top, left in tile_grid:
                pixel_tile = video[:, :, :, top : top + tile_height, left : left + tile_width]
                frames = pixel_tile.permute(0, 2, 1, 3, 4).flatten(0, 1)
                frames = F.interpolate(frames, size=(base_height, base_width), mode="bilinear", align_corners=False)
                pixel_tile = frames.unflatten(0, (batch_size, num_frames)).permute(0, 2, 1, 3, 4)
                lq_tiles.append(self.encode_video(pixel_tile))
        lq_tiles = torch.stack(lq_tiles, dim=1).to(dtype)  # (batch_size, num_tiles, channels, frames, height, width)

        # 5. Denoise the tiles and blend the decoded tiles into the output canvas
        output_height, output_width = height * tiling_scale, width * tiling_scale
        video_acc = torch.zeros((batch_size, 3, num_frames, output_height, output_width), device=device)
        window = _hann_window_2d(base_height, base_width, device)
        weight_acc = torch.zeros((1, 1, 1, output_height, output_width), device=device)
        for top, left in tile_grid:
            top, left = top * tiling_scale, left * tiling_scale
            weight_acc[..., top : top + base_height, left : left + base_width] += window
        num_chunks = math.ceil(len(tile_grid) / tiles_batch_size)

        with self.progress_bar(total=batch_size * num_chunks * num_inference_steps) as progress_bar:
            for sample_index in range(batch_size):
                sample_generator = generator[sample_index] if isinstance(generator, list) else generator
                for start in range(0, len(tile_grid), tiles_batch_size):
                    tile_indices = range(start, min(start + tiles_batch_size, len(tile_grid)))
                    # The transformer works on `(batch, frames, height, width, channels)` tiles
                    lq_latents = lq_tiles[sample_index, list(tile_indices)].permute(0, 2, 3, 4, 1)

                    # Variance-preserving noise mixed into the low-resolution latents is the starting point
                    noise = randn_tensor(lq_latents.shape, generator=sample_generator, device=device, dtype=dtype)
                    latents = math.sqrt(1 - lq_noise_scale**2) * lq_latents + lq_noise_scale * noise

                    timesteps_, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
                    for t in timesteps_:
                        # The SR transformer takes `[latent | anchor latent | anchor mask]`; the released checkpoints
                        # are anchor-free, so the anchor channels are zeros.
                        anchor = torch.zeros_like(latents)
                        anchor_mask = torch.zeros((*latents.shape[:-1], 1), dtype=dtype, device=device)
                        latent_model_input = torch.cat([latents, anchor, anchor_mask], dim=-1)
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t.expand(latents.shape[0]),
                            return_dict=False,
                        )[0]
                        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                        progress_bar.update()

                    decoded = self.decode_latents(latents.permute(0, 4, 1, 2, 3))
                    for tile, tile_index in zip(decoded, tile_indices):
                        top, left = tile_grid[tile_index]
                        top, left = top * tiling_scale, left * tiling_scale
                        video_acc[sample_index, :, :, top : top + base_height, left : left + base_width] += (
                            tile * window
                        )

        video = (video_acc / weight_acc).clamp(-1, 1)
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return Kandinsky6SRPipelineOutput(frames=video)
