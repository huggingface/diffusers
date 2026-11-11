# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

import numpy as np
import torch

from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging
from ...utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# The spatial rotary axes are normalized by the square root of the latent area and scaled by 32.
_ROPE_SPATIAL_SCALE = 32


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise.patchify_video_latents
def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    r"""
    Pack video latents into transformer rows.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
            The latents to pack.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size * num_patches, channels * prod(patch_size))`: The packed rows, ordered
        frame-major then row-major.
    """
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._spatial_position_grid
def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    r"""
    One aspect-normalized spatial rotary axis: `dim // patch` coordinates centred on the unit interval, scaled up by
    32. The right endpoint is excluded, so a square canvas spans `[0, 32)`.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # Built with numpy: `np.linspace(..., endpoint=False)` is `start + arange(num) * (stop - start) / num`, which is
    # not what `torch.linspace` computes, and the float64 grid has to be reproduced exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._frame_position_grid
def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def get_image_canvas_size(height: int, width: int, min_generation_area: int, multiple: int) -> tuple[int, int]:
    r"""
    The canvas MiniMax-H3-Image generates a `height x width` image on.

    An image below `min_generation_area` pixels is generated larger at the same aspect ratio: both sides are scaled by
    `sqrt(min_generation_area / (height * width))`. Either way both sides round to the nearest `multiple`.

    Returns:
        `tuple[int, int]`: the `(height, width)` of the canvas.
    """
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def build_image_packed_sequence(
    text_token_tags: torch.Tensor,
    reference_grids: list[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
    image_tag: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    r"""
    Build the `[text | reference images | generated image]` layout of MiniMax-H3-Image.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision block,
            which are tagged `0` (image).
        reference_grids (`list[tuple[int, int]]`):
            The `(latent_height, latent_width)` of every reference image, in packed order. Empty for `text2image`.
        latent_height (`int`): Latent height of the generated image.
        latent_width (`int`): Latent width of the generated image.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        image_tag (`int`): The modality tag an image row carries, i.e. `components.image_tag`.

    Returns:
        `tuple`: `position_ids`, `token_tags`, `image_indices`, `text_indices`, and the number of leading image rows
        that are reference rows rather than generated ones.
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_reference_rows = sum((height // patch_h) * (width // patch_w) for height, width in reference_grids)
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    sequence_length = num_text_tokens + num_reference_rows + num_target_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    # Every image is one frame on the rotary clock, which starts where the text rows end: a reference takes one
    # integer slot each, and the generated image takes the slot after the last reference.
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for height, width in reference_grids:
        rows = slice(cursor, cursor + (height // patch_h) * (width // patch_w))
        cursor = rows.stop
        frame_grid, _ = _frame_position_grid(height, width, patch_h, patch_w)
        position_ids[rows, 0] = rotary_time
        position_ids[rows, 1:] = frame_grid
        rotary_time += 1.0
    frame_grid, _ = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_grid

    text_indices = torch.arange(num_text_tokens)
    image_indices = torch.arange(num_text_tokens, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = image_tag

    return position_ids, token_tags, image_indices, text_indices, num_reference_rows


class MiniMaxH3ImagePrepareCanvasStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Resolves the canvas an image is generated on. An image smaller than `min_generation_area` pixels is "
            "generated on a larger canvas of the same aspect ratio and resized back after decoding; either way the "
            "canvas sides are rounded to multiples of 32."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=2048, description="Height of the returned image in pixels."),
            InputParam.template("width", default=2048, description="Width of the returned image in pixels."),
            InputParam(
                name="min_generation_area",
                type_hint=int,
                default=2048 * 2048,
                description=(
                    "The smallest area, in pixels, an image is generated at. A smaller `height x width` is generated "
                    "on a canvas scaled up to this area and resized back to `height x width` after decoding. Pass "
                    "1024 to always generate at the requested size."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Height of the canvas the image is generated on."),
            OutputParam("width", type_hint=int, description="Width of the canvas the image is generated on."),
            OutputParam("output_height", type_hint=int, description="Height of the returned image in pixels."),
            OutputParam("output_width", type_hint=int, description="Width of the returned image in pixels."),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.height <= 0 or block_state.width <= 0:
            raise ValueError(f"`height` and `width` must be positive, got {block_state.height}x{block_state.width}.")
        if not 1024 <= block_state.min_generation_area <= 4096 * 4096:
            raise ValueError(
                f"`min_generation_area` must be within [1024, {4096 * 4096}], got {block_state.min_generation_area}."
            )

        block_state.output_height, block_state.output_width = block_state.height, block_state.width
        block_state.height, block_state.width = get_image_canvas_size(
            block_state.height, block_state.width, block_state.min_generation_area, components.canvas_multiple
        )

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImagePrepareLayoutStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Builds the packed layout of a `text2image` request, `[text | generated image]`, and its float64 rotary "
            "grid."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="text_token_tags",
                type_hint=torch.Tensor,
                required=True,
                description="The modality tag of every row of `prompt_embeds`.",
            ),
            InputParam.template("height", required=True, description="Height of the canvas in pixels."),
            InputParam.template("width", required=True, description="Width of the canvas in pixels."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latent_height", type_hint=int, description="Height of the generated latents."),
            OutputParam("latent_width", type_hint=int, description="Width of the generated latents."),
            OutputParam(
                "position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The `(t, h, w)` rotary coordinate of every row, in float64.",
            ),
            OutputParam(
                "token_tags",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The modality tag of every row.",
            ),
            OutputParam(
                "image_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the image rows.",
            ),
            OutputParam(
                "text_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the text rows.",
            ),
            OutputParam(
                "num_reference_rows",
                type_hint=int,
                description="How many leading image rows are reference rows: none for `text2image`.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        multiple = components.canvas_multiple
        if block_state.height % multiple or block_state.width % multiple:
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got {block_state.height}x{block_state.width}."
            )
        ratio = components.vae_spatial_compression_ratio
        block_state.latent_height = block_state.height // ratio
        block_state.latent_width = block_state.width // ratio

        (
            position_ids,
            token_tags,
            image_indices,
            text_indices,
            block_state.num_reference_rows,
        ) = build_image_packed_sequence(
            block_state.text_token_tags,
            [],
            block_state.latent_height,
            block_state.latent_width,
            components.patch_size,
            components.image_tag,
        )
        # The grid is built in fp64 to reproduce the released coordinates exactly, but MPS, NPU and Neuron have no
        # fp64; the transformer's rope casts to fp32 anyway, so downcasting here is all the transfer needs.
        position_ids_dtype = maybe_adjust_dtype_for_device(position_ids.dtype, device)
        block_state.position_ids = position_ids.to(device, position_ids_dtype)
        block_state.token_tags = token_tags.to(device)
        block_state.image_indices = image_indices.to(device)
        block_state.text_indices = text_indices.to(device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageEditPrepareLayoutStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Builds the packed layout of an edit, `[text | reference images | generated image]`, and its float64 "
            "rotary grid. Every reference keeps its own latent grid."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="text_token_tags",
                type_hint=torch.Tensor,
                required=True,
                description="The modality tag of every row of `prompt_embeds`.",
            ),
            InputParam(
                name="reference_latents",
                type_hint=list[torch.Tensor],
                required=True,
                description=(
                    "The conditioning latents of the reference images. Their shape is where every reference's "
                    "geometry comes from."
                ),
            ),
            InputParam.template("height", required=True, description="Height of the canvas in pixels."),
            InputParam.template("width", required=True, description="Width of the canvas in pixels."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latent_height", type_hint=int, description="Height of the generated latents."),
            OutputParam("latent_width", type_hint=int, description="Width of the generated latents."),
            OutputParam(
                "position_ids",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The `(t, h, w)` rotary coordinate of every row, in float64.",
            ),
            OutputParam(
                "token_tags",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="The modality tag of every row.",
            ),
            OutputParam(
                "image_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the image rows, reference rows first.",
            ),
            OutputParam(
                "text_indices",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="Sequence positions of the text rows.",
            ),
            OutputParam(
                "num_reference_rows",
                type_hint=int,
                description="How many leading image rows are reference rows rather than generated rows.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        multiple = components.canvas_multiple
        if block_state.height % multiple or block_state.width % multiple:
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got {block_state.height}x{block_state.width}."
            )
        ratio = components.vae_spatial_compression_ratio
        block_state.latent_height = block_state.height // ratio
        block_state.latent_width = block_state.width // ratio

        reference_grids = [tuple(latents.shape[-2:]) for latents in block_state.reference_latents]
        (
            position_ids,
            token_tags,
            image_indices,
            text_indices,
            block_state.num_reference_rows,
        ) = build_image_packed_sequence(
            block_state.text_token_tags,
            reference_grids,
            block_state.latent_height,
            block_state.latent_width,
            components.patch_size,
            components.image_tag,
        )
        # The grid is built in fp64 to reproduce the released coordinates exactly, but MPS, NPU and Neuron have no
        # fp64; the transformer's rope casts to fp32 anyway, so downcasting here is all the transfer needs.
        position_ids_dtype = maybe_adjust_dtype_for_device(position_ids.dtype, device)
        block_state.position_ids = position_ids.to(device, position_ids_dtype)
        block_state.token_tags = token_tags.to(device)
        block_state.image_indices = image_indices.to(device)
        block_state.text_indices = text_indices.to(device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImagePrepareLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Draws the noise of the generated image as a latent tensor and packs it into rows. An edit draws its "
            "reference noise off the same generator after it."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latent_height", type_hint=int, required=True, description="Height of the generated latents."
            ),
            InputParam(
                name="latent_width", type_hint=int, required=True, description="Width of the generated latents."
            ),
            InputParam.template("generator"),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                description=(
                    "Pre-generated noise of shape `(1, latent_channels, latent_height, latent_width)`, used instead "
                    "of the draw."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated image rows of the packed sequence.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        shape = (1, components.vae_latent_channels, 1, block_state.latent_height, block_state.latent_width)
        if block_state.latents is None:
            latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=torch.float32)
        else:
            latents = block_state.latents.to(device, torch.float32).reshape(shape)
        block_state.latents = patchify_video_latents(latents, components.patch_size)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageEditPrepareLatentsStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Noises the reference latents of an edit to the conditioning level, packs them into rows and puts them "
            "in front of the generated rows, which is where the layout reserved them. One noise tensor per reference "
            "is drawn off the request's generator, after the noise of the generated image. The denoising loop only "
            "ever writes the generated rows, so the conditioning rides through every step unchanged."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", MiniMaxH3Scheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="reference_latents",
                type_hint=list[torch.Tensor],
                required=True,
                description="The conditioning latents of the reference images, in packed order.",
            ),
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The generated image rows of the packed sequence.",
            ),
            InputParam(
                name="num_reference_rows",
                type_hint=int,
                required=True,
                description="How many reference rows the layout reserved, which the packed references must match.",
            ),
            InputParam.template(
                "generator",
                description="The generator of the request, after the noise of the generated image is drawn from it.",
            ),
            InputParam(
                name="reference_noise",
                type_hint=list[torch.Tensor],
                description=(
                    "Pre-generated noise for the reference images, one tensor of the shape of each entry of "
                    "`reference_latents`, used instead of the draws."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The image rows of the packed sequence, reference rows first.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        reference_noise = block_state.reference_noise
        if reference_noise is None:
            reference_noise = [
                randn_tensor(latents.shape, generator=block_state.generator, device=device, dtype=torch.float32)
                for latents in block_state.reference_latents
            ]
        elif len(reference_noise) != len(block_state.reference_latents):
            raise ValueError(
                f"`reference_noise` needs one tensor per reference image, got {len(reference_noise)} for "
                f"{len(block_state.reference_latents)} references."
            )

        condition_rows = []
        for latents, noise in zip(block_state.reference_latents, reference_noise):
            latents = latents.to(device, torch.float32)
            noise = noise.to(device, torch.float32).reshape(latents.shape)
            # The references are not fully clean: they are noised to `t = 0.999` and held there for every step. The
            # level goes in as a tensor so `1 - t` is evaluated in float32, as the released model mixes it.
            noised = components.scheduler.scale_noise(latents, torch.tensor(components.reference_noise_aug), noise)
            condition_rows.append(patchify_video_latents(noised[:, :, None], components.patch_size))
        condition_rows = torch.cat(condition_rows)

        if condition_rows.shape[0] != block_state.num_reference_rows:
            raise ValueError(
                f"The layout reserved {block_state.num_reference_rows} reference rows but the references pack "
                f"into {condition_rows.shape[0]}. The references the layout was built from and the ones encoded here "
                "do not agree."
            )
        block_state.latents = torch.cat([condition_rows, block_state.latents])

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageSetTimestepsStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Sets the scheduler to the checkpoint's sampling grid and plans the per-row timesteps of every step. The "
            "grid is fixed: the transformer has one output head per step of it. The reference rows of an edit stay "
            "at their conditioning level for every step."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", MiniMaxH3Scheduler)]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="sample_sigmas",
                default=[
                    1.0,
                    0.9675236940383911,
                    0.8875485062599182,
                    0.75,
                    0.5327739119529724,
                    0.23200887441635132,
                    0.0,
                ],
                description="The sampling grid the checkpoint's per-step output heads were distilled on.",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="image_indices",
                type_hint=torch.Tensor,
                required=True,
                description="Sequence positions of the image rows, reference rows first.",
            ),
            InputParam(
                name="text_indices",
                type_hint=torch.Tensor,
                required=True,
                description="Sequence positions of the text rows.",
            ),
            InputParam(
                name="num_reference_rows",
                type_hint=int,
                default=0,
                description="How many leading image rows are reference rows.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps of the schedule."),
            OutputParam(
                "row_timestep_plan",
                type_hint=list,
                description=(
                    "One `(timestep, timestep_indices)` pair per step: the distinct timesteps of the sequence and the "
                    "index of every row into them."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device

        # The released sampler rounds every sigma through its float32 timestep `1 - sigma` and back, and steps on the
        # result; below 0.5 that round trip moves a sigma by an ulp.
        timesteps = torch.tensor([1.0 - sigma for sigma in components.config.sample_sigmas[:-1]], dtype=torch.float32)
        sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
        components.scheduler.set_timesteps(sigmas=sigmas, device=device)
        block_state.timesteps = components.scheduler.timesteps

        image_indices = block_state.image_indices.cpu()
        sequence_length = int(image_indices.numel() + block_state.text_indices.numel())
        block_state.row_timestep_plan = []
        for timestep in block_state.timesteps:
            row_timesteps = torch.full((sequence_length,), float(timestep), dtype=torch.float32)
            row_timesteps[image_indices[: block_state.num_reference_rows]] = components.reference_noise_aug
            unique_timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
            block_state.row_timestep_plan.append((unique_timesteps.to(device), timestep_indices.to(device)))

        self.set_block_state(state, block_state)
        return components, state
