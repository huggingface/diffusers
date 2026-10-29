# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import torch

from ...models.transformers.transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import KandinskyS2VModularPipeline, align, split_audio_embeds


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def memory_coefs(config) -> tuple[tuple[int, int, int], ...]:
    return (config.patch_proj1x_coef, config.patch_proj2x_coef, config.patch_proj4x_coef)


def init_memory(config, height: int, width: int, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
    """Zero FramePack memory buffers for the first chunk, in ``[1x, 2x, 4x]`` order."""
    patch_size = config.patch_size
    buffers = []
    for bucket, coef in zip(config.frame_buckets, memory_coefs(config), strict=True):
        h_mult, w_mult = patch_size[1] * coef[1], patch_size[2] * coef[2]
        buffers.append(
            torch.zeros(
                bucket, align(height, h_mult), align(width, w_mult), config.in_visual_dim, dtype=dtype, device=device
            )
        )
    return buffers


def roll_memory(config, latents: torch.Tensor) -> list[torch.Tensor]:
    """The tail of this chunk becomes the next chunk's memory, in ``[1x, 2x, 4x]`` order."""
    patch_size = config.patch_size
    frame_buckets = config.frame_buckets

    needed = sum(frame_buckets)
    if latents.shape[0] < needed:
        raise ValueError(
            f"latents have {latents.shape[0]} frames but the FramePack memory window "
            f"sum(frame_buckets)={needed} requires at least that many"
        )

    offsets = [0]
    for bucket in frame_buckets:
        offsets.append(offsets[-1] + bucket)

    height, width = latents.shape[1], latents.shape[2]
    buffers = []
    for i, coef in enumerate(memory_coefs(config)):
        window = latents[latents.shape[0] - offsets[i + 1] : latents.shape[0] - offsets[i]]
        h_mult, w_mult = patch_size[1] * coef[1], patch_size[2] * coef[2]
        buffers.append(window[:, : align(height, h_mult), : align(width, w_mult), :].clone())
    return buffers


def memory_rope_pos(config, memory: list[torch.Tensor], device: torch.device) -> list[list[torch.Tensor]]:
    """RoPE positions for ``[1x, 2x, 4x]`` memory, which sits at negative temporal indices."""
    patch_size = config.patch_size

    bounds = [0]
    for bucket in config.frame_buckets:
        bounds.append(bounds[-1] + bucket)

    out = []
    for i, (buffer, coef) in enumerate(zip(memory, memory_coefs(config), strict=True)):
        height = buffer.shape[1] // patch_size[1]
        width = buffer.shape[2] // patch_size[2]
        out.append(
            [
                torch.arange(-bounds[i + 1], -bounds[i], coef[0], device=device),
                torch.arange(0, height, coef[1], device=device),
                torch.arange(0, width, coef[2], device=device),
            ]
        )
    return out


class KandinskyS2VPrepareStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Prepare step that crops the reference latent to whole patches, casts the conditions to the "
            "transformer's dtype, splits the speech features evenly over the chunks, builds the RoPE positions and "
            "the empty FramePack memory the first chunk starts from."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", KandinskyS2VTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "reference_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Scaled VAE latent of the reference frame, from the VAE encoder step.",
            ),
            InputParam(
                "audio_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Audio-tower features of the speech track, from the text encoder step.",
            ),
            InputParam("num_chunks", required=True, type_hint=int, description="Number of chunks to generate."),
            InputParam(
                "num_latent_frames",
                default=32,
                type_hint=int,
                description="Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).",
            ),
            InputParam.template("prompt_embeds", required=True),
            InputParam.template("negative_prompt_embeds"),
            InputParam(
                "pooled_prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="CLIP pooled prompt embedding.",
            ),
            InputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="CLIP pooled negative prompt embedding.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_hidden_states", type_hint=torch.Tensor, description="Reference latent, cropped and cast."
            ),
            OutputParam("latent_height", type_hint=int, description="Height of the latents."),
            OutputParam("latent_width", type_hint=int, description="Width of the latents."),
            OutputParam("audio_chunks", type_hint=list, description="Speech features of each chunk."),
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor, description="CLIP pooled prompt embedding."),
            OutputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="CLIP pooled negative prompt embedding.",
            ),
            OutputParam("text_rope_pos", type_hint=torch.Tensor, description="RoPE positions of the prompt tokens."),
            OutputParam(
                "negative_text_rope_pos",
                type_hint=torch.Tensor,
                description="RoPE positions of the negative prompt tokens.",
            ),
            OutputParam("visual_rope_pos", type_hint=list, description="Per-axis RoPE positions of a chunk."),
            OutputParam(
                "reference_rope_pos", type_hint=list, description="Per-axis RoPE positions of the reference frame."
            ),
            OutputParam("memory", type_hint=list, description="FramePack memory in [1x, 2x, 4x] order."),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        needed = sum(components.transformer.config.frame_buckets)
        if block_state.num_latent_frames < needed:
            raise ValueError(
                f"`num_latent_frames` must be >= {needed}, the FramePack memory window "
                f"sum(frame_buckets), got {block_state.num_latent_frames}"
            )

    @staticmethod
    def prepare_negative_embeds(block_state, dtype: torch.dtype, device: torch.device) -> None:
        block_state.negative_text_rope_pos = None
        if (block_state.negative_prompt_embeds is None) != (block_state.negative_pooled_prompt_embeds is None):
            raise ValueError("`negative_prompt_embeds` and `negative_pooled_prompt_embeds` must be passed together.")
        if block_state.negative_prompt_embeds is not None:
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.to(dtype)
            block_state.negative_pooled_prompt_embeds = block_state.negative_pooled_prompt_embeds.to(dtype)
            block_state.negative_text_rope_pos = torch.arange(
                block_state.negative_prompt_embeds.shape[0], device=device
            )

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)
        device = components._execution_device
        config = components.transformer.config
        dtype = components.transformer.dtype
        patch_size = config.patch_size

        reference = block_state.reference_latents
        reference = reference[
            :, : align(reference.shape[1], patch_size[1]), : align(reference.shape[2], patch_size[2])
        ]
        block_state.latent_height, block_state.latent_width = reference.shape[1], reference.shape[2]
        block_state.reference_hidden_states = reference.to(dtype)

        block_state.prompt_embeds = block_state.prompt_embeds.to(dtype)
        block_state.pooled_prompt_embeds = block_state.pooled_prompt_embeds.to(dtype)
        block_state.text_rope_pos = torch.arange(block_state.prompt_embeds.shape[0], device=device)
        self.prepare_negative_embeds(block_state, dtype, device)

        block_state.audio_chunks = split_audio_embeds(block_state.audio_embeds.to(dtype), block_state.num_chunks)

        height, width = block_state.latent_height, block_state.latent_width
        block_state.visual_rope_pos = [
            torch.arange(block_state.num_latent_frames, device=device),
            torch.arange(height // patch_size[1], device=device),
            torch.arange(width // patch_size[2], device=device),
        ]
        block_state.reference_rope_pos = [
            torch.tensor([config.index_rope_ref_image], device=device),
            torch.arange(height // patch_size[1], device=device),
            torch.arange(width // patch_size[2], device=device),
        ]
        # Latents and memory stay fp32; only the forward runs in the transformer's dtype.
        block_state.memory = init_memory(config, height, width, torch.float32, device)

        self.set_block_state(state, block_state)
        return components, state


class KandinskyS2VDistilledPrepareStep(KandinskyS2VPrepareStep):
    model_name = "kandinsky-s2v-distilled"

    @property
    def inputs(self) -> list[InputParam]:
        return [param for param in super().inputs if not param.name.startswith("negative_")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [param for param in super().intermediate_outputs if not param.name.startswith("negative_")]

    @staticmethod
    def prepare_negative_embeds(block_state, dtype: torch.dtype, device: torch.device) -> None:
        # No guidance on the distilled checkpoints, so there are no negative embeddings to prepare.
        pass
