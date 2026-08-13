# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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


import numpy as np
import torch

from ...models.transformers.transformer_krea2 import Krea2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Krea2ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.krea2.pipeline_krea2.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# auto_docstring
class Krea2TextInputsStep(ModularPipelineBlocks):
    """
    Input step that determines `batch_size`/`dtype` from the per-prompt `prompt_embeds` and replicates the text
    conditioning (and the optional negative branch) to `batch_size * num_images_per_prompt`. Place after the text
    encoder.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.

      Outputs:
          batch_size (`int`):
              Effective batch size (num prompts * num_images_per_prompt).
          dtype (`dtype`):
              The dtype of the text features.
          prompt_embeds (`Tensor`):
              Text features, batch-expanded.
          prompt_embeds_mask (`Tensor`):
              Text mask, batch-expanded.
          negative_prompt_embeds (`Tensor`):
              Negative text features, batch-expanded.
          negative_prompt_embeds_mask (`Tensor`):
              Negative text mask, batch-expanded.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Input step that determines `batch_size`/`dtype` from the per-prompt `prompt_embeds` and replicates the "
            "text conditioning (and the optional negative branch) to `batch_size * num_images_per_prompt`. Place after "
            "the text encoder."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt", default=1),
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).",
            ),
            InputParam(
                name="prompt_embeds_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt boolean text mask (B, text_seq_len).",
            ),
            InputParam(
                name="negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Per-prompt negative text features.",
            ),
            InputParam(
                name="negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Per-prompt negative text mask.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="batch_size",
                type_hint=int,
                description="Effective batch size (num prompts * num_images_per_prompt).",
            ),
            OutputParam(name="dtype", type_hint=torch.dtype, description="The dtype of the text features."),
            OutputParam(name="prompt_embeds", type_hint=torch.Tensor, description="Text features, batch-expanded."),
            OutputParam(name="prompt_embeds_mask", type_hint=torch.Tensor, description="Text mask, batch-expanded."),
            OutputParam(
                name="negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="Negative text features, batch-expanded.",
            ),
            OutputParam(
                name="negative_prompt_embeds_mask",
                type_hint=torch.Tensor,
                description="Negative text mask, batch-expanded.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_batch, seq_len, num_layers, dim = block_state.prompt_embeds.shape
        n = block_state.num_images_per_prompt

        block_state.dtype = block_state.prompt_embeds.dtype
        block_state.batch_size = prompt_batch * n

        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, n, 1, 1).view(
            prompt_batch * n, seq_len, num_layers, dim
        )
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.repeat(1, n).view(prompt_batch * n, seq_len)

        if block_state.negative_prompt_embeds is not None:
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds.repeat(1, n, 1, 1).view(
                prompt_batch * n, seq_len, num_layers, dim
            )
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask.repeat(1, n).view(
                prompt_batch * n, seq_len
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2TurboTextInputsStep(ModularPipelineBlocks):
    """
    Input step for the distilled Krea 2 turbo checkpoint that determines `batch_size`/`dtype` from the per-prompt
    `prompt_embeds` and replicates the text conditioning to `batch_size * num_images_per_prompt`. The distilled
    checkpoint runs without classifier-free guidance, so there is no negative branch. Place after the text encoder.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).

      Outputs:
          batch_size (`int`):
              Effective batch size (num prompts * num_images_per_prompt).
          dtype (`dtype`):
              The dtype of the text features.
          prompt_embeds (`Tensor`):
              Text features, batch-expanded.
          prompt_embeds_mask (`Tensor`):
              Text mask, batch-expanded.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Input step for the distilled Krea 2 turbo checkpoint that determines `batch_size`/`dtype` from the "
            "per-prompt `prompt_embeds` and replicates the text conditioning to `batch_size * num_images_per_prompt`. "
            "The distilled checkpoint runs without classifier-free guidance, so there is no negative branch. Place "
            "after the text encoder."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_images_per_prompt", default=1),
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).",
            ),
            InputParam(
                name="prompt_embeds_mask",
                required=True,
                type_hint=torch.Tensor,
                description="Per-prompt boolean text mask (B, text_seq_len).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="batch_size",
                type_hint=int,
                description="Effective batch size (num prompts * num_images_per_prompt).",
            ),
            OutputParam(name="dtype", type_hint=torch.dtype, description="The dtype of the text features."),
            OutputParam(name="prompt_embeds", type_hint=torch.Tensor, description="Text features, batch-expanded."),
            OutputParam(name="prompt_embeds_mask", type_hint=torch.Tensor, description="Text mask, batch-expanded."),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt_batch, seq_len, num_layers, dim = block_state.prompt_embeds.shape
        n = block_state.num_images_per_prompt

        block_state.dtype = block_state.prompt_embeds.dtype
        block_state.batch_size = prompt_batch * n

        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, n, 1, 1).view(
            prompt_batch * n, seq_len, num_layers, dim
        )
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask.repeat(1, n).view(prompt_batch * n, seq_len)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2ImageInputsStep(ModularPipelineBlocks):
    """
    Pack image latents into Krea 2 image tokens and expand image and mask inputs to the effective prompt batch.

      Inputs:
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The preprocessed inpainting mask.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              Effective batch size.

      Outputs:
          image_latents (`Tensor`):
              The latent representation of the input image.
          processed_mask_image (`Tensor`):
              The batch-expanded inpainting mask.
          height (`int`):
              The generation height inferred from the image.
          width (`int`):
              The generation width inferred from the image.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Pack image latents into Krea 2 image tokens and expand image and mask inputs to the effective prompt "
            "batch."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image_latents"),
            InputParam(
                name="processed_mask_image", type_hint=torch.Tensor, description="The preprocessed inpainting mask."
            ),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("num_images_per_prompt", default=1),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("image_latents"),
            OutputParam(
                name="processed_mask_image", type_hint=torch.Tensor, description="The batch-expanded inpainting mask."
            ),
            OutputParam(name="height", type_hint=int, description="The generation height inferred from the image."),
            OutputParam(name="width", type_hint=int, description="The generation width inferred from the image."),
        ]

    @staticmethod
    def repeat_to_batch_size(input_name, input_tensor, prompt_batch_size, num_images_per_prompt):
        if input_tensor.shape[0] == 1:
            repeat_by = prompt_batch_size * num_images_per_prompt
        elif input_tensor.shape[0] == prompt_batch_size:
            repeat_by = num_images_per_prompt
        else:
            raise ValueError(
                f"`{input_name}` must have batch size 1 or {prompt_batch_size}, but got {input_tensor.shape[0]}"
            )
        return input_tensor.repeat_interleave(repeat_by, dim=0)

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image_latents = block_state.image_latents
        if image_latents.ndim != 5 or image_latents.shape[2] != 1:
            raise ValueError(
                f"`image_latents` must have shape (batch, channels, 1, height, width), got {image_latents.shape}"
            )

        image_height = image_latents.shape[-2] * components.vae_scale_factor
        image_width = image_latents.shape[-1] * components.vae_scale_factor
        block_state.height = block_state.height or image_height
        block_state.width = block_state.width or image_width
        if block_state.height != image_height or block_state.width != image_width:
            raise ValueError(
                f"The encoded image is {image_height}x{image_width}, but the requested output is "
                f"{block_state.height}x{block_state.width}."
            )

        p = components.patch_size
        batch_size, channels, _, latent_height, latent_width = image_latents.shape
        image_latents = image_latents[:, :, 0].view(batch_size, channels, latent_height // p, p, latent_width // p, p)
        image_latents = image_latents.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size, (latent_height // p) * (latent_width // p), channels * p * p
        )

        prompt_batch_size = block_state.batch_size // block_state.num_images_per_prompt
        block_state.image_latents = self.repeat_to_batch_size(
            "image_latents", image_latents, prompt_batch_size, block_state.num_images_per_prompt
        )
        if block_state.processed_mask_image is not None:
            block_state.processed_mask_image = self.repeat_to_batch_size(
                "processed_mask_image",
                block_state.processed_mask_image,
                prompt_batch_size,
                block_state.num_images_per_prompt,
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2ReferenceInputsStep(ModularPipelineBlocks):
    """
    Pack reference-image latents and expand them to the effective prompt batch.

      Inputs:
          reference_image_latents (`list`):
              Normalized reference-image latents from the VAE encoder in conditioning order.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              Effective batch size.

      Outputs:
          reference_image_latents (`list`):
              Packed reference-image latents expanded to the effective batch in conditioning order.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Pack reference-image latents and expand them to the effective prompt batch."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="reference_image_latents",
                required=True,
                type_hint=list[torch.Tensor],
                description="Normalized reference-image latents from the VAE encoder in conditioning order.",
            ),
            InputParam.template("num_images_per_prompt", default=1),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="reference_image_latents",
                type_hint=list[torch.Tensor],
                description="Packed reference-image latents expanded to the effective batch in conditioning order.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        p = components.patch_size
        prompt_batch_size = block_state.batch_size // block_state.num_images_per_prompt
        packed_reference_image_latents = []
        for reference_image_latents in block_state.reference_image_latents:
            if reference_image_latents.ndim != 5 or reference_image_latents.shape[2] != 1:
                raise ValueError(
                    "Each `reference_image_latents` tensor must have shape (batch, channels, 1, height, width), but "
                    f"got {reference_image_latents.shape}."
                )
            batch_size, channels, _, latent_height, latent_width = reference_image_latents.shape
            reference_image_latents = reference_image_latents[:, :, 0].view(
                batch_size, channels, latent_height // p, p, latent_width // p, p
            )
            reference_image_latents = reference_image_latents.permute(0, 2, 4, 1, 3, 5).reshape(
                batch_size, (latent_height // p) * (latent_width // p), channels * p * p
            )
            if batch_size == 1:
                repeat_by = prompt_batch_size * block_state.num_images_per_prompt
            elif batch_size == prompt_batch_size:
                repeat_by = block_state.num_images_per_prompt
            else:
                raise ValueError(
                    f"Each reference must have batch size 1 or {prompt_batch_size}, but got {batch_size}."
                )
            packed_reference_image_latents.append(reference_image_latents.repeat_interleave(repeat_by, dim=0))
        block_state.reference_image_latents = packed_reference_image_latents
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2PrepareLatentsStep(ModularPipelineBlocks):
    """
    Step that samples the spatial image latents and patch-packs them into (B, image_seq_len, in_channels) for the
    denoising loop.

      Components:
          transformer (`Krea2Transformer2DModel`)

      Inputs:
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              Effective batch size.
          dtype (`dtype`):
              The working dtype.

      Outputs:
          latents (`Tensor`):
              The initial packed image latents (B, image_seq_len, in_channels).
          image_seq_len (`int`):
              Number of image tokens (grid_h * grid_w).
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that samples the spatial image latents and patch-packs them into (B, image_seq_len, in_channels) "
            "for the denoising loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Krea2Transformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents"),
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam.template("generator"),
            InputParam(name="batch_size", required=True, type_hint=int, description="Effective batch size."),
            InputParam(name="dtype", required=True, type_hint=torch.dtype, description="The working dtype."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="latents",
                type_hint=torch.Tensor,
                description="The initial packed image latents (B, image_seq_len, in_channels).",
            ),
            OutputParam(name="image_seq_len", type_hint=int, description="Number of image tokens (grid_h * grid_w)."),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        p = components.patch_size
        num_channels_latents = components.transformer.config.in_channels // (p**2)

        multiple = components.vae_scale_factor * components.patch_size
        if block_state.height % multiple != 0 or block_state.width % multiple != 0:
            rounded_height = ((block_state.height + multiple - 1) // multiple) * multiple
            rounded_width = ((block_state.width + multiple - 1) // multiple) * multiple
            logger.warning(
                f"`height` and `width` must be multiples of {multiple}; rounding up from {block_state.height}x{block_state.width} to"
                f" {rounded_height}x{rounded_width}."
            )
            block_state.height, block_state.width = rounded_height, rounded_width

        latent_height = block_state.height // components.vae_scale_factor
        latent_width = block_state.width // components.vae_scale_factor

        if block_state.latents is not None:
            block_state.latents = block_state.latents.to(device=device, dtype=block_state.dtype)
        else:
            latents = randn_tensor(
                (block_state.batch_size, num_channels_latents, latent_height, latent_width),
                generator=block_state.generator,
                device=device,
                dtype=block_state.dtype,
            )
            latents = latents.view(
                block_state.batch_size, num_channels_latents, latent_height // p, p, latent_width // p, p
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            block_state.latents = latents.reshape(
                block_state.batch_size, (latent_height // p) * (latent_width // p), num_channels_latents * p * p
            )

        block_state.image_seq_len = block_state.latents.shape[1]

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2ApplyStrengthStep(ModularPipelineBlocks):
    """
    Truncate the Krea 2 denoising schedule according to image-to-image or inpainting strength.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
              The full denoising schedule.

      Outputs:
          timesteps (`Tensor`):
              The strength-adjusted timesteps.
          num_inference_steps (`int`):
              The strength-adjusted denoising step count.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Truncate the Krea 2 denoising schedule according to image-to-image or inpainting strength."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("strength", default=0.9),
            InputParam.template("num_inference_steps", required=True),
            InputParam(
                name="timesteps", required=True, type_hint=torch.Tensor, description="The full denoising schedule."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="timesteps", type_hint=torch.Tensor, description="The strength-adjusted timesteps."),
            OutputParam(
                name="num_inference_steps", type_hint=int, description="The strength-adjusted denoising step count."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if block_state.strength < 0 or block_state.strength > 1:
            raise ValueError(f"`strength` must be in [0.0, 1.0], but is {block_state.strength}")
        init_timestep = min(block_state.num_inference_steps * block_state.strength, block_state.num_inference_steps)
        t_start = int(max(block_state.num_inference_steps - init_timestep, 0))
        begin_index = t_start * components.scheduler.order
        block_state.timesteps = block_state.timesteps[begin_index:]
        block_state.num_inference_steps -= t_start
        if block_state.num_inference_steps < 1:
            raise ValueError(
                f"After applying `strength={block_state.strength}`, the number of denoising steps is "
                f"{block_state.num_inference_steps}, but it must be at least 1."
            )
        components.scheduler.set_begin_index(begin_index)
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2PrepareImageLatentsStep(ModularPipelineBlocks):
    """
    Add noise at the first selected timestep to packed Krea 2 image latents.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          timesteps (`Tensor`):
              The selected denoising timesteps.

      Outputs:
          initial_noise (`Tensor`):
              The sampled initial noise.
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Add noise at the first selected timestep to packed Krea 2 image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam.template("image_latents", required=True),
            InputParam(
                name="timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="The selected denoising timesteps.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="initial_noise", type_hint=torch.Tensor, description="The sampled initial noise."),
            OutputParam.template("latents"),
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if block_state.image_latents.shape != block_state.latents.shape:
            raise ValueError(
                f"`image_latents` and `latents` must have the same shape, got "
                f"{block_state.image_latents.shape} and {block_state.latents.shape}"
            )
        latent_timestep = block_state.timesteps[:1].repeat(block_state.latents.shape[0])
        block_state.initial_noise = block_state.latents
        block_state.latents = components.scheduler.scale_noise(
            block_state.image_latents, latent_timestep, block_state.initial_noise
        )
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2PrepareMaskLatentsStep(ModularPipelineBlocks):
    """
    Resize and pack a preprocessed inpainting mask into Krea 2 image-token space.

      Inputs:
          processed_mask_image (`Tensor`):
              The preprocessed inpainting mask.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          mask (`Tensor`):
              The packed latent-space mask.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Resize and pack a preprocessed inpainting mask into Krea 2 image-token space."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="processed_mask_image",
                required=True,
                type_hint=torch.Tensor,
                description="The preprocessed inpainting mask.",
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="mask", type_hint=torch.Tensor, description="The packed latent-space mask.")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        p = components.patch_size
        latent_height = block_state.height // components.vae_scale_factor
        latent_width = block_state.width // components.vae_scale_factor
        mask = torch.nn.functional.interpolate(
            block_state.processed_mask_image, size=(latent_height, latent_width), mode="nearest"
        )
        channels = components.transformer.config.in_channels // (p**2)
        mask = mask.repeat(1, channels, 1, 1).to(device=components._execution_device, dtype=block_state.dtype)
        batch_size = mask.shape[0]
        mask = mask.view(batch_size, channels, latent_height // p, p, latent_width // p, p)
        mask = mask.permute(0, 2, 4, 1, 3, 5)
        block_state.mask = mask.reshape(batch_size, (latent_height // p) * (latent_width // p), channels * p * p)
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2SetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the Krea 2 flow-matching schedule on the scheduler: a linear sigma schedule with a resolution-aware
    dynamic time shift `mu`.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 28):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          image_seq_len (`int`):
              Number of image tokens, used to compute the resolution-aware shift.

      Outputs:
          timesteps (`Tensor`):
              The denoising timesteps.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that sets the Krea 2 flow-matching schedule on the scheduler: a linear sigma schedule with a "
            "resolution-aware dynamic time shift `mu`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=28),
            InputParam(
                name="sigmas", type_hint=list, description="Custom sigma schedule (defaults to a linear ramp)."
            ),
            InputParam(
                name="image_seq_len",
                required=True,
                type_hint=int,
                description="Number of image tokens, used to compute the resolution-aware shift.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="timesteps", type_hint=torch.Tensor, description="The denoising timesteps.")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        num_inference_steps = block_state.num_inference_steps

        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        else:
            block_state.num_inference_steps = len(sigmas)

        config = components.scheduler.config
        mu = calculate_shift(
            block_state.image_seq_len,
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 6400),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.15),
        )

        components.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
        components.scheduler.set_begin_index(0)
        block_state.timesteps = components.scheduler.timesteps

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2TurboSetTimestepsStep(ModularPipelineBlocks):
    """
    Step that sets the flow-matching schedule for the distilled Krea 2 turbo checkpoint on the scheduler: a linear
    sigma schedule with the fixed time shift `mu=1.15` the checkpoint was distilled with.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).

      Outputs:
          timesteps (`Tensor`):
              The denoising timesteps.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that sets the flow-matching schedule for the distilled Krea 2 turbo checkpoint on the scheduler: a "
            "linear sigma schedule with the fixed time shift `mu=1.15` the checkpoint was distilled with."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("num_inference_steps", default=8),
            InputParam(
                name="sigmas", type_hint=list, description="Custom sigma schedule (defaults to a linear ramp)."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="timesteps", type_hint=torch.Tensor, description="The denoising timesteps.")]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        num_inference_steps = block_state.num_inference_steps

        sigmas = block_state.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        else:
            block_state.num_inference_steps = len(sigmas)

        components.scheduler.set_timesteps(sigmas=sigmas, mu=1.15, device=device)
        components.scheduler.set_begin_index(0)
        block_state.timesteps = components.scheduler.timesteps

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2PreparePositionIdsStep(ModularPipelineBlocks):
    """
    Step that builds the shared rotary position ids for the combined [text | image] sequence: text at the origin, image
    tokens at their (0, h, w) latent-grid coordinates. Place after prepare_latents.

      Inputs:
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          prompt_embeds (`Tensor`):
              Batch-expanded text features (only text_seq_len is used).

      Outputs:
          position_ids (`Tensor`):
              Shared rotary coordinates (text_seq_len + grid_h * grid_w, 3).
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return (
            "Step that builds the shared rotary position ids for the combined [text | image] sequence: text at the "
            "origin, image tokens at their (0, h, w) latent-grid coordinates. Place after prepare_latents."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Batch-expanded text features (only text_seq_len is used).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="position_ids",
                type_hint=torch.Tensor,
                description="Shared rotary coordinates (text_seq_len + grid_h * grid_w, 3).",
            )
        ]

    @staticmethod
    # Copied from diffusers.pipelines.krea2.pipeline_krea2.Krea2Pipeline.prepare_position_ids
    def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device):
        """Build the `(text_seq_len + grid_height * grid_width, 3)` rotary coordinates for the combined sequence:
        text tokens sit at the origin, image tokens carry their `(0, h, w)` latent-grid coordinates."""
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        image_ids = image_ids.reshape(grid_height * grid_width, 3)
        return torch.cat([text_ids, image_ids], dim=0)

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        p = components.patch_size
        grid_h = block_state.height // (components.vae_scale_factor * p)
        grid_w = block_state.width // (components.vae_scale_factor * p)
        text_seq_len = block_state.prompt_embeds.shape[1]

        block_state.position_ids = self.prepare_position_ids(text_seq_len, grid_h, grid_w, device)

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class Krea2PrepareReferencePositionIdsStep(ModularPipelineBlocks):
    """
    Build rotary position ids for a [text | reference | target] Krea 2 sequence.

      Inputs:
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          prompt_embeds (`Tensor`):
              Batch-expanded text features.
          reference_image_latents (`list`):
              Packed reference-image latents in conditioning order.

      Outputs:
          position_ids (`Tensor`):
              Rotary coordinates for the [text | reference | target] sequence.
    """

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Build rotary position ids for a [text | reference | target] Krea 2 sequence."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam(
                name="prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Batch-expanded text features.",
            ),
            InputParam(
                name="reference_image_latents",
                required=True,
                type_hint=list[torch.Tensor],
                description="Packed reference-image latents in conditioning order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="position_ids",
                type_hint=torch.Tensor,
                description="Rotary coordinates for the [text | reference | target] sequence.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: Krea2ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        p = components.patch_size
        grid_height = block_state.height // (components.vae_scale_factor * p)
        grid_width = block_state.width // (components.vae_scale_factor * p)
        image_seq_len = grid_height * grid_width
        if any(reference.shape[1] != image_seq_len for reference in block_state.reference_image_latents):
            reference_lengths = [reference.shape[1] for reference in block_state.reference_image_latents]
            raise ValueError(
                "Each packed reference image and the target must have the same token count, but got reference "
                f"lengths {reference_lengths} and target length {image_seq_len}."
            )

        text_ids = torch.zeros(block_state.prompt_embeds.shape[1], 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        reference_ids = []
        for frame in range(1, len(block_state.reference_image_latents) + 1):
            ids = image_ids.clone()
            ids[..., 0] = frame
            reference_ids.append(ids.reshape(image_seq_len, 3))
        target_ids = image_ids.clone()
        target_ids[..., 0] = 0
        block_state.position_ids = torch.cat([text_ids, *reference_ids, target_ids.reshape(image_seq_len, 3)], dim=0)
        self.set_block_state(state, block_state)
        return components, state
