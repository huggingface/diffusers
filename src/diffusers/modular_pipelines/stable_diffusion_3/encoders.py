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

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import SD3LoraLoaderMixin
from ...models import AutoencoderKL
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import StableDiffusion3ModularPipeline


logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def encode_vae_image(
    vae: AutoencoderKL,
    image: torch.Tensor,
    generator: torch.Generator,
    sample_mode="sample",
):
    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(
                vae.encode(image[i : i + 1]),
                generator=generator[i],
                sample_mode=sample_mode,
            )
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode=sample_mode)

    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return image_latents


def _get_t5_prompt_embeds(
    text_encoder: T5EncoderModel | None,
    tokenizer: T5TokenizerFast | None,
    prompt: str | list[str] = None,
    max_sequence_length: int = 256,
    device: torch.device | None = None,
    joint_attention_dim: int = 4096,
    dtype: torch.dtype | None = None,
):
    device = device or (text_encoder.device if text_encoder is not None else torch.device("cpu"))
    dtype = dtype or (text_encoder.dtype if text_encoder is not None else torch.float32)

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if text_encoder is None or tokenizer is None:
        return torch.zeros(
            (batch_size, max_sequence_length, joint_attention_dim),
            device=device,
            dtype=dtype,
        )

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        logger.warning(
            f"The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds


def _get_clip_prompt_embeds(
    text_encoder: CLIPTextModelWithProjection | None,
    tokenizer: CLIPTokenizer | None,
    prompt: str | list[str],
    device: torch.device | None = None,
    clip_skip: int | None = None,
    hidden_size: int = 768,
    dtype: torch.dtype | None = None,
):
    device = device or (text_encoder.device if text_encoder is not None else torch.device("cpu"))
    dtype = dtype or (text_encoder.dtype if text_encoder is not None else torch.float32)

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if text_encoder is None or tokenizer is None:
        prompt_embeds = torch.zeros((batch_size, 77, hidden_size), device=device, dtype=dtype)
        pooled_prompt_embeds = torch.zeros((batch_size, hidden_size), device=device, dtype=dtype)
        return prompt_embeds, pooled_prompt_embeds

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        logger.warning(
            f"The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]

    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    components,
    prompt: str | list[str],
    prompt_2: str | list[str] | None = None,
    prompt_3: str | list[str] | None = None,
    device: torch.device | None = None,
    negative_prompt: str | list[str] | None = None,
    negative_prompt_2: str | list[str] | None = None,
    negative_prompt_3: str | list[str] | None = None,
    clip_skip: int | None = None,
    max_sequence_length: int = 256,
    lora_scale: float | None = None,
):
    device = device or components._execution_device

    expected_dtype = None
    if components.text_encoder is not None:
        expected_dtype = components.text_encoder.dtype
    elif components.text_encoder_2 is not None:
        expected_dtype = components.text_encoder_2.dtype
    elif getattr(components, "transformer", None) is not None:
        expected_dtype = components.transformer.dtype
    else:
        expected_dtype = torch.float32

    if lora_scale is not None and isinstance(components, SD3LoraLoaderMixin):
        components._lora_scale = lora_scale
        if components.text_encoder is not None and USE_PEFT_BACKEND:
            scale_lora_layers(components.text_encoder, lora_scale)
        if components.text_encoder_2 is not None and USE_PEFT_BACKEND:
            scale_lora_layers(components.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt_2 = prompt_2 or prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    prompt_3 = prompt_3 or prompt
    prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

    prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
        components.text_encoder,
        components.tokenizer,
        prompt=prompt,
        device=device,
        clip_skip=clip_skip,
        hidden_size=768,
        dtype=expected_dtype,
    )
    prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
        components.text_encoder_2,
        components.tokenizer_2,
        prompt=prompt_2,
        device=device,
        clip_skip=clip_skip,
        hidden_size=1280,
        dtype=expected_dtype,
    )
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

    t5_prompt_embed = _get_t5_prompt_embeds(
        components.text_encoder_3,
        components.tokenizer_3,
        prompt=prompt_3,
        max_sequence_length=max_sequence_length,
        device=device,
        joint_attention_dim=(
            components.transformer.config.joint_attention_dim
            if getattr(components, "transformer", None) is not None
            else 4096
        ),
        dtype=expected_dtype,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

    negative_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    if negative_prompt is not None or negative_prompt_2 is not None or negative_prompt_3 is not None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt
        negative_prompt_3 = negative_prompt_3 or negative_prompt

        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_2 = (
            batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        )
        negative_prompt_3 = (
            batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
        )

        negative_prompt_embed, negative_pooled_prompt_embed = _get_clip_prompt_embeds(
            components.text_encoder,
            components.tokenizer,
            prompt=negative_prompt,
            device=device,
            clip_skip=None,
            hidden_size=768,
            dtype=expected_dtype,
        )
        negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
            components.text_encoder_2,
            components.tokenizer_2,
            prompt=negative_prompt_2,
            device=device,
            clip_skip=None,
            hidden_size=1280,
            dtype=expected_dtype,
        )
        negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        t5_negative_prompt_embed = _get_t5_prompt_embeds(
            components.text_encoder_3,
            components.tokenizer_3,
            prompt=negative_prompt_3,
            max_sequence_length=max_sequence_length,
            device=device,
            joint_attention_dim=(
                components.transformer.config.joint_attention_dim
                if getattr(components, "transformer", None) is not None
                else 4096
            ),
            dtype=expected_dtype,
        )

        negative_clip_prompt_embeds = torch.nn.functional.pad(
            negative_clip_prompt_embeds,
            (
                0,
                t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1],
            ),
        )
        negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
        negative_pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
        )

    if components.text_encoder is not None and isinstance(components, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
        unscale_lora_layers(components.text_encoder, lora_scale)
    if components.text_encoder_2 is not None and isinstance(components, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
        unscale_lora_layers(components.text_encoder_2, lora_scale)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


class StableDiffusion3ProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def description(self) -> str:
        return "Image Preprocess step for SD3."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8, "vae_latent_channels": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return[
            InputParam(
                "image",
                description="The input image to be used as the starting point for the image-to-image process.",
            ),
            InputParam("height", description="The height in pixels of the generated image."),
            InputParam("width", description="The width in pixels of the generated image."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam(name="processed_image", description="The pre-processed image tensor.")]

    @staticmethod
    def check_inputs(height, width, vae_scale_factor, patch_size):
        if height is not None and height % (vae_scale_factor * patch_size) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * patch_size} but is {height}")

        if width is not None and width % (vae_scale_factor * patch_size) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * patch_size} but is {width}")

    @torch.no_grad()
    def __call__(
        self, components: StableDiffusion3ModularPipeline, state: PipelineState
    ):
        block_state = self.get_block_state(state)

        if block_state.image is None:
            raise ValueError("`image` cannot be None")

        image = block_state.image
        self.check_inputs(
            height=block_state.height,
            width=block_state.width,
            vae_scale_factor=components.vae_scale_factor,
            patch_size=components.patch_size,
        )
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        block_state.processed_image = components.image_processor.preprocess(
            image=image, height=height, width=width
        )

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusion3VaeEncoderStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    def __init__(
        self,
        input_name: str = "processed_image",
        output_name: str = "image_latents",
        sample_mode: str = "sample",
    ):
        self._image_input_name = input_name
        self._image_latents_output_name = output_name
        self.sample_mode = sample_mode
        super().__init__()

    @property
    def description(self) -> str:
        return f"Dynamic VAE Encoder step that converts {self._image_input_name} into latent representations {self._image_latents_output_name}."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKL)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                self._image_input_name,
                description="The processed image input to be encoded.",
            ),
            InputParam(
                "generator",
                description="One or a list of torch generator(s) to make generation deterministic.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                self._image_latents_output_name,
                type_hint=torch.Tensor,
                description="The latents representing the reference image",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: StableDiffusion3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image = getattr(block_state, self._image_input_name)

        if image is None:
            setattr(block_state, self._image_latents_output_name, None)
        else:
            device = components._execution_device
            dtype = components.vae.dtype
            image = image.to(device=device, dtype=dtype)
            image_latents = encode_vae_image(
                image=image,
                vae=components.vae,
                generator=block_state.generator,
                sample_mode=self.sample_mode,
            )
            setattr(block_state, self._image_latents_output_name, image_latents)

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusion3TextEncoderStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings to guide the image generation for SD3."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", CLIPTextModelWithProjection),
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("text_encoder_2", CLIPTextModelWithProjection),
            ComponentSpec("tokenizer_2", CLIPTokenizer),
            ComponentSpec("text_encoder_3", T5EncoderModel),
            ComponentSpec("tokenizer_3", T5TokenizerFast),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompt",
                description="The prompt or prompts to guide the image generation.",
            ),
            InputParam(
                "prompt_2",
                description="The prompt or prompts to be sent to tokenizer_2 and text_encoder_2.",
            ),
            InputParam(
                "prompt_3",
                description="The prompt or prompts to be sent to tokenizer_3 and text_encoder_3.",
            ),
            InputParam(
                "negative_prompt",
                description="The prompt or prompts not to guide the image generation.",
            ),
            InputParam(
                "negative_prompt_2",
                description="The prompt or prompts not to guide the image generation for tokenizer_2.",
            ),
            InputParam(
                "negative_prompt_3",
                description="The prompt or prompts not to guide the image generation for tokenizer_3.",
            ),
            InputParam(
                "clip_skip",
                type_hint=int,
                description="Number of layers to be skipped from CLIP while computing the prompt embeddings.",
            ),
            InputParam(
                "max_sequence_length",
                type_hint=int,
                default=256,
                description="Maximum sequence length to use with the prompt.",
            ),
            InputParam(
                "joint_attention_kwargs",
                description="A kwargs dictionary passed along to the AttentionProcessor.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt_embeds", type_hint=torch.Tensor),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor),
            OutputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(
        self, components: StableDiffusion3ModularPipeline, state: PipelineState
    ) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        lora_scale = (
            block_state.joint_attention_kwargs.get("scale", None)
            if block_state.joint_attention_kwargs
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = encode_prompt(
            components=components,
            prompt=block_state.prompt,
            prompt_2=block_state.prompt_2,
            prompt_3=block_state.prompt_3,
            device=block_state.device,
            negative_prompt=block_state.negative_prompt,
            negative_prompt_2=block_state.negative_prompt_2,
            negative_prompt_3=block_state.negative_prompt_3,
            clip_skip=block_state.clip_skip,
            max_sequence_length=block_state.max_sequence_length,
            lora_scale=lora_scale,
        )

        block_state.prompt_embeds = prompt_embeds
        block_state.negative_prompt_embeds = negative_prompt_embeds
        block_state.pooled_prompt_embeds = pooled_prompt_embeds
        block_state.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

        self.set_block_state(state, block_state)
        return components, state
