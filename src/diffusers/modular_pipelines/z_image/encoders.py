# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union

import PIL
import torch
from transformers import Qwen2Tokenizer, Qwen3Model

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL
from ...utils import is_ftfy_available, logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import ZImageModularPipeline


if is_ftfy_available():
    pass

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_qwen_prompt_embeds(
    text_encoder: Qwen3Model,
    tokenizer: Qwen2Tokenizer,
    prompt: Union[str, List[str]],
    device: torch.device,
    max_sequence_length: int = 512,
) -> List[torch.Tensor]:
    prompt = [prompt] if isinstance(prompt, str) else prompt

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_list = []

    for i in range(len(prompt_embeds)):
        prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])

    return prompt_embeds_list


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
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
    image_tensor: torch.Tensor,
    vae: AutoencoderKL,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    latent_channels: int = 16,
):
    if not isinstance(image_tensor, torch.Tensor):
        raise ValueError(f"Expected image_tensor to be a tensor, got {type(image_tensor)}.")

    if isinstance(generator, list) and len(generator) != image_tensor.shape[0]:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but it is not same as number of images {image_tensor.shape[0]}."
        )

    image_tensor = image_tensor.to(device=device, dtype=dtype)

    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(vae.encode(image_tensor[i : i + 1]), generator=generator[i])
            for i in range(image_tensor.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image_tensor), generator=generator)

    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor

    return image_latents


class ZImageTextEncoderStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3Model),
            ComponentSpec("tokenizer", Qwen2Tokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0, "enabled": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam("max_sequence_length", default=512),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=List[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=List[torch.Tensor],
                kwargs_type="denoiser_input_fields",
                description="negative text embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

    @staticmethod
    def encode_prompt(
        components,
        prompt: str,
        device: Optional[torch.device] = None,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: Optional[str] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            prepare_unconditional_embeds (`bool`):
                whether to use prepare unconditional embeddings or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of text tokens to be used for the generation process.
        """
        device = device or components._execution_device
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = len(prompt)

        prompt_embeds = get_qwen_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        negative_prompt_embeds = None
        if prepare_unconditional_embeds:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = get_qwen_prompt_embeds(
                text_encoder=components.text_encoder,
                tokenizer=components.tokenizer,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device

        # Encode input prompt
        (
            block_state.prompt_embeds,
            block_state.negative_prompt_embeds,
        ) = self.encode_prompt(
            components=components,
            prompt=block_state.prompt,
            device=block_state.device,
            prepare_unconditional_embeds=components.requires_unconditional_embeds,
            negative_prompt=block_state.negative_prompt,
            max_sequence_length=block_state.max_sequence_length,
        )

        # Add outputs
        self.set_block_state(state, block_state)
        return components, state


class ZImageVaeImageEncoderStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def description(self) -> str:
        return "Vae Image Encoder step that generate condition_latents based on image to guide the image generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8 * 2}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image", type_hint=PIL.Image.Image, required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="video latent representation with the first frame image condition",
            ),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        if (block_state.height is not None and block_state.height % components.vae_scale_factor_spatial != 0) or (
            block_state.width is not None and block_state.width % components.vae_scale_factor_spatial != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {components.vae_scale_factor_spatial} but are {block_state.height} and {block_state.width}."
            )

    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        image = block_state.image

        device = components._execution_device
        dtype = torch.float32
        vae_dtype = components.vae.dtype

        image_tensor = components.image_processor.preprocess(
            image, height=block_state.height, width=block_state.width
        ).to(device=device, dtype=dtype)

        block_state.image_latents = encode_vae_image(
            image_tensor=image_tensor,
            vae=components.vae,
            generator=block_state.generator,
            device=device,
            dtype=vae_dtype,
            latent_channels=components.num_channels_latents,
        )

        self.set_block_state(state, block_state)
        return components, state
