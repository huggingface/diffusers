# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import html
from typing import List, Optional, Union

import regex as re
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import FluxLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL
from ...utils import USE_PEFT_BACKEND, is_ftfy_available, logging, scale_lora_layers, unscale_lora_layers
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import FluxModularPipeline


if is_ftfy_available():
    import ftfy


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


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


def encode_vae_image(vae: AutoencoderKL, image: torch.Tensor, generator: torch.Generator, sample_mode="sample"):
    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i], sample_mode=sample_mode)
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode=sample_mode)

    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor

    return image_latents


class FluxProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "Flux"

    @property
    def description(self) -> str:
        return "Image Preprocess step. Resizing is needed in Flux Kontext (will be implemented later.)"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [InputParam("resized_image"), InputParam("image"), InputParam("height"), InputParam("width")]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="processed_image"),
        ]

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):
        if height is not None and height % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * 2} but is {height}")

        if width is not None and width % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * 2} but is {width}")

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        if block_state.resized_image is None and block_state.image is None:
            raise ValueError("`resized_image` and `image` cannot be None at the same time")

        if block_state.resized_image is None:
            image = block_state.image
            self.check_inputs(
                height=block_state.height, width=block_state.width, vae_scale_factor=components.vae_scale_factor
            )
            height = block_state.height or components.default_height
            width = block_state.width or components.default_width
        else:
            width, height = block_state.resized_image[0].size
            image = block_state.resized_image

        block_state.processed_image = components.image_processor.preprocess(image=image, height=height, width=width)

        self.set_block_state(state, block_state)
        return components, state


class FluxVaeEncoderDynamicStep(ModularPipelineBlocks):
    model_name = "flux"

    def __init__(
        self,
        input_name: str = "processed_image",
        output_name: str = "image_latents",
    ):
        """Initialize a VAE encoder step for converting images to latent representations.

        Both the input and output names are configurable so this block can be configured to process to different image
        inputs (e.g., "processed_image" -> "image_latents", "processed_control_image" -> "control_image_latents").

        Args:
            input_name (str, optional): Name of the input image tensor. Defaults to "processed_image".
                Examples: "processed_image" or "processed_control_image"
            output_name (str, optional): Name of the output latent tensor. Defaults to "image_latents".
                Examples: "image_latents" or "control_image_latents"

        Examples:
            # Basic usage with default settings (includes image processor): # FluxImageVaeEncoderDynamicStep()

            # Custom input/output names for control image: # FluxImageVaeEncoderDynamicStep(
                input_name="processed_control_image", output_name="control_image_latents"
            )
        """
        self._image_input_name = input_name
        self._image_latents_output_name = output_name
        super().__init__()

    @property
    def description(self) -> str:
        return f"Dynamic VAE Encoder step that converts {self._image_input_name} into latent representations {self._image_latents_output_name}.\n"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        components = [ComponentSpec("vae", AutoencoderKL)]
        return components

    @property
    def inputs(self) -> List[InputParam]:
        inputs = [InputParam(self._image_input_name, required=True), InputParam("generator")]
        return inputs

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                self._image_latents_output_name,
                type_hint=torch.Tensor,
                description="The latents representing the reference image",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.vae.dtype

        image = getattr(block_state, self._image_input_name)
        image = image.to(device=device, dtype=dtype)

        # Encode image into latents
        image_latents = encode_vae_image(image=image, vae=components.vae, generator=block_state.generator)
        setattr(block_state, self._image_latents_output_name, image_latents)

        self.set_block_state(state, block_state)

        return components, state


class FluxTextEncoderStep(ModularPipelineBlocks):
    model_name = "flux"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the image generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", CLIPTextModel),
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("text_encoder_2", T5EncoderModel),
            ComponentSpec("tokenizer_2", T5TokenizerFast),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("prompt_2"),
            InputParam("max_sequence_length", type_hint=int, default=512, required=False),
            InputParam("joint_attention_kwargs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="pooled text embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        for prompt in [block_state.prompt, block_state.prompt_2]:
            if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
                raise ValueError(f"`prompt` or `prompt_2` has to be of type `str` or `list` but is {type(prompt)}")

    @staticmethod
    def _get_t5_prompt_embeds(
        components, prompt: Union[str, List[str]], max_sequence_length: int, device: torch.device
    ):
        dtype = components.text_encoder_2.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if isinstance(components, TextualInversionLoaderMixin):
            prompt = components.maybe_convert_prompt(prompt, components.tokenizer_2)

        text_inputs = components.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        untruncated_ids = components.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = components.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = components.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    @staticmethod
    def _get_clip_prompt_embeds(components, prompt: Union[str, List[str]], device: torch.device):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if isinstance(components, TextualInversionLoaderMixin):
            prompt = components.maybe_convert_prompt(prompt, components.tokenizer)

        text_inputs = components.tokenizer(
            prompt,
            padding="max_length",
            max_length=components.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        tokenizer_max_length = components.tokenizer.model_max_length
        untruncated_ids = components.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = components.tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = components.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=components.text_encoder.dtype, device=device)

        return prompt_embeds

    @staticmethod
    def encode_prompt(
        components,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        device = device or components._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(components, FluxLoraLoaderMixin):
            components._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if components.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(components.text_encoder, lora_scale)
            if components.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(components.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = FluxTextEncoderStep._get_clip_prompt_embeds(
                components,
                prompt=prompt,
                device=device,
            )
            prompt_embeds = FluxTextEncoderStep._get_t5_prompt_embeds(
                components,
                prompt=prompt_2,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if components.text_encoder is not None:
            if isinstance(components, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(components.text_encoder, lora_scale)

        if components.text_encoder_2 is not None:
            if isinstance(components, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(components.text_encoder_2, lora_scale)

        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: FluxModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.device = components._execution_device

        # Encode input prompt
        block_state.text_encoder_lora_scale = (
            block_state.joint_attention_kwargs.get("scale", None)
            if block_state.joint_attention_kwargs is not None
            else None
        )
        block_state.prompt_embeds, block_state.pooled_prompt_embeds = self.encode_prompt(
            components,
            prompt=block_state.prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=block_state.device,
            max_sequence_length=block_state.max_sequence_length,
            lora_scale=block_state.text_encoder_lora_scale,
        )

        # Add outputs
        self.set_block_state(state, block_state)
        return components, state
