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
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import SD3LoraLoaderMixin
from ...models import AutoencoderKL
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import SD3ModularPipeline


logger = logging.get_logger(__name__)

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

def encode_vae_image(vae: AutoencoderKL, image: torch.Tensor, generator: torch.Generator, sample_mode="sample"):
    if isinstance(generator, list):
        image_latents =[
            retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i], sample_mode=sample_mode)
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode=sample_mode)

    image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return image_latents

class SD3ProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def description(self) -> str:
        return "Image Preprocess step for SD3."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return[
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8, "vae_latent_channels": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return[InputParam("resized_image"), InputParam("image"), InputParam("height"), InputParam("width")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return[OutputParam(name="processed_image")]

    @staticmethod
    def check_inputs(height, width, vae_scale_factor, patch_size):
        if height is not None and height % (vae_scale_factor * patch_size) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * patch_size} but is {height}")

        if width is not None and width % (vae_scale_factor * patch_size) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * patch_size} but is {width}")

    @torch.no_grad()
    def __call__(self, components: SD3ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        if block_state.resized_image is None and block_state.image is None:
            raise ValueError("`resized_image` and `image` cannot be None at the same time")

        if block_state.resized_image is None:
            image = block_state.image
            self.check_inputs(
                height=block_state.height, width=block_state.width,
                vae_scale_factor=components.vae_scale_factor, patch_size=components.patch_size
            )
            height = block_state.height or components.default_height
            width = block_state.width or components.default_width
        else:
            width, height = block_state.resized_image[0].size
            image = block_state.resized_image

        block_state.processed_image = components.image_processor.preprocess(image=image, height=height, width=width)

        self.set_block_state(state, block_state)
        return components, state

class SD3VaeEncoderStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    def __init__(self, input_name: str = "processed_image", output_name: str = "image_latents", sample_mode: str = "sample"):
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
        return[InputParam(self._image_input_name), InputParam("generator")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return[
            OutputParam(self._image_latents_output_name, type_hint=torch.Tensor, description="The latents representing the reference image")
        ]

    @torch.no_grad()
    def __call__(self, components: SD3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image = getattr(block_state, self._image_input_name)

        if image is None:
            setattr(block_state, self._image_latents_output_name, None)
        else:
            device = components._execution_device
            dtype = components.vae.dtype
            image = image.to(device=device, dtype=dtype)
            image_latents = encode_vae_image(
                image=image, vae=components.vae, generator=block_state.generator, sample_mode=self.sample_mode
            )
            setattr(block_state, self._image_latents_output_name, image_latents)

        self.set_block_state(state, block_state)
        return components, state

class SD3TextEncoderStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-3"

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings to guide the image generation for SD3."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return[
            ComponentSpec("text_encoder", CLIPTextModelWithProjection),
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("text_encoder_2", CLIPTextModelWithProjection),
            ComponentSpec("tokenizer_2", CLIPTokenizer),
            ComponentSpec("text_encoder_3", T5EncoderModel),
            ComponentSpec("tokenizer_3", T5TokenizerFast),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return[
            InputParam("prompt"),
            InputParam("prompt_2"),
            InputParam("prompt_3"),
            InputParam("negative_prompt"),
            InputParam("negative_prompt_2"),
            InputParam("negative_prompt_3"),
            InputParam("prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("pooled_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
            InputParam("guidance_scale", default=7.0),
            InputParam("clip_skip", type_hint=int),
            InputParam("max_sequence_length", type_hint=int, default=256),
            InputParam("joint_attention_kwargs"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return[
            OutputParam("prompt_embeds", type_hint=torch.Tensor),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor),
            OutputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
        ]

    @staticmethod
    def _get_t5_prompt_embeds(components, prompt, max_sequence_length, device):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if components.text_encoder_3 is None:
            return torch.zeros(
                (batch_size, max_sequence_length, components.transformer.config.joint_attention_dim),
                device=device,
                dtype=components.text_encoder.dtype,
            )

        text_inputs = components.tokenizer_3(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_embeds = components.text_encoder_3(text_inputs.input_ids.to(device))[0]
        return prompt_embeds.to(dtype=components.text_encoder_3.dtype, device=device)

    @staticmethod
    def _get_clip_prompt_embeds(components, prompt, device, clip_skip, clip_model_index):
        clip_tokenizers = [components.tokenizer, components.tokenizer_2]
        clip_text_encoders =[components.text_encoder, components.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )

        prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        return prompt_embeds.to(dtype=components.text_encoder.dtype, device=device), pooled_prompt_embeds

    @staticmethod
    def encode_prompt(components, block_state, device, do_classifier_free_guidance, lora_scale):
        if lora_scale is not None and isinstance(components, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
            if components.text_encoder is not None:
                scale_lora_layers(components.text_encoder, lora_scale)
            if components.text_encoder_2 is not None:
                scale_lora_layers(components.text_encoder_2, lora_scale)

        prompt_embeds = block_state.prompt_embeds
        pooled_prompt_embeds = block_state.pooled_prompt_embeds

        if prompt_embeds is None:
            prompt = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
            prompt_2 = block_state.prompt_2 or prompt
            prompt_3 = block_state.prompt_3 or prompt

            prompt_embed, pooled_embed = SD3TextEncoderStep._get_clip_prompt_embeds(components, prompt, device, block_state.clip_skip, 0)
            prompt_2_embed, pooled_2_embed = SD3TextEncoderStep._get_clip_prompt_embeds(components, prompt_2, device, block_state.clip_skip, 1)
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = SD3TextEncoderStep._get_t5_prompt_embeds(components, prompt_3, block_state.max_sequence_length, device)
            clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_embed, pooled_2_embed], dim=-1)

        negative_prompt_embeds = block_state.negative_prompt_embeds
        negative_pooled_prompt_embeds = block_state.negative_pooled_prompt_embeds

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            batch_size = prompt_embeds.shape[0]
            neg_prompt = block_state.negative_prompt or ""
            neg_prompt_2 = block_state.negative_prompt_2 or neg_prompt
            neg_prompt_3 = block_state.negative_prompt_3 or neg_prompt

            neg_prompt = batch_size * [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt
            neg_prompt_2 = batch_size * [neg_prompt_2] if isinstance(neg_prompt_2, str) else neg_prompt_2
            neg_prompt_3 = batch_size * [neg_prompt_3] if isinstance(neg_prompt_3, str) else neg_prompt_3

            neg_embed, neg_pooled_embed = SD3TextEncoderStep._get_clip_prompt_embeds(components, neg_prompt, device, None, 0)
            neg_2_embed, neg_2_pooled_embed = SD3TextEncoderStep._get_clip_prompt_embeds(components, neg_prompt_2, device, None, 1)
            neg_clip_embeds = torch.cat([neg_embed, neg_2_embed], dim=-1)

            t5_neg_embed = SD3TextEncoderStep._get_t5_prompt_embeds(components, neg_prompt_3, block_state.max_sequence_length, device)
            neg_clip_embeds = torch.nn.functional.pad(neg_clip_embeds, (0, t5_neg_embed.shape[-1] - neg_clip_embeds.shape[-1]))

            negative_prompt_embeds = torch.cat([neg_clip_embeds, t5_neg_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat([neg_pooled_embed, neg_2_pooled_embed], dim=-1)

        if lora_scale is not None and isinstance(components, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
            if components.text_encoder is not None:
                unscale_lora_layers(components.text_encoder, lora_scale)
            if components.text_encoder_2 is not None:
                unscale_lora_layers(components.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: SD3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device

        do_classifier_free_guidance = block_state.guidance_scale > 1.0
        lora_scale = block_state.joint_attention_kwargs.get("scale", None) if block_state.joint_attention_kwargs else None

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            components, block_state, block_state.device, do_classifier_free_guidance, lora_scale
        )

        block_state.prompt_embeds = prompt_embeds
        block_state.negative_prompt_embeds = negative_prompt_embeds
        block_state.pooled_prompt_embeds = pooled_prompt_embeds
        block_state.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

        self.set_block_state(state, block_state)
        return components, state
