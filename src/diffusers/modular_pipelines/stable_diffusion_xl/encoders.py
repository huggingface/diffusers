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

from typing import List, Optional, Tuple

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import StableDiffusionXLModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


class StableDiffusionXLIPAdapterStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return (
            "IP Adapter step that prepares ip adapter image embeddings.\n"
            "Note that this step only prepares the embeddings - in order for it to work correctly, "
            "you need to load ip adapter weights into unet via ModularPipeline.load_ip_adapter() and pipeline.set_ip_adapter_scale().\n"
            "See [ModularIPAdapterMixin](https://huggingface.co/docs/diffusers/api/loaders/ip_adapter#diffusers.loaders.ModularIPAdapterMixin)"
            " for more details"
        )

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("image_encoder", CLIPVisionModelWithProjection),
            ComponentSpec(
                "feature_extractor",
                CLIPImageProcessor,
                config=FrozenDict({"size": 224, "crop_size": 224}),
                default_creation_method="from_config",
            ),
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "ip_adapter_image",
                PipelineImageInput,
                required=True,
                description="The image(s) to be used as ip adapter",
            )
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("ip_adapter_embeds", type_hint=torch.Tensor, description="IP adapter image embeddings"),
            OutputParam(
                "negative_ip_adapter_embeds",
                type_hint=torch.Tensor,
                description="Negative IP adapter image embeddings",
            ),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image with self->components
    def encode_image(components, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(components.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = components.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = components.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = components.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = components.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self,
        components,
        ip_adapter_image,
        ip_adapter_image_embeds,
        device,
        num_images_per_prompt,
        prepare_unconditional_embeds,
    ):
        image_embeds = []
        if prepare_unconditional_embeds:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(components.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(components.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, components.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    components, single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if prepare_unconditional_embeds:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if prepare_unconditional_embeds:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if prepare_unconditional_embeds:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.prepare_unconditional_embeds = components.guider.num_conditions > 1
        block_state.device = components._execution_device

        block_state.ip_adapter_embeds = self.prepare_ip_adapter_image_embeds(
            components,
            ip_adapter_image=block_state.ip_adapter_image,
            ip_adapter_image_embeds=None,
            device=block_state.device,
            num_images_per_prompt=1,
            prepare_unconditional_embeds=block_state.prepare_unconditional_embeds,
        )
        if block_state.prepare_unconditional_embeds:
            block_state.negative_ip_adapter_embeds = []
            for i, image_embeds in enumerate(block_state.ip_adapter_embeds):
                negative_image_embeds, image_embeds = image_embeds.chunk(2)
                block_state.negative_ip_adapter_embeds.append(negative_image_embeds)
                block_state.ip_adapter_embeds[i] = image_embeds

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusionXLTextEncoderStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the image generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", CLIPTextModel),
            ComponentSpec("text_encoder_2", CLIPTextModelWithProjection),
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("tokenizer_2", CLIPTokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return [ConfigSpec("force_zeros_for_empty_prompt", True)]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("prompt_2"),
            InputParam("negative_prompt"),
            InputParam("negative_prompt_2"),
            InputParam("cross_attention_kwargs"),
            InputParam("clip_skip"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="negative text embeddings used to guide the image generation",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="pooled text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="negative pooled text embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")
        elif block_state.prompt_2 is not None and (
            not isinstance(block_state.prompt_2, str) and not isinstance(block_state.prompt_2, list)
        ):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(block_state.prompt_2)}")

    @staticmethod
    def encode_prompt(
        components,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prepare_unconditional_embeds (`bool`):
                whether to use prepare unconditional embeddings or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or components._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(components, StableDiffusionXLLoraLoaderMixin):
            components._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if components.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(components.text_encoder, lora_scale)
                else:
                    scale_lora_layers(components.text_encoder, lora_scale)

            if components.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(components.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(components.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = (
            [components.tokenizer, components.tokenizer_2]
            if components.tokenizer is not None
            else [components.tokenizer_2]
        )
        text_encoders = (
            [components.text_encoder, components.text_encoder_2]
            if components.text_encoder is not None
            else [components.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(components, TextualInversionLoaderMixin):
                    prompt = components.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and components.config.force_zeros_for_empty_prompt
        if prepare_unconditional_embeds and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif prepare_unconditional_embeds and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
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
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(components, TextualInversionLoaderMixin):
                    negative_prompt = components.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if components.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=components.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=components.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if prepare_unconditional_embeds:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if components.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=components.text_encoder_2.dtype, device=device
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=components.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if prepare_unconditional_embeds:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if components.text_encoder is not None:
            if isinstance(components, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(components.text_encoder, lora_scale)

        if components.text_encoder_2 is not None:
            if isinstance(components, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(components.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.prepare_unconditional_embeds = components.guider.num_conditions > 1
        block_state.device = components._execution_device

        # Encode input prompt
        block_state.text_encoder_lora_scale = (
            block_state.cross_attention_kwargs.get("scale", None)
            if block_state.cross_attention_kwargs is not None
            else None
        )
        (
            block_state.prompt_embeds,
            block_state.negative_prompt_embeds,
            block_state.pooled_prompt_embeds,
            block_state.negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            components,
            block_state.prompt,
            block_state.prompt_2,
            block_state.device,
            1,
            block_state.prepare_unconditional_embeds,
            block_state.negative_prompt,
            block_state.negative_prompt_2,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=block_state.text_encoder_lora_scale,
            clip_skip=block_state.clip_skip,
        )
        # Add outputs
        self.set_block_state(state, block_state)
        return components, state


class StableDiffusionXLVaeEncoderStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "Vae Encoder step that encode the input image into a latent representation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image", required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("generator"),
            InputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
            InputParam(
                "preprocess_kwargs",
                type_hint=Optional[dict],
                description="A kwargs dictionary that if specified is passed along to the `ImageProcessor` as defined under `self.image_processor` in [diffusers.image_processor.VaeImageProcessor]",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "image_latents",
                type_hint=torch.Tensor,
                description="The latents representing the reference image for image-to-image/inpainting generation",
            )
        ]

    # Modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._encode_vae_image with self -> components
    # YiYi TODO: update the _encode_vae_image so that we can use #Coped from
    def _encode_vae_image(self, components, image: torch.Tensor, generator: torch.Generator):
        latents_mean = latents_std = None
        if hasattr(components.vae.config, "latents_mean") and components.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(components.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(components.vae.config, "latents_std") and components.vae.config.latents_std is not None:
            latents_std = torch.tensor(components.vae.config.latents_std).view(1, 4, 1, 1)

        dtype = image.dtype
        if components.vae.config.force_upcast:
            image = image.float()
            components.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(components.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(components.vae.encode(image), generator=generator)

        if components.vae.config.force_upcast:
            components.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=image_latents.device, dtype=dtype)
            latents_std = latents_std.to(device=image_latents.device, dtype=dtype)
            image_latents = (image_latents - latents_mean) * components.vae.config.scaling_factor / latents_std
        else:
            image_latents = components.vae.config.scaling_factor * image_latents

        return image_latents

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.preprocess_kwargs = block_state.preprocess_kwargs or {}
        block_state.device = components._execution_device
        block_state.dtype = block_state.dtype if block_state.dtype is not None else components.vae.dtype

        image = components.image_processor.preprocess(
            block_state.image, height=block_state.height, width=block_state.width, **block_state.preprocess_kwargs
        )
        image = image.to(device=block_state.device, dtype=block_state.dtype)
        block_state.batch_size = image.shape[0]

        # if generator is a list, make sure the length of it matches the length of images (both should be batch_size)
        if isinstance(block_state.generator, list) and len(block_state.generator) != block_state.batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(block_state.generator)}, but requested an effective batch"
                f" size of {block_state.batch_size}. Make sure the batch size matches the length of the generators."
            )

        block_state.image_latents = self._encode_vae_image(components, image=image, generator=block_state.generator)

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLInpaintVaeEncoderStep(ModularPipelineBlocks):
    model_name = "stable-diffusion-xl"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
            ComponentSpec(
                "mask_processor",
                VaeImageProcessor,
                config=FrozenDict(
                    {"do_normalize": False, "vae_scale_factor": 8, "do_binarize": True, "do_convert_grayscale": True}
                ),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Vae encoder step that prepares the image and mask for the inpainting process"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("height"),
            InputParam("width"),
            InputParam("image", required=True),
            InputParam("mask_image", required=True),
            InputParam("padding_mask_crop"),
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "image_latents", type_hint=torch.Tensor, description="The latents representation of the input image"
            ),
            OutputParam("mask", type_hint=torch.Tensor, description="The mask to use for the inpainting process"),
            OutputParam(
                "masked_image_latents",
                type_hint=torch.Tensor,
                description="The masked image latents to use for the inpainting process (only for inpainting-specifid unet)",
            ),
            OutputParam(
                "crops_coords",
                type_hint=Optional[Tuple[int, int]],
                description="The crop coordinates to use for the preprocess/postprocess of the image and mask",
            ),
        ]

    # Modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._encode_vae_image with self -> components
    # YiYi TODO: update the _encode_vae_image so that we can use #Coped from
    def _encode_vae_image(self, components, image: torch.Tensor, generator: torch.Generator):
        latents_mean = latents_std = None
        if hasattr(components.vae.config, "latents_mean") and components.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(components.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(components.vae.config, "latents_std") and components.vae.config.latents_std is not None:
            latents_std = torch.tensor(components.vae.config.latents_std).view(1, 4, 1, 1)

        dtype = image.dtype
        if components.vae.config.force_upcast:
            image = image.float()
            components.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(components.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(components.vae.encode(image), generator=generator)

        if components.vae.config.force_upcast:
            components.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=image_latents.device, dtype=dtype)
            latents_std = latents_std.to(device=image_latents.device, dtype=dtype)
            image_latents = (image_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
        else:
            image_latents = components.vae.config.scaling_factor * image_latents

        return image_latents

    # modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline.prepare_mask_latents
    # do not accept do_classifier_free_guidance
    def prepare_mask_latents(
        self, components, mask, masked_image, batch_size, height, width, dtype, device, generator
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // components.vae_scale_factor, width // components.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(components, masked_image, generator=generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        block_state.dtype = block_state.dtype if block_state.dtype is not None else components.vae.dtype
        block_state.device = components._execution_device

        if block_state.height is None:
            block_state.height = components.default_height
        if block_state.width is None:
            block_state.width = components.default_width

        if block_state.padding_mask_crop is not None:
            block_state.crops_coords = components.mask_processor.get_crop_region(
                block_state.mask_image, block_state.width, block_state.height, pad=block_state.padding_mask_crop
            )
            block_state.resize_mode = "fill"
        else:
            block_state.crops_coords = None
            block_state.resize_mode = "default"

        image = components.image_processor.preprocess(
            block_state.image,
            height=block_state.height,
            width=block_state.width,
            crops_coords=block_state.crops_coords,
            resize_mode=block_state.resize_mode,
        )
        image = image.to(dtype=torch.float32)

        mask = components.mask_processor.preprocess(
            block_state.mask_image,
            height=block_state.height,
            width=block_state.width,
            resize_mode=block_state.resize_mode,
            crops_coords=block_state.crops_coords,
        )
        block_state.masked_image = image * (mask < 0.5)

        block_state.batch_size = image.shape[0]
        image = image.to(device=block_state.device, dtype=block_state.dtype)
        block_state.image_latents = self._encode_vae_image(components, image=image, generator=block_state.generator)

        # 7. Prepare mask latent variables
        block_state.mask, block_state.masked_image_latents = self.prepare_mask_latents(
            components,
            mask,
            block_state.masked_image,
            block_state.batch_size,
            block_state.height,
            block_state.width,
            block_state.dtype,
            block_state.device,
            block_state.generator,
        )

        self.set_block_state(state, block_state)

        return components, state
