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
from PIL import Image
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
from ..modular_pipeline import PipelineBlock, PipelineState
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


def get_clip_prompt_embeds(
    prompt,
    text_encoder,
    tokenizer,
    device,
    clip_skip=None,
    max_length=None,
):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length if max_length is not None else tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    # We are only using the pooled output of the text_encoder_2, which has 2 dimensions
    # (pooled output for text_encoder has 3 dimensions)
    pooled_prompt_embeds = prompt_embeds[0]

    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        # "2" because SDXL always indexes from the penultimate layer.
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

    return prompt_embeds, pooled_prompt_embeds


# Modified from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._encode_vae_image with self -> components
def encode_vae_image(
    image: torch.Tensor, vae: AutoencoderKL, generator: torch.Generator, dtype: torch.dtype, device: torch.device
):
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    image = image.to(device=device, dtype=dtype)

    if vae.config.force_upcast:
        image = image.float()
        vae.to(dtype=torch.float32)

    if isinstance(generator, list) and len(generator) != image.shape[0]:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {image.shape[0]}. Make sure the batch size matches the length of the generators."
        )

    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i]) for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image), generator=generator)

    if vae.config.force_upcast:
        vae.to(dtype)

    image_latents = image_latents.to(dtype)
    if latents_mean is not None and latents_std is not None:
        latents_mean = latents_mean.to(device=device, dtype=dtype)
        latents_std = latents_std.to(device=device, dtype=dtype)
        image_latents = (image_latents - latents_mean) * vae.config.scaling_factor / latents_std
    else:
        image_latents = vae.config.scaling_factor * image_latents

    return image_latents


class StableDiffusionXLIPAdapterStep(PipelineBlock):
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
                required=False,
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
            OutputParam(
                "ip_adapter_embeds",
                type_hint=List[torch.Tensor],
                kwargs_type="guider_input_fields",
                description="IP adapter image embeddings",
            ),
            OutputParam(
                "negative_ip_adapter_embeds",
                type_hint=List[torch.Tensor],
                kwargs_type="guider_input_fields",
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

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        block_state.ip_adapter_embeds = []
        if components.requires_unconditional_embeds:
            block_state.negative_ip_adapter_embeds = []

        if not isinstance(block_state.ip_adapter_image, list):
            block_state.ip_adapter_image = [block_state.ip_adapter_image]

        if len(block_state.ip_adapter_image) != len(components.unet.encoder_hid_proj.image_projection_layers):
            raise ValueError(
                f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(block_state.ip_adapter_image)} images and {len(components.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
            )

        for single_ip_adapter_image, image_proj_layer in zip(
            block_state.ip_adapter_image, components.unet.encoder_hid_proj.image_projection_layers
        ):
            output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
            single_image_embeds, single_negative_image_embeds = self.encode_image(
                components, single_ip_adapter_image, device, 1, output_hidden_state
            )

            block_state.ip_adapter_embeds.append(single_image_embeds[None, :])
            if components.requires_unconditional_embeds:
                block_state.negative_ip_adapter_embeds.append(single_negative_image_embeds[None, :])

        self.set_block_state(state, block_state)
        return components, state


class StableDiffusionXLTextEncoderStep(PipelineBlock):
    model_name = "stable-diffusion-xl"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the image generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", CLIPTextModel, required=False),
            ComponentSpec("text_encoder_2", CLIPTextModelWithProjection),
            ComponentSpec("tokenizer", CLIPTokenizer, required=False),
            ComponentSpec("tokenizer_2", CLIPTokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 7.5}),
                default_creation_method="from_config",
                required=False,
            ),
        ]

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return [ConfigSpec("force_zeros_for_empty_prompt", True)]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt", required=True),
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
                kwargs_type="guider_input_fields",
                description="text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="guider_input_fields",
                description="negative text embeddings used to guide the image generation",
            ),
            OutputParam(
                "pooled_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="guider_input_fields",
                description="pooled text embeddings used to guide the image generation",
            ),
            OutputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="guider_input_fields",
                description="negative pooled text embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(prompt, prompt_2, negative_prompt, negative_prompt_2):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if negative_prompt_2 is not None and (
            not isinstance(negative_prompt_2, str) and not isinstance(negative_prompt_2, list)
        ):
            raise ValueError(f"`negative_prompt_2` has to be of type `str` or `list` but is {type(negative_prompt_2)}")

    @staticmethod
    def encode_prompt(
        components,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        requires_unconditional_embeds: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
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
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        dtype = components.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        batch_size = len(prompt)

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
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(components, StableDiffusionXLLoraLoaderMixin):
            components._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            for text_encoder in text_encoders:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(text_encoder, lora_scale)
                else:
                    scale_lora_layers(text_encoder, lora_scale)

        # Define prompts
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        prompts = [prompt, prompt_2]

        # generate prompt_embeds & pooled_prompt_embeds
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []

        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            if isinstance(components, TextualInversionLoaderMixin):
                prompt = components.maybe_convert_prompt(prompt, tokenizer)

            prompt_embeds, pooled_prompt_embeds = get_clip_prompt_embeds(
                prompt=prompt,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                device=device,
                clip_skip=clip_skip,
                max_length=tokenizer.model_max_length,
            )

            prompt_embeds_list.append(prompt_embeds)
            if pooled_prompt_embeds.ndim == 2:
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.concat(pooled_prompt_embeds_list, dim=0)

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        zero_out_negative_prompt = negative_prompt is None and components.config.force_zeros_for_empty_prompt
        # generate negative_prompt_embeds & negative_pooled_prompt_embeds
        if requires_unconditional_embeds and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif requires_unconditional_embeds:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            if batch_size != len(negative_prompt_2):
                raise ValueError(
                    f"`negative_prompt_2`: {negative_prompt_2} has batch size {len(negative_prompt_2)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt_2` matches"
                    " the batch size of `prompt`."
                )
            uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            negative_pooled_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(components, TextualInversionLoaderMixin):
                    negative_prompt = components.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                negative_prompt_embeds, negative_pooled_prompt_embeds = get_clip_prompt_embeds(
                    prompt=negative_prompt,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    device=device,
                    clip_skip=None,
                    max_length=max_length,
                )
                negative_prompt_embeds_list.append(negative_prompt_embeds)
                if negative_pooled_prompt_embeds.ndim == 2:
                    negative_pooled_prompt_embeds_list.append(negative_pooled_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
            negative_pooled_prompt_embeds = torch.concat(negative_pooled_prompt_embeds_list, dim=0)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype, device=device)
        if requires_unconditional_embeds:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=dtype, device=device)

        for text_encoder in text_encoders:
            if isinstance(components, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)

        self.check_inputs(
            block_state.prompt, block_state.prompt_2, block_state.negative_prompt, block_state.negative_prompt_2
        )

        device = components._execution_device

        # Encode input prompt
        lora_scale = (
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
            prompt=block_state.prompt,
            prompt_2=block_state.prompt_2,
            device=device,
            requires_unconditional_embeds=components.requires_unconditional_embeds,
            negative_prompt=block_state.negative_prompt,
            negative_prompt_2=block_state.negative_prompt_2,
            lora_scale=lora_scale,
            clip_skip=block_state.clip_skip,
        )
        # Add outputs
        self.set_block_state(state, block_state)
        return components, state


class StableDiffusionXLVaeEncoderStep(PipelineBlock):
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
        ]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam("generator"),
            InputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
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

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = block_state.dtype if block_state.dtype is not None else components.vae.dtype

        image = components.image_processor.preprocess(block_state.image)

        # Encode image into latents
        block_state.image_latents = encode_vae_image(
            image=image, vae=components.vae, generator=block_state.generator, dtype=dtype, device=device
        )

        self.set_block_state(state, block_state)

        return components, state


class StableDiffusionXLInpaintVaeEncoderStep(PipelineBlock):
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
        ]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam("dtype", type_hint=torch.dtype, description="The dtype of the model inputs"),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "image_latents", type_hint=torch.Tensor, description="The latents representation of the input image"
            ),
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
            OutputParam(
                "mask",
                type_hint=torch.Tensor,
                description="The mask to apply on the latents for the inpainting generation.",
            ),
        ]

    def check_inputs(self, image, mask_image, padding_mask_crop):
        if padding_mask_crop is not None and not isinstance(image, Image.Image):
            raise ValueError(
                f"The image should be a PIL image when inpainting mask crop, but is of type {type(image)}."
            )

        if padding_mask_crop is not None and not isinstance(mask_image, Image.Image):
            raise ValueError(
                f"The mask image should be a PIL image when inpainting mask crop, but is of type {type(mask_image)}."
            )

    @torch.no_grad()
    def __call__(self, components: StableDiffusionXLModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.image, block_state.mask_image, block_state.padding_mask_crop)

        dtype = block_state.dtype if block_state.dtype is not None else components.vae.dtype
        device = components._execution_device

        height = block_state.height if block_state.height is not None else components.default_height
        width = block_state.width if block_state.width is not None else components.default_width

        if block_state.padding_mask_crop is not None:
            block_state.crops_coords = components.mask_processor.get_crop_region(
                mask_image=block_state.mask_image, width=width, height=height, pad=block_state.padding_mask_crop
            )
            resize_mode = "fill"
        else:
            block_state.crops_coords = None
            resize_mode = "default"

        image = components.image_processor.preprocess(
            block_state.image,
            height=height,
            width=width,
            crops_coords=block_state.crops_coords,
            resize_mode=resize_mode,
        )

        image = image.to(dtype=torch.float32)

        mask_image = components.mask_processor.preprocess(
            block_state.mask_image,
            height=height,
            width=width,
            resize_mode=resize_mode,
            crops_coords=block_state.crops_coords,
        )

        masked_image = image * (mask_image < 0.5)

        # Prepare image latent variables
        block_state.image_latents = encode_vae_image(
            image=image, vae=components.vae, generator=block_state.generator, dtype=dtype, device=device
        )

        # Prepare masked image latent variables
        block_state.masked_image_latents = encode_vae_image(
            image=masked_image, vae=components.vae, generator=block_state.generator, dtype=dtype, device=device
        )

        # resize mask to match the image latents
        _, _, height_latents, width_latents = block_state.image_latents.shape
        block_state.mask = torch.nn.functional.interpolate(
            mask_image,
            size=(height_latents, width_latents),
        )
        block_state.mask = block_state.mask.to(dtype=dtype, device=device)

        self.set_block_state(state, block_state)

        return components, state
