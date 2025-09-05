# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import Dict, List, Optional, Union

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import InpaintProcessor, VaeImageProcessor
from ...models import AutoencoderKLQwenImage
from ...pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline


logger = logging.get_logger(__name__)


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
    return split_result


def get_qwen_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]] = None,
    prompt_template_encode: str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    prompt_template_encode_start_idx: int = 34,
    tokenizer_max_length: int = 1024,
    device: Optional[torch.device] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]
    txt_tokens = tokenizer(
        txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    encoder_hidden_states = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    split_hidden_states = _extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(device=device)

    return prompt_embeds, encoder_attention_mask


def get_qwen_prompt_embeds_edit(
    text_encoder,
    processor,
    prompt: Union[str, List[str]] = None,
    image: Optional[torch.Tensor] = None,
    prompt_template_encode: str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n",
    prompt_template_encode_start_idx: int = 64,
    device: Optional[torch.device] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]

    model_inputs = processor(
        text=txt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(device=device)

    return prompt_embeds, encoder_attention_mask


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


# Modified from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._encode_vae_image
def encode_vae_image(
    image: torch.Tensor, vae: AutoencoderKLQwenImage, generator: torch.Generator, latent_channels: int = 16
):
    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode="argmax")
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std

    return image_latents


class QwenImageResizeStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to the target size."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_resize_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="image", required=True, type_hint=torch.Tensor, description="The image to resize"),
            InputParam(name="height"),
            InputParam(name="width"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = block_state.image
        if not isinstance(images, list):
            images = [images]

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        resized_images = [
            components.image_resize_processor.resize(image, height=height, width=width) for image in images
        ]

        block_state.image = resized_images
        self.set_block_state(state, block_state)
        return components, state


class QwenImageEditResizeStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to the target area while maintaining the aspect ratio."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_resize_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="image", required=True, type_hint=torch.Tensor, description="The image to resize"),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = block_state.image
        if not isinstance(images, list):
            images = [images]

        image_width, image_height = images[0].size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_width / image_height)

        resized_images = [
            components.image_resize_processor.resize(image, height=calculated_height, width=calculated_width)
            for image in images
        ]

        block_state.image = resized_images
        self.set_block_state(state, block_state)
        return components, state


class QwenImageTextEncoderStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the image generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen2_5_VLForConditionalGeneration, description="The text encoder to use"),
            ComponentSpec("tokenizer", Qwen2Tokenizer, description="The tokenizer to use"),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return [
            ConfigSpec(
                name="prompt_template_encode",
                default="<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            ),
            ConfigSpec(name="prompt_template_encode_start_idx", default=34),
            ConfigSpec(name="tokenizer_max_length", default=1024),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="prompt", required=True, type_hint=str, description="The prompt to encode"),
            InputParam(name="negative_prompt", type_hint=str, description="The negative prompt to encode"),
            InputParam(
                name="max_sequence_length", type_hint=int, description="The max sequence length to use", default=1024
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The prompt embeddings",
            ),
            OutputParam(
                name="prompt_embeds_mask",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The encoder attention mask",
            ),
            OutputParam(
                name="negative_prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The negative prompt embeddings",
            ),
            OutputParam(
                name="negative_prompt_embeds_mask",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The negative prompt embeddings mask",
            ),
        ]

    @staticmethod
    def check_inputs(prompt, negative_prompt, max_sequence_length):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if (
            negative_prompt is not None
            and not isinstance(negative_prompt, str)
            and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        device = components._execution_device
        self.check_inputs(block_state.prompt, block_state.negative_prompt, block_state.max_sequence_length)

        block_state.prompt_embeds, block_state.prompt_embeds_mask = get_qwen_prompt_embeds(
            components.text_encoder,
            components.tokenizer,
            prompt=block_state.prompt,
            prompt_template_encode=components.config.prompt_template_encode,
            prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
            tokenizer_max_length=components.config.tokenizer_max_length,
            device=device,
        )

        block_state.prompt_embeds = block_state.prompt_embeds[:, : block_state.max_sequence_length]
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask[:, : block_state.max_sequence_length]

        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or ""
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = get_qwen_prompt_embeds(
                components.text_encoder,
                components.tokenizer,
                prompt=negative_prompt,
                prompt_template_encode=components.config.prompt_template_encode,
                prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
                tokenizer_max_length=components.config.tokenizer_max_length,
                device=device,
            )
            block_state.negative_prompt_embeds = block_state.negative_prompt_embeds[
                :, : block_state.max_sequence_length
            ]
            block_state.negative_prompt_embeds_mask = block_state.negative_prompt_embeds_mask[
                :, : block_state.max_sequence_length
            ]

        self.set_block_state(state, block_state)
        return components, state


class QwenImageEditTextEncoderStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Text Encoder step that processes both prompt and image together to generate text embeddings for guiding image generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen2_5_VLForConditionalGeneration),
            ComponentSpec("processor", Qwen2VLProcessor),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 4.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return [
            ConfigSpec(
                name="prompt_template_encode",
                default="<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n",
            ),
            ConfigSpec(name="prompt_template_encode_start_idx", default=64),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="prompt", required=True, type_hint=str, description="The prompt to encode"),
            InputParam(name="negative_prompt", type_hint=str, description="The negative prompt to encode"),
            InputParam(
                name="image",
                required=True,
                type_hint=torch.Tensor,
                description="The image prompt to encode, should be resized using resize step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name="prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The prompt embeddings",
            ),
            OutputParam(
                name="prompt_embeds_mask",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The encoder attention mask",
            ),
            OutputParam(
                name="negative_prompt_embeds",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The negative prompt embeddings",
            ),
            OutputParam(
                name="negative_prompt_embeds_mask",
                kwargs_type="denoiser_input_fields",
                type_hint=torch.Tensor,
                description="The negative prompt embeddings mask",
            ),
        ]

    @staticmethod
    def check_inputs(prompt, negative_prompt):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if (
            negative_prompt is not None
            and not isinstance(negative_prompt, str)
            and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.prompt, block_state.negative_prompt)

        device = components._execution_device

        block_state.prompt_embeds, block_state.prompt_embeds_mask = get_qwen_prompt_embeds_edit(
            components.text_encoder,
            components.processor,
            prompt=block_state.prompt,
            image=block_state.image,
            prompt_template_encode=components.config.prompt_template_encode,
            prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
            device=device,
        )

        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or ""
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = get_qwen_prompt_embeds_edit(
                components.text_encoder,
                components.processor,
                prompt=negative_prompt,
                image=block_state.image,
                prompt_template_encode=components.config.prompt_template_encode,
                prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
                device=device,
            )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageInpaintProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Preprocess step for inpainting task. This processes the image and mask inputs together. Images need to be resized first using either the QwenImageResizeStep or QwenImageEditResizeStep."

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image", required=True),
            InputParam("mask_image", required=True),
            InputParam("padding_mask_crop"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="processed_image"),
            OutputParam(name="processed_mask_image"),
            OutputParam(
                name="mask_overlay_kwargs",
                type_hint=Dict,
                description="The kwargs for the postprocess step to apply the mask overlay",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        width, height = block_state.image[0].size

        block_state.processed_image, block_state.processed_mask_image, block_state.mask_overlay_kwargs = (
            components.image_mask_processor.preprocess(
                image=block_state.image,
                mask=block_state.mask_image,
                height=height,
                width=width,
                padding_mask_crop=block_state.padding_mask_crop,
            )
        )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageVaeEncoderDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    def __init__(
        self,
        input_name: str = "image",
        output_name: str = "image_latents",
        include_image_processor: bool = True,
        **image_processor_kwargs,
    ):
        """Initialize a dynamic VAE encoder step for converting images to latent representations.

        Args:
            input_name (str, optional): Name of the input image tensor. Defaults to "image".
                Examples: "image", "control_image", "reference_image"
            output_name (str, optional): Name of the output latent tensor. Defaults to "image_latents".
                Examples: "image_latents", "control_image_latents", "reference_image_latents"
            include_image_processor (bool, optional): Whether to include preprocessing step before encoding.
                If True, will resize and preprocess the image. If False, expects preprocessed image. Defaults to True.
            **image_processor_kwargs: Additional kwargs to configure the image processor.
                Common options:
                - vae_scale_factor (int): Scale factor for VAE compression. Defaults to 16.
                - do_resize (bool): Whether to resize images. Defaults to True.

        Examples:
            # Basic usage with default settings QwenImageVaeEncoderDynamicStep()

            # Custom input/output names for control image QwenImageVaeEncoderDynamicStep(
                input_name="control_image", output_name="control_image_latents"
            )

            # Without preprocessing (for already preprocessed images)
            QwenImageVaeEncoderDynamicStep(include_image_processor=False)

            # With custom processor configuration QwenImageVaeEncoderDynamicStep(vae_scale_factor=8)
        """
        if not include_image_processor and len(image_processor_kwargs) > 0:
            logger.warning(
                f"these kwargs will be ignored: {image_processor_kwargs} since image_processor is not used in this block"
            )

        self._image_input_name = input_name
        self._image_latents_output_name = output_name
        self._include_image_processor = include_image_processor
        self._image_processor_kwargs = image_processor_kwargs
        super().__init__()

    @property
    def description(self) -> str:
        # Dynamic configuration info
        preprocessor_info = ""
        if self._include_image_processor:
            processor_config = f"vae_scale_factor: {self._image_processor_kwargs.get('vae_scale_factor', 16)}"
            if self._image_processor_kwargs:
                additional_configs = [
                    f"{k}: {v}" for k, v in self._image_processor_kwargs.items() if k != "vae_scale_factor"
                ]
                if additional_configs:
                    processor_config += f", {', '.join(additional_configs)}"
            preprocessor_info = f" (includes preprocessor with config: {processor_config})"
        else:
            preprocessor_info = " (no preprocessor)"

        return (
            f"Dynamic VAE Encoder step that converts {self._image_input_name} into latent representations {self._image_latents_output_name}.\n\n"
            f"Configuration: {preprocessor_info}\n\n"
        )

    @property
    def expected_components(self) -> List[ComponentSpec]:
        components = [
            ComponentSpec("vae", AutoencoderKLQwenImage),
        ]
        if self._include_image_processor:
            image_processor_config = {"vae_scale_factor": 16}
            image_processor_config.update(self._image_processor_kwargs)
            components.append(
                ComponentSpec(
                    f"{self._image_input_name}_processor",
                    VaeImageProcessor,
                    config=FrozenDict(image_processor_config),
                    default_creation_method="from_config",
                )
            )
        return components

    @property
    def inputs(self) -> List[InputParam]:
        inputs = [
            InputParam(self._image_input_name, required=True),
            InputParam("generator"),
        ]
        if self._include_image_processor:
            inputs.append(InputParam("height"))
            inputs.append(InputParam("width"))
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

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):
        if height is not None and height % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * 2} but is {height}")

        if width is not None and width % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * 2} but is {width}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.vae.dtype

        image = getattr(block_state, self._image_input_name)

        if self._include_image_processor:
            image_processor = getattr(components, f"{self._image_input_name}_processor")
            self.check_inputs(block_state.height, block_state.width, components.vae_scale_factor)

            if not image_processor.config.do_resize and (
                block_state.height is not None or block_state.width is not None
            ):
                logger.warning(
                    "height and width are provided but image_processor.config.do_resize is False, these will be ignored"
                )

            height = block_state.height or components.default_height
            width = block_state.width or components.default_width
            image = image_processor.preprocess(image, height=height, width=width)

        image = image.unsqueeze(2)
        image = image.to(device=device, dtype=dtype)

        # Encode image into latents
        image_latents = encode_vae_image(
            image=image,
            vae=components.vae,
            generator=block_state.generator,
            latent_channels=components.num_channels_latents,
        )

        setattr(block_state, self._image_latents_output_name, image_latents)

        self.set_block_state(state, block_state)

        return components, state


class QwenImageInpaintVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    """This step is used for processing image and mask inputs forinpainting tasks. It:
        - Processes and updates `image` and `mask_image`.
        - Creates `image_latents`.

    Components:
        image_processor (`InpaintProcessor`) vae (`AutoencoderKLQwenImage`)

    Inputs:
        image (`None`) mask_image (`None`) height (`None`) width (`None`) padding_mask_crop (`None`, optional)
        generator (`None`, optional):

    New Outputs:
        processed_image (`Tensor`):
            The processed image
        processed_mask_image (`Tensor`):
            The processed mask_imagge
        mask_overlay_kwargs (`Dict`):
            The kwargs for the postprocess step to apply the mask overlay
        image_latents (`Tensor`):
            The latents representing the reference image
    """

    block_classes = [
        QwenImageResizeStep,
        QwenImageInpaintProcessImagesInputStep,
        QwenImageVaeEncoderDynamicStep(
            input_name="processed_image", output_name="image_latents", include_image_processor=False
        ),  # encode
    ]

    block_names = ["resize", "preprocess", "encode"]

    @property
    def description(self) -> str:
        return (
            "This step is used for processing image and mask inputs for inpainting tasks. It:\n"
            " - Resizes the image to the target size, based on `height` and `width`.\n"
            " - Processes and updates `image` and `mask_image`.\n"
            " - Creates `image_latents`."
        )


class QwenImageEditInpaintVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    """This step is used for processing image and mask inputs forinpainting tasks. It:
        - Processes and updates `image` and `mask_image`.
        - Creates `image_latents`.

    Components:
        image_processor (`InpaintProcessor`) vae (`AutoencoderKLQwenImage`)

    Inputs:
        image (`None`) mask_image (`None`) padding_mask_crop (`None`, optional) generator (`None`, optional)

    New Outputs:
        processed_image (`Tensor`):
            The processed image
        processed_mask_image (`Tensor`):
            The processed mask_imagge
        mask_overlay_kwargs (`Dict`):
            The kwargs for the postprocess step to apply the mask overlay
        image_latents (`Tensor`):
            The latents representing the reference image
    """

    block_classes = [
        QwenImageInpaintProcessImagesInputStep,
        QwenImageVaeEncoderDynamicStep(
            input_name="processed_image", output_name="image_latents", include_image_processor=False
        ),  # encode
    ]

    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return (
            "This step is used for processing image and mask inputs for inpainting tasks. It:\n"
            " - Processes and updates `image` and `mask_image`.\n"
            " - Creates `image_latents`."
            " Note: This step expects the image that has already been resized using the `QwenImageEditResizeStep` block."
        )
