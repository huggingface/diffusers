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

import PIL
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import InpaintProcessor, VaeImageProcessor, is_valid_image, is_valid_image_imagelist
from ...models import AutoencoderKLQwenImage, QwenImageControlNetModel, QwenImageMultiControlNetModel
from ...pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions
from ...utils import logging
from ...utils.torch_utils import unwrap_module
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
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


def get_qwen_prompt_embeds_edit_plus(
    text_encoder,
    processor,
    prompt: Union[str, List[str]] = None,
    image: Optional[Union[torch.Tensor, List[PIL.Image.Image], PIL.Image.Image]] = None,
    prompt_template_encode: str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    img_template_encode: str = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>",
    prompt_template_encode_start_idx: int = 64,
    device: Optional[torch.device] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if isinstance(image, list):
        base_img_prompt = ""
        for i, img in enumerate(image):
            base_img_prompt += img_template_encode.format(i + 1)
    elif image is not None:
        base_img_prompt = img_template_encode.format(1)
    else:
        base_img_prompt = ""

    template = prompt_template_encode

    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(base_img_prompt + e) for e in prompt]

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
    image: torch.Tensor,
    vae: AutoencoderKLQwenImage,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    latent_channels: int = 16,
    sample_mode: str = "argmax",
):
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Expected image to be a tensor, got {type(image)}.")

    # preprocessed image should be a 4D tensor: batch_size, num_channels, height, width
    if image.dim() == 4:
        image = image.unsqueeze(2)
    elif image.dim() != 5:
        raise ValueError(f"Expected image dims 4 or 5, got {image.dim()}.")

    image = image.to(device=device, dtype=dtype)

    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(vae.encode(image[i : i + 1]), generator=generator[i], sample_mode=sample_mode)
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode=sample_mode)
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


class QwenImageEditResizeDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    def __init__(self, input_name: str = "image", output_name: str = "resized_image"):
        """Create a configurable step for resizing images to the target area (1024 * 1024) while maintaining the aspect ratio.

        This block resizes an input image tensor and exposes the resized result under configurable input and output
        names. Use this when you need to wire the resize step to different image fields (e.g., "image",
        "control_image")

        Args:
            input_name (str, optional): Name of the image field to read from the
                pipeline state. Defaults to "image".
            output_name (str, optional): Name of the resized image field to write
                back to the pipeline state. Defaults to "resized_image".
        """
        if not isinstance(input_name, str) or not isinstance(output_name, str):
            raise ValueError(
                f"input_name and output_name must be strings but are {type(input_name)} and {type(output_name)}"
            )
        self._image_input_name = input_name
        self._resized_image_output_name = output_name
        super().__init__()

    @property
    def description(self) -> str:
        return f"Image Resize step that resize the {self._image_input_name} to the target area (1024 * 1024) while maintaining the aspect ratio."

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
            InputParam(
                name=self._image_input_name, required=True, type_hint=torch.Tensor, description="The image to resize"
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                name=self._resized_image_output_name, type_hint=List[PIL.Image.Image], description="The resized images"
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = getattr(block_state, self._image_input_name)

        if not is_valid_image_imagelist(images):
            raise ValueError(f"Images must be image or list of images but are {type(images)}")

        if is_valid_image(images):
            images = [images]

        image_width, image_height = images[0].size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_width / image_height)

        resized_images = [
            components.image_resize_processor.resize(image, height=calculated_height, width=calculated_width)
            for image in images
        ]

        setattr(block_state, self._resized_image_output_name, resized_images)
        self.set_block_state(state, block_state)
        return components, state


class QwenImageEditPlusResizeDynamicStep(QwenImageEditResizeDynamicStep):
    model_name = "qwenimage"

    def __init__(
        self,
        input_name: str = "image",
        output_name: str = "resized_image",
        vae_image_output_name: str = "vae_image",
    ):
        """Create a configurable step for resizing images to the target area (1024 * 1024) while maintaining the aspect ratio.

        This block resizes an input image or a list input images and exposes the resized result under configurable
        input and output names. Use this when you need to wire the resize step to different image fields (e.g.,
        "image", "control_image")

        Args:
            input_name (str, optional): Name of the image field to read from the
                pipeline state. Defaults to "image".
            output_name (str, optional): Name of the resized image field to write
                back to the pipeline state. Defaults to "resized_image".
            vae_image_output_name (str, optional): Name of the image field
                to write back to the pipeline state. This is used by the VAE encoder step later on. QwenImage Edit Plus
                processes the input image(s) differently for the VL and the VAE.
        """
        if not isinstance(input_name, str) or not isinstance(output_name, str):
            raise ValueError(
                f"input_name and output_name must be strings but are {type(input_name)} and {type(output_name)}"
            )
        self.condition_image_size = 384 * 384
        self._image_input_name = input_name
        self._resized_image_output_name = output_name
        self._vae_image_output_name = vae_image_output_name
        super().__init__()

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return super().intermediate_outputs + [
            OutputParam(
                name=self._vae_image_output_name,
                type_hint=List[PIL.Image.Image],
                description="The images to be processed which will be further used by the VAE encoder.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = getattr(block_state, self._image_input_name)

        if not is_valid_image_imagelist(images):
            raise ValueError(f"Images must be image or list of images but are {type(images)}")

        if (
            not isinstance(images, torch.Tensor)
            and isinstance(images, PIL.Image.Image)
            and not isinstance(images, list)
        ):
            images = [images]

        # TODO (sayakpaul): revisit this when the inputs are `torch.Tensor`s
        condition_images = []
        vae_images = []
        for img in images:
            image_width, image_height = img.size
            condition_width, condition_height, _ = calculate_dimensions(
                self.condition_image_size, image_width / image_height
            )
            condition_images.append(components.image_resize_processor.resize(img, condition_height, condition_width))
            vae_images.append(img)

        setattr(block_state, self._resized_image_output_name, condition_images)
        setattr(block_state, self._vae_image_output_name, vae_images)
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
                name="resized_image",
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
            image=block_state.resized_image,
            prompt_template_encode=components.config.prompt_template_encode,
            prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
            device=device,
        )

        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or " "
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = get_qwen_prompt_embeds_edit(
                components.text_encoder,
                components.processor,
                prompt=negative_prompt,
                image=block_state.resized_image,
                prompt_template_encode=components.config.prompt_template_encode,
                prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
                device=device,
            )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageEditPlusTextEncoderStep(QwenImageEditTextEncoderStep):
    model_name = "qwenimage"

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return [
            ConfigSpec(
                name="prompt_template_encode",
                default="<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            ),
            ConfigSpec(
                name="img_template_encode",
                default="Picture {}: <|vision_start|><|image_pad|><|vision_end|>",
            ),
            ConfigSpec(name="prompt_template_encode_start_idx", default=64),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        self.check_inputs(block_state.prompt, block_state.negative_prompt)

        device = components._execution_device

        block_state.prompt_embeds, block_state.prompt_embeds_mask = get_qwen_prompt_embeds_edit_plus(
            components.text_encoder,
            components.processor,
            prompt=block_state.prompt,
            image=block_state.resized_image,
            prompt_template_encode=components.config.prompt_template_encode,
            img_template_encode=components.config.img_template_encode,
            prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
            device=device,
        )

        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or " "
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = (
                get_qwen_prompt_embeds_edit_plus(
                    components.text_encoder,
                    components.processor,
                    prompt=negative_prompt,
                    image=block_state.resized_image,
                    prompt_template_encode=components.config.prompt_template_encode,
                    img_template_encode=components.config.img_template_encode,
                    prompt_template_encode_start_idx=components.config.prompt_template_encode_start_idx,
                    device=device,
                )
            )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageInpaintProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Preprocess step for inpainting task. This processes the image and mask inputs together. Images can be resized first using QwenImageEditResizeDynamicStep."

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
            InputParam("mask_image", required=True),
            InputParam("resized_image"),
            InputParam("image"),
            InputParam("height"),
            InputParam("width"),
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

    @staticmethod
    def check_inputs(height, width, vae_scale_factor):
        if height is not None and height % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Height must be divisible by {vae_scale_factor * 2} but is {height}")

        if width is not None and width % (vae_scale_factor * 2) != 0:
            raise ValueError(f"Width must be divisible by {vae_scale_factor * 2} but is {width}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        if block_state.resized_image is None and block_state.image is None:
            raise ValueError("resized_image and image cannot be None at the same time")

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

        block_state.processed_image, block_state.processed_mask_image, block_state.mask_overlay_kwargs = (
            components.image_mask_processor.preprocess(
                image=image,
                mask=block_state.mask_image,
                height=height,
                width=width,
                padding_mask_crop=block_state.padding_mask_crop,
            )
        )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageProcessImagesInputStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Preprocess step. Images can be resized first using QwenImageEditResizeDynamicStep."

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
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        if block_state.resized_image is None and block_state.image is None:
            raise ValueError("resized_image and image cannot be None at the same time")

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

        block_state.processed_image = components.image_processor.preprocess(
            image=image,
            height=height,
            width=width,
        )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageEditPlusProcessImagesInputStep(QwenImageProcessImagesInputStep):
    model_name = "qwenimage-edit-plus"
    vae_image_size = 1024 * 1024

    @property
    def description(self) -> str:
        return "Image Preprocess step for QwenImage Edit Plus. Unlike QwenImage Edit, QwenImage Edit Plus doesn't use the same resized image for further preprocessing."

    @property
    def inputs(self) -> List[InputParam]:
        return [InputParam("vae_image"), InputParam("image"), InputParam("height"), InputParam("width")]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        if block_state.vae_image is None and block_state.image is None:
            raise ValueError("`vae_image` and `image` cannot be None at the same time")

        if block_state.vae_image is None:
            image = block_state.image
            self.check_inputs(
                height=block_state.height, width=block_state.width, vae_scale_factor=components.vae_scale_factor
            )
            height = block_state.height or components.default_height
            width = block_state.width or components.default_width
            block_state.processed_image = components.image_processor.preprocess(
                image=image, height=height, width=width
            )
        else:
            width, height = block_state.vae_image[0].size
            image = block_state.vae_image

            block_state.processed_image = components.image_processor.preprocess(
                image=image, height=height, width=width
            )

        self.set_block_state(state, block_state)
        return components, state


class QwenImageVaeEncoderDynamicStep(ModularPipelineBlocks):
    model_name = "qwenimage"

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
            # Basic usage with default settings (includes image processor) QwenImageVaeEncoderDynamicStep()

            # Custom input/output names for control image QwenImageVaeEncoderDynamicStep(
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
        components = [
            ComponentSpec("vae", AutoencoderKLQwenImage),
        ]
        return components

    @property
    def inputs(self) -> List[InputParam]:
        inputs = [
            InputParam(self._image_input_name, required=True),
            InputParam("generator"),
        ]
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
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.vae.dtype

        image = getattr(block_state, self._image_input_name)

        # Encode image into latents
        image_latents = encode_vae_image(
            image=image,
            vae=components.vae,
            generator=block_state.generator,
            device=device,
            dtype=dtype,
            latent_channels=components.num_channels_latents,
        )
        setattr(block_state, self._image_latents_output_name, image_latents)

        self.set_block_state(state, block_state)

        return components, state


class QwenImageControlNetVaeEncoderStep(ModularPipelineBlocks):
    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "VAE Encoder step that converts `control_image` into latent representations control_image_latents.\n"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        components = [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec("controlnet", QwenImageControlNetModel),
            ComponentSpec(
                "control_image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]
        return components

    @property
    def inputs(self) -> List[InputParam]:
        inputs = [
            InputParam("control_image", required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("generator"),
        ]
        return inputs

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "control_image_latents",
                type_hint=torch.Tensor,
                description="The latents representing the control image",
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

        self.check_inputs(block_state.height, block_state.width, components.vae_scale_factor)

        device = components._execution_device
        dtype = components.vae.dtype

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        controlnet = unwrap_module(components.controlnet)
        if isinstance(controlnet, QwenImageMultiControlNetModel) and not isinstance(block_state.control_image, list):
            block_state.control_image = [block_state.control_image]

        if isinstance(controlnet, QwenImageMultiControlNetModel):
            block_state.control_image_latents = []
            for control_image_ in block_state.control_image:
                control_image_ = components.control_image_processor.preprocess(
                    image=control_image_,
                    height=height,
                    width=width,
                )

                control_image_latents_ = encode_vae_image(
                    image=control_image_,
                    vae=components.vae,
                    generator=block_state.generator,
                    device=device,
                    dtype=dtype,
                    latent_channels=components.num_channels_latents,
                    sample_mode="sample",
                )
                block_state.control_image_latents.append(control_image_latents_)

        elif isinstance(controlnet, QwenImageControlNetModel):
            control_image = components.control_image_processor.preprocess(
                image=block_state.control_image,
                height=height,
                width=width,
            )
            block_state.control_image_latents = encode_vae_image(
                image=control_image,
                vae=components.vae,
                generator=block_state.generator,
                device=device,
                dtype=dtype,
                latent_channels=components.num_channels_latents,
                sample_mode="sample",
            )

        else:
            raise ValueError(
                f"Expected controlnet to be a QwenImageControlNetModel or QwenImageMultiControlNetModel, got {type(controlnet)}"
            )

        self.set_block_state(state, block_state)

        return components, state
