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

"""
Text and VAE encoder blocks for QwenImage pipelines.
"""

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
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImageModularPipeline
from .prompt_templates import (
    QWENIMAGE_EDIT_PLUS_IMG_TEMPLATE,
    QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE,
    QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE_START_IDX,
    QWENIMAGE_EDIT_PROMPT_TEMPLATE,
    QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX,
    QWENIMAGE_LAYERED_CAPTION_PROMPT_CN,
    QWENIMAGE_LAYERED_CAPTION_PROMPT_EN,
    QWENIMAGE_PROMPT_TEMPLATE,
    QWENIMAGE_PROMPT_TEMPLATE_START_IDX,
)


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
    prompt: str | list[str] = None,
    prompt_template_encode: str = QWENIMAGE_PROMPT_TEMPLATE,
    prompt_template_encode_start_idx: int = QWENIMAGE_PROMPT_TEMPLATE_START_IDX,
    tokenizer_max_length: int = 1024,
    device: torch.device | None = None,
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
    prompt: str | list[str] = None,
    image: torch.Tensor | None = None,
    prompt_template_encode: str = QWENIMAGE_EDIT_PROMPT_TEMPLATE,
    prompt_template_encode_start_idx: int = QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX,
    device: torch.device | None = None,
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
    prompt: str | list[str] = None,
    image: torch.Tensor | list[PIL.Image.Image, PIL.Image.Image] | None = None,
    prompt_template_encode: str = QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE,
    img_template_encode: str = QWENIMAGE_EDIT_PLUS_IMG_TEMPLATE,
    prompt_template_encode_start_idx: int = QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE_START_IDX,
    device: torch.device | None = None,
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


# ====================
# 1. RESIZE
# ====================
# In QwenImage pipelines, resize is a separate step because the resized image is used in VL encoding and vae encoder blocks:
#
#   image (PIL.Image.Image)
#       │
#       ▼
#   resized_image ([PIL.Image.Image])
#       │
#       ├──► text_encoder ──► prompt_embeds, prompt_embeds_mask
#       │    (VL encoding needs the resized image for vision-language fusion)
#       │
#       └──► image_processor ──► processed_image (torch.Tensor, pixel space)
#                │
#                ▼
#            vae_encoder ──► image_latents (torch.Tensor, latent space)
#
# In most of our other pipelines, resizing is done as part of the image preprocessing step.
# ====================


# auto_docstring
class QwenImageEditResizeStep(ModularPipelineBlocks):
    """
    Image Resize step that resize the image to target area while maintaining the aspect ratio.

      Components:
          image_resize_processor (`VaeImageProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.

      Outputs:
          resized_image (`list`):
              The resized images
    """

    model_name = "qwenimage-edit"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to target area while maintaining the aspect ratio."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_resize_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam.template("image")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="resized_image",
                type_hint=list[PIL.Image.Image],
                description="The resized images",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = block_state.image

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

        block_state.resized_image = resized_images
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageLayeredResizeStep(ModularPipelineBlocks):
    """
    Image Resize step that resize the image to a target area (defined by the resolution parameter from user) while
    maintaining the aspect ratio.

      Components:
          image_resize_processor (`VaeImageProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          resolution (`int`, *optional*, defaults to 640):
              The target area to resize the image to, can be 1024 or 640

      Outputs:
          resized_image (`list`):
              The resized images
    """

    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to a target area (defined by the resolution parameter from user) while maintaining the aspect ratio."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_resize_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image"),
            InputParam(
                name="resolution",
                default=640,
                type_hint=int,
                description="The target area to resize the image to, can be 1024 or 640",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="resized_image",
                type_hint=list[PIL.Image.Image],
                description="The resized images",
            )
        ]

    @staticmethod
    def check_inputs(resolution: int):
        if resolution not in [1024, 640]:
            raise ValueError(f"Resolution must be 1024 or 640 but is {resolution}")

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        self.check_inputs(resolution=block_state.resolution)

        images = block_state.image

        if not is_valid_image_imagelist(images):
            raise ValueError(f"Images must be image or list of images but are {type(images)}")

        if is_valid_image(images):
            images = [images]

        image_width, image_height = images[0].size
        target_area = block_state.resolution * block_state.resolution
        calculated_width, calculated_height, _ = calculate_dimensions(target_area, image_width / image_height)

        resized_images = [
            components.image_resize_processor.resize(image, height=calculated_height, width=calculated_width)
            for image in images
        ]

        block_state.resized_image = resized_images
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageEditPlusResizeStep(ModularPipelineBlocks):
    """
    Resize images for QwenImage Edit Plus pipeline.
      Produces two outputs: resized_image (1024x1024) for VAE encoding, resized_cond_image (384x384) for VL text
      encoding. Each image is resized independently based on its own aspect ratio.

      Components:
          image_resize_processor (`VaeImageProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.

      Outputs:
          resized_image (`list`):
              Images resized to 1024x1024 target area for VAE encoding
          resized_cond_image (`list`):
              Images resized to 384x384 target area for VL text encoding
    """

    model_name = "qwenimage-edit-plus"

    @property
    def description(self) -> str:
        return (
            "Resize images for QwenImage Edit Plus pipeline.\n"
            "Produces two outputs: resized_image (1024x1024) for VAE encoding, "
            "resized_cond_image (384x384) for VL text encoding.\n"
            "Each image is resized independently based on its own aspect ratio."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_resize_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        # image
        return [InputParam.template("image")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="resized_image",
                type_hint=list[PIL.Image.Image],
                description="Images resized to 1024x1024 target area for VAE encoding",
            ),
            OutputParam(
                name="resized_cond_image",
                type_hint=list[PIL.Image.Image],
                description="Images resized to 384x384 target area for VL text encoding",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        images = block_state.image

        if not is_valid_image_imagelist(images):
            raise ValueError(f"Images must be image or list of images but are {type(images)}")

        if is_valid_image(images):
            images = [images]

        # Resize each image independently based on its own aspect ratio
        resized_images = []
        resized_cond_images = []
        for image in images:
            image_width, image_height = image.size

            # For VAE encoder (1024x1024 target area)
            vae_width, vae_height, _ = calculate_dimensions(1024 * 1024, image_width / image_height)
            resized_images.append(components.image_resize_processor.resize(image, height=vae_height, width=vae_width))

            # For VL text encoder (384x384 target area)
            vl_width, vl_height, _ = calculate_dimensions(384 * 384, image_width / image_height)
            resized_cond_images.append(
                components.image_resize_processor.resize(image, height=vl_height, width=vl_width)
            )

        block_state.resized_image = resized_images
        block_state.resized_cond_image = resized_cond_images
        self.set_block_state(state, block_state)
        return components, state


# ====================
# 2. GET IMAGE PROMPT
# ====================


# auto_docstring
class QwenImageLayeredGetImagePromptStep(ModularPipelineBlocks):
    """
    Auto-caption step that generates a text prompt from the input image if none is provided.
      Uses the VL model (text_encoder) to generate a description of the image. If prompt is already provided, this step
      passes through unchanged.

      Components:
          text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor (`Qwen2VLProcessor`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          resized_image (`Image`):
              The image to generate caption from, should be resized use the resize step
          use_en_prompt (`bool`, *optional*, defaults to False):
              Whether to use English prompt template

      Outputs:
          prompt (`str`):
              The prompt or prompts to guide image generation. If not provided, updated using image caption
    """

    model_name = "qwenimage-layered"

    def __init__(self):
        self.image_caption_prompt_en = QWENIMAGE_LAYERED_CAPTION_PROMPT_EN
        self.image_caption_prompt_cn = QWENIMAGE_LAYERED_CAPTION_PROMPT_CN
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Auto-caption step that generates a text prompt from the input image if none is provided.\n"
            "Uses the VL model (text_encoder) to generate a description of the image.\n"
            "If prompt is already provided, this step passes through unchanged."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen2_5_VLForConditionalGeneration),
            ComponentSpec("processor", Qwen2VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt", required=False
            ),  # it is not required for qwenimage-layered, unlike other pipelines
            InputParam(
                name="resized_image",
                required=True,
                type_hint=PIL.Image.Image,
                description="The image to generate caption from, should be resized use the resize step",
            ),
            InputParam(
                name="use_en_prompt",
                default=False,
                type_hint=bool,
                description="Whether to use English prompt template",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="prompt",
                type_hint=str,
                description="The prompt or prompts to guide image generation. If not provided, updated using image caption",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        # If prompt is empty or None, generate caption from image
        if block_state.prompt is None or block_state.prompt == "" or block_state.prompt == " ":
            if block_state.use_en_prompt:
                caption_prompt = self.image_caption_prompt_en
            else:
                caption_prompt = self.image_caption_prompt_cn

            model_inputs = components.processor(
                text=caption_prompt,
                images=block_state.resized_image,
                padding=True,
                return_tensors="pt",
            ).to(device)

            generated_ids = components.text_encoder.generate(**model_inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = components.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            block_state.prompt = output_text.strip()

        self.set_block_state(state, block_state)
        return components, state


# ====================
# 3. TEXT ENCODER
# ====================


# auto_docstring
class QwenImageTextEncoderStep(ModularPipelineBlocks):
    """
    Text Encoder step that generates text embeddings to guide the image generation.

      Components:
          text_encoder (`Qwen2_5_VLForConditionalGeneration`): The text encoder to use tokenizer (`Qwen2Tokenizer`):
          The tokenizer to use guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "qwenimage"

    def __init__(self):
        self.prompt_template_encode = QWENIMAGE_PROMPT_TEMPLATE
        self.prompt_template_encode_start_idx = QWENIMAGE_PROMPT_TEMPLATE_START_IDX
        self.tokenizer_max_length = 1024
        super().__init__()

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings to guide the image generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam.template("max_sequence_length", default=1024),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
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
            prompt_template_encode=self.prompt_template_encode,
            prompt_template_encode_start_idx=self.prompt_template_encode_start_idx,
            tokenizer_max_length=self.tokenizer_max_length,
            device=device,
        )

        block_state.prompt_embeds = block_state.prompt_embeds[:, : block_state.max_sequence_length]
        block_state.prompt_embeds_mask = block_state.prompt_embeds_mask[:, : block_state.max_sequence_length]

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or ""
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = get_qwen_prompt_embeds(
                components.text_encoder,
                components.tokenizer,
                prompt=negative_prompt,
                prompt_template_encode=self.prompt_template_encode,
                prompt_template_encode_start_idx=self.prompt_template_encode_start_idx,
                tokenizer_max_length=self.tokenizer_max_length,
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


# auto_docstring
class QwenImageEditTextEncoderStep(ModularPipelineBlocks):
    """
    Text Encoder step that processes both prompt and image together to generate text embeddings for guiding image
    generation.

      Components:
          text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor (`Qwen2VLProcessor`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          resized_image (`Image`):
              The image prompt to encode, should be resized using resize step

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "qwenimage"

    def __init__(self):
        self.prompt_template_encode = QWENIMAGE_EDIT_PROMPT_TEMPLATE
        self.prompt_template_encode_start_idx = QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX
        super().__init__()

    @property
    def description(self) -> str:
        return "Text Encoder step that processes both prompt and image together to generate text embeddings for guiding image generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam(
                name="resized_image",
                required=True,
                type_hint=PIL.Image.Image,
                description="The image prompt to encode, should be resized using resize step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
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
            prompt_template_encode=self.prompt_template_encode,
            prompt_template_encode_start_idx=self.prompt_template_encode_start_idx,
            device=device,
        )

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or " "
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = get_qwen_prompt_embeds_edit(
                components.text_encoder,
                components.processor,
                prompt=negative_prompt,
                image=block_state.resized_image,
                prompt_template_encode=self.prompt_template_encode,
                prompt_template_encode_start_idx=self.prompt_template_encode_start_idx,
                device=device,
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageEditPlusTextEncoderStep(ModularPipelineBlocks):
    """
    Text Encoder step for QwenImage Edit Plus that processes prompt and multiple images together to generate text
    embeddings for guiding image generation.

      Components:
          text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor (`Qwen2VLProcessor`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          resized_cond_image (`Tensor`):
              The image(s) to encode, can be a single image or list of images, should be resized to 384x384 using
              resize step

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "qwenimage-edit-plus"

    def __init__(self):
        self.prompt_template_encode = QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE
        self.img_template_encode = QWENIMAGE_EDIT_PLUS_IMG_TEMPLATE
        self.prompt_template_encode_start_idx = QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE_START_IDX
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Text Encoder step for QwenImage Edit Plus that processes prompt and multiple images together "
            "to generate text embeddings for guiding image generation."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam(
                name="resized_cond_image",
                required=True,
                type_hint=torch.Tensor,
                description="The image(s) to encode, can be a single image or list of images, should be resized to 384x384 using resize step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
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

        block_state.prompt_embeds, block_state.prompt_embeds_mask = get_qwen_prompt_embeds_edit_plus(
            components.text_encoder,
            components.processor,
            prompt=block_state.prompt,
            image=block_state.resized_cond_image,
            prompt_template_encode=self.prompt_template_encode,
            img_template_encode=self.img_template_encode,
            prompt_template_encode_start_idx=self.prompt_template_encode_start_idx,
            device=device,
        )

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or " "
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = (
                get_qwen_prompt_embeds_edit_plus(
                    components.text_encoder,
                    components.processor,
                    prompt=negative_prompt,
                    image=block_state.resized_cond_image,
                    prompt_template_encode=self.prompt_template_encode,
                    img_template_encode=self.img_template_encode,
                    prompt_template_encode_start_idx=self.prompt_template_encode_start_idx,
                    device=device,
                )
            )

        self.set_block_state(state, block_state)
        return components, state


# ====================
# 4. IMAGE PREPROCESS
# ====================


# auto_docstring
class QwenImageInpaintProcessImagesInputStep(ModularPipelineBlocks):
    """
    Image Preprocess step for inpainting task. This processes the image and mask inputs together. Images will be
    resized to the given height and width.

      Components:
          image_mask_processor (`InpaintProcessor`)

      Inputs:
          mask_image (`Image`):
              Mask image for inpainting.
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.

      Outputs:
          processed_image (`Tensor`):
              The processed image
          processed_mask_image (`Tensor`):
              The processed mask image
          mask_overlay_kwargs (`dict`):
              The kwargs for the postprocess step to apply the mask overlay
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Preprocess step for inpainting task. This processes the image and mask inputs together. Images will be resized to the given height and width."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("mask_image"),
            InputParam.template("image"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("padding_mask_crop"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="processed_image",
                type_hint=torch.Tensor,
                description="The processed image",
            ),
            OutputParam(
                name="processed_mask_image",
                type_hint=torch.Tensor,
                description="The processed mask image",
            ),
            OutputParam(
                name="mask_overlay_kwargs",
                type_hint=dict,
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

        self.check_inputs(
            height=block_state.height, width=block_state.width, vae_scale_factor=components.vae_scale_factor
        )
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

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


# auto_docstring
class QwenImageEditInpaintProcessImagesInputStep(ModularPipelineBlocks):
    """
    Image Preprocess step for inpainting task. This processes the image and mask inputs together. Images should be
    resized first.

      Components:
          image_mask_processor (`InpaintProcessor`)

      Inputs:
          mask_image (`Image`):
              Mask image for inpainting.
          resized_image (`Image`):
              The resized image. should be generated using a resize step
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.

      Outputs:
          processed_image (`Tensor`):
              The processed image
          processed_mask_image (`Tensor`):
              The processed mask image
          mask_overlay_kwargs (`dict`):
              The kwargs for the postprocess step to apply the mask overlay
    """

    model_name = "qwenimage-edit"

    @property
    def description(self) -> str:
        return "Image Preprocess step for inpainting task. This processes the image and mask inputs together. Images should be resized first."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("mask_image"),
            InputParam(
                name="resized_image",
                required=True,
                type_hint=PIL.Image.Image,
                description="The resized image. should be generated using a resize step",
            ),
            InputParam.template("padding_mask_crop"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(name="processed_image", type_hint=torch.Tensor, description="The processed image"),
            OutputParam(
                name="processed_mask_image",
                type_hint=torch.Tensor,
                description="The processed mask image",
            ),
            OutputParam(
                name="mask_overlay_kwargs",
                type_hint=dict,
                description="The kwargs for the postprocess step to apply the mask overlay",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        width, height = block_state.resized_image[0].size

        block_state.processed_image, block_state.processed_mask_image, block_state.mask_overlay_kwargs = (
            components.image_mask_processor.preprocess(
                image=block_state.resized_image,
                mask=block_state.mask_image,
                height=height,
                width=width,
                padding_mask_crop=block_state.padding_mask_crop,
            )
        )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageProcessImagesInputStep(ModularPipelineBlocks):
    """
    Image Preprocess step. will resize the image to the given height and width.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.

      Outputs:
          processed_image (`Tensor`):
              The processed image
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "Image Preprocess step. will resize the image to the given height and width."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image"),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="processed_image",
                type_hint=torch.Tensor,
                description="The processed image",
            )
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

        self.check_inputs(
            height=block_state.height, width=block_state.width, vae_scale_factor=components.vae_scale_factor
        )
        height = block_state.height or components.default_height
        width = block_state.width or components.default_width

        block_state.processed_image = components.image_processor.preprocess(
            image=block_state.image,
            height=height,
            width=width,
        )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageEditProcessImagesInputStep(ModularPipelineBlocks):
    """
    Image Preprocess step. Images needs to be resized first.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          resized_image (`list`):
              The resized image. should be generated using a resize step

      Outputs:
          processed_image (`Tensor`):
              The processed image
    """

    model_name = "qwenimage-edit"

    @property
    def description(self) -> str:
        return "Image Preprocess step. Images needs to be resized first."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="resized_image",
                required=True,
                type_hint=list[PIL.Image.Image],
                description="The resized image. should be generated using a resize step",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="processed_image",
                type_hint=torch.Tensor,
                description="The processed image",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        width, height = block_state.resized_image[0].size

        block_state.processed_image = components.image_processor.preprocess(
            image=block_state.resized_image,
            height=height,
            width=width,
        )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImageEditPlusProcessImagesInputStep(ModularPipelineBlocks):
    """
    Image Preprocess step. Images can be resized first. If a list of images is provided, will return a list of
    processed images.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          resized_image (`list`):
              The resized image. should be generated using a resize step

      Outputs:
          processed_image (`Tensor`):
              The processed image
    """

    model_name = "qwenimage-edit-plus"

    @property
    def description(self) -> str:
        return "Image Preprocess step. Images can be resized first. If a list of images is provided, will return a list of processed images."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="resized_image",
                required=True,
                type_hint=list[PIL.Image.Image],
                description="The resized image. should be generated using a resize step",
            )
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="processed_image",
                type_hint=torch.Tensor,
                description="The processed image",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)

        image = block_state.resized_image

        is_image_list = isinstance(image, list)
        if not is_image_list:
            image = [image]

        processed_images = []
        for img in image:
            img_width, img_height = img.size
            processed_images.append(
                components.image_processor.preprocess(image=img, height=img_height, width=img_width)
            )

        if is_image_list:
            block_state.processed_image = processed_images
        else:
            block_state.processed_image = processed_images[0]

        self.set_block_state(state, block_state)
        return components, state


# ====================
# 5. VAE ENCODER
# ====================


# auto_docstring
class QwenImageVaeEncoderStep(ModularPipelineBlocks):
    """
    VAE Encoder step that converts processed_image into latent representations image_latents.
      Handles both single images and lists of images with varied resolutions.

      Components:
          vae (`AutoencoderKLQwenImage`)

      Inputs:
          processed_image (`Tensor`):
              The image tensor to encode
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "qwenimage"

    def __init__(self, input: InputParam | None = None, output: OutputParam | None = None):
        """Initialize a VAE encoder step for converting images to latent representations.

        Handles both single images and lists of images. When input is a list, outputs a list of latents. When input is
        a single tensor, outputs a single latent tensor.

        Args:
            input (InputParam, optional): Input parameter for the processed image. Defaults to "processed_image".
            output (OutputParam, optional): Output parameter for the image latents. Defaults to "image_latents".
        """
        if input is None:
            input = InputParam(
                name="processed_image", required=True, type_hint=torch.Tensor, description="The image tensor to encode"
            )

        if output is None:
            output = OutputParam.template("image_latents")

        if not isinstance(input, InputParam):
            raise ValueError(f"input must be InputParam but is {type(input)}")
        if not isinstance(output, OutputParam):
            raise ValueError(f"output must be OutputParam but is {type(output)}")

        self._input = input
        self._output = output
        self._image_input_name = input.name
        self._image_latents_output_name = output.name
        super().__init__()

    @property
    def description(self) -> str:
        return (
            f"VAE Encoder step that converts {self._image_input_name} into latent representations {self._image_latents_output_name}.\n"
            "Handles both single images and lists of images with varied resolutions."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLQwenImage)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            self._input,  # default is "processed_image"
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [self._output]  # default is "image_latents"

    @torch.no_grad()
    def __call__(self, components: QwenImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.vae.dtype

        image = getattr(block_state, self._image_input_name)
        is_image_list = isinstance(image, list)
        if not is_image_list:
            image = [image]

        # Handle both single image and list of images
        image_latents = []
        for img in image:
            image_latents.append(
                encode_vae_image(
                    image=img,
                    vae=components.vae,
                    generator=block_state.generator,
                    device=device,
                    dtype=dtype,
                    latent_channels=components.num_channels_latents,
                )
            )
        if not is_image_list:
            image_latents = image_latents[0]

        setattr(block_state, self._image_latents_output_name, image_latents)

        self.set_block_state(state, block_state)

        return components, state


# auto_docstring
class QwenImageControlNetVaeEncoderStep(ModularPipelineBlocks):
    """
    VAE Encoder step that converts `control_image` into latent representations control_image_latents.

      Components:
          vae (`AutoencoderKLQwenImage`) controlnet (`QwenImageControlNetModel`) control_image_processor
          (`VaeImageProcessor`)

      Inputs:
          control_image (`Image`):
              Control image for ControlNet conditioning.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          control_image_latents (`Tensor`):
              The latents representing the control image
    """

    model_name = "qwenimage"

    @property
    def description(self) -> str:
        return "VAE Encoder step that converts `control_image` into latent representations control_image_latents.\n"

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        inputs = [
            InputParam.template("control_image"),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("generator"),
        ]
        return inputs

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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


# ====================
# 6. PERMUTE LATENTS
# ====================


# auto_docstring
class QwenImageLayeredPermuteLatentsStep(ModularPipelineBlocks):
    """
    Permute image latents from (B, C, 1, H, W) to (B, 1, C, H, W) for Layered packing.

      Inputs:
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.

      Outputs:
          image_latents (`Tensor`):
              The latent representation of the input image. (permuted from [B, C, 1, H, W] to [B, 1, C, H, W])
    """

    model_name = "qwenimage-layered"

    @property
    def description(self) -> str:
        return "Permute image latents from (B, C, 1, H, W) to (B, 1, C, H, W) for Layered packing."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image_latents"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("image_latents", note="permuted from [B, C, 1, H, W] to [B, 1, C, H, W]"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Permute: (B, C, 1, H, W) -> (B, 1, C, H, W)
        latents = block_state.image_latents
        block_state.image_latents = latents.permute(0, 2, 1, 3, 4)

        self.set_block_state(state, block_state)
        return components, state
