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

import math

import numpy as np
import PIL
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import VaeImageProcessor, is_valid_image, is_valid_image_imagelist
from ...models import AutoencoderKLQwenImage21
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import QwenImage21ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

QWENIMAGE21_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
QWENIMAGE21_IMAGE_TEMPLATE = "<image{}><|vision_start|><|image_pad|><|vision_end|>"


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.calculate_dimensions
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


def get_qwenimage21_prompt_embeds(
    text_encoder: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    prompt: str | list[str],
    image: list[PIL.Image.Image] | None = None,
    device: torch.device | None = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # Qwen has no bos token, so an empty string leaves the encoder with nothing to read.
    prompt = [" " if not p else p for p in prompt]

    # The prompt is built as a raw template string and passed straight to `processor(text=..., images=...)` rather
    # than going through `apply_chat_template`: the two tokenize differently and the checkpoint expects this one. The
    # number of leading system-role tokens to drop is derived from the tokenized system message so it tracks the
    # processor's template.
    sys_message = [{"role": "system", "content": [{"type": "text", "text": QWENIMAGE21_SYSTEM_PROMPT}]}]
    drop_idx = len(processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
    img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]

    image_prompt = ""
    condition_images = None
    if image is not None:
        image_prompt = " ".join(QWENIMAGE21_IMAGE_TEMPLATE.format(i + 1) for i in range(len(image)))
        # Each prompt's template repeats the `<|image_pad|>` placeholders, so hand the processor one set of images
        # per prompt, in the order the placeholders appear.
        condition_images = []
        for _ in prompt:
            for img in image:
                if img.mode == "RGBA":
                    # The checkpoint was trained with the alpha composited over white for the vision encoder. Only
                    # this copy is flattened; the VAE still reads all four channels.
                    white = PIL.Image.new("RGB", img.size, (255, 255, 255))
                    white.paste(img, mask=img.getchannel("A"))
                    img = white
                condition_images.append(img)
    template = (
        f"<|im_start|>system\n{QWENIMAGE21_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{image_prompt}{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    txt = [template.format(p) for p in prompt]

    # Left padding, as the checkpoint was trained with. The padding is dropped either way, but the side decides the
    # positions the encoder sees for a batch of prompts of different lengths.
    processor_kwargs = {"text": txt, "padding": True, "padding_side": "left", "return_tensors": "pt"}
    if condition_images is not None:
        processor_kwargs["images"] = condition_images
    model_inputs = processor(**processor_kwargs).to(device)

    forward_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "output_hidden_states": True,
    }
    if condition_images is not None and hasattr(model_inputs, "pixel_values"):
        forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
    if hasattr(model_inputs, "mm_token_type_ids"):
        forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

    # `hidden_states[-1]` has to be the last decoder layer's output, before the text encoder's final RMSNorm: that is
    # what the transformer was trained on. From transformers 5.0 that entry comes back normalized, so a forward hook
    # returning the module's input neutralizes the norm for this call on either version.
    text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_encoder(**forward_kwargs)
    finally:
        handle.remove()
    hidden_states = outputs.hidden_states[-1]

    attention_mask = model_inputs.attention_mask
    split_hidden_states = torch.split(hidden_states[attention_mask.bool()], attention_mask.sum(dim=1).tolist(), dim=0)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

    image_pad_mask = [
        (sample_ids[sample_mask.bool()] == img_token_id)[drop_idx:]
        for sample_ids, sample_mask in zip(model_inputs.input_ids, attention_mask)
    ]

    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])

    return prompt_embeds, encoder_attention_mask, image_pad_mask


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


# Copied from diffusers.modular_pipelines.qwenimage.encoders.encode_vae_image with AutoencoderKLQwenImage->AutoencoderKLQwenImage21
def encode_vae_image(
    image: torch.Tensor,
    vae: AutoencoderKLQwenImage21,
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


# auto_docstring
class QwenImage21ResizeStep(ModularPipelineBlocks):
    """
    Resize condition images for Qwen-Image 2.1. Each image is converted to RGBA and resized to the `output_resolution`
    target area while keeping its aspect ratio. The same resized images feed both the vision-language text encoder and
    the VAE encoder.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.

      Outputs:
          resized_image (`list`):
              RGBA condition images resized to the `output_resolution` target area
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Resize condition images for Qwen-Image 2.1. Each image is converted to RGBA and resized to the "
            "`output_resolution` target area while keeping its aspect ratio. The same resized images feed both the "
            "vision-language text encoder and the VAE encoder."
        )

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
            InputParam(
                name="output_resolution",
                type_hint=int,
                default=1024,
                description="Target side length used to derive the output size and to resize condition images.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="resized_image",
                type_hint=list[PIL.Image.Image],
                description="RGBA condition images resized to the `output_resolution` target area",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        images = block_state.image
        if not is_valid_image_imagelist(images):
            raise ValueError(f"Images must be image or list of images but are {type(images)}")
        if is_valid_image(images):
            images = [images]

        # The text encoder reads each condition image as vision context, so the pixels have to be there. Normalize
        # to PIL up front, and everything downstream sees one type.
        resized_images = []
        for image in images:
            if isinstance(image, np.ndarray):
                image = PIL.Image.fromarray(image)
            elif not isinstance(image, PIL.Image.Image):
                raise ValueError(
                    f"`image` accepts a PIL image or a numpy array, or a list of either, but got "
                    f"{type(image).__name__}. Latents cannot stand in for a condition image here, because the text "
                    f"encoder has to see the image itself."
                )
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            image_width, image_height = image.size
            width, height, _ = calculate_dimensions(
                block_state.output_resolution * block_state.output_resolution, image_width / image_height
            )
            resized_images.append(components.image_processor.resize(image, height=height, width=width))

        block_state.resized_image = resized_images
        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21TextEncoderStep(ModularPipelineBlocks):
    """
    Text Encoder step that generates text embeddings with Qwen3-VL to guide text-to-image generation.

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) processor (`Qwen3VLProcessor`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings
          negative_image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the negative prompt embeddings
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings with Qwen3-VL to guide text-to-image generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("processor", Qwen3VLProcessor),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
            OutputParam(
                name="image_pad_mask",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the prompt embeddings",
            ),
            OutputParam(
                name="negative_image_pad_mask",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the negative prompt embeddings",
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
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt, block_state.negative_prompt)

        device = components._execution_device

        block_state.prompt_embeds, block_state.prompt_embeds_mask, block_state.image_pad_mask = (
            get_qwenimage21_prompt_embeds(
                components.text_encoder,
                components.processor,
                prompt=block_state.prompt,
                device=device,
            )
        )

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        block_state.negative_image_pad_mask = None
        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or ""
            (
                block_state.negative_prompt_embeds,
                block_state.negative_prompt_embeds_mask,
                block_state.negative_image_pad_mask,
            ) = get_qwenimage21_prompt_embeds(
                components.text_encoder,
                components.processor,
                prompt=negative_prompt,
                device=device,
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21VLTextEncoderStep(ModularPipelineBlocks):
    """
    Text Encoder step that encodes the prompt together with the condition images with Qwen3-VL to guide
    image-conditioned generation.

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) processor (`Qwen3VLProcessor`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          resized_image (`list`):
              RGBA condition images resized to the output resolution. Can be generated in the resize step.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings
          negative_image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the negative prompt embeddings
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Text Encoder step that encodes the prompt together with the condition images with Qwen3-VL to guide "
            "image-conditioned generation."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("processor", Qwen3VLProcessor),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
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
                type_hint=list[PIL.Image.Image],
                description="RGBA condition images resized to the output resolution. Can be generated in the resize step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("negative_prompt_embeds_mask"),
            OutputParam(
                name="image_pad_mask",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the prompt embeddings",
            ),
            OutputParam(
                name="negative_image_pad_mask",
                type_hint=torch.Tensor,
                description="Bool mask marking the vision positions of the negative prompt embeddings",
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
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt, block_state.negative_prompt)

        device = components._execution_device

        block_state.prompt_embeds, block_state.prompt_embeds_mask, block_state.image_pad_mask = (
            get_qwenimage21_prompt_embeds(
                components.text_encoder,
                components.processor,
                prompt=block_state.prompt,
                image=block_state.resized_image,
                device=device,
            )
        )

        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        block_state.negative_image_pad_mask = None
        if components.requires_unconditional_embeds:
            negative_prompt = block_state.negative_prompt or ""
            (
                block_state.negative_prompt_embeds,
                block_state.negative_prompt_embeds_mask,
                block_state.negative_image_pad_mask,
            ) = get_qwenimage21_prompt_embeds(
                components.text_encoder,
                components.processor,
                prompt=negative_prompt,
                image=block_state.resized_image,
                device=device,
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21ProcessImagesInputStep(ModularPipelineBlocks):
    """
    Image preprocess step that turns the resized RGBA condition images into normalized tensors for the VAE.

      Components:
          image_processor (`VaeImageProcessor`)

      Inputs:
          resized_image (`list`):
              RGBA condition images resized to the output resolution. Can be generated in the resize step.

      Outputs:
          processed_image (`list`):
              Normalized RGBA image tensors, one per condition image
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "Image preprocess step that turns the resized RGBA condition images into normalized tensors for the VAE."
        )

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
                description="RGBA condition images resized to the output resolution. Can be generated in the resize step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="processed_image",
                type_hint=list[torch.Tensor],
                description="Normalized RGBA image tensors, one per condition image",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        block_state.processed_image = [
            components.image_processor.preprocess(image, height=image.height, width=image.width)
            for image in block_state.resized_image
        ]

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class QwenImage21VaeEncoderStep(ModularPipelineBlocks):
    """
    VAE Encoder step that encodes each processed condition image into normalized latents. Images can have different
    resolutions, so the latents are returned as a list.

      Components:
          vae (`AutoencoderKLQwenImage21`)

      Inputs:
          processed_image (`list`):
              Normalized RGBA image tensors to encode. Can be generated in the preprocess step.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          image_latents (`list`):
              Normalized latents of each condition image, each of shape (1, C, 1, H, W)
    """

    model_name = "qwenimage21"

    @property
    def description(self) -> str:
        return (
            "VAE Encoder step that encodes each processed condition image into normalized latents. Images can have "
            "different resolutions, so the latents are returned as a list."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLQwenImage21)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="processed_image",
                required=True,
                type_hint=list[torch.Tensor],
                description="Normalized RGBA image tensors to encode. Can be generated in the preprocess step.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                name="image_latents",
                type_hint=list[torch.Tensor],
                description="Normalized latents of each condition image, each of shape (1, C, 1, H, W)",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: QwenImage21ModularPipeline, state: PipelineState
    ) -> tuple[QwenImage21ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        device = components._execution_device
        dtype = components.vae.dtype

        block_state.image_latents = [
            encode_vae_image(
                image=image,
                vae=components.vae,
                generator=block_state.generator,
                device=device,
                dtype=dtype,
                latent_channels=components.vae.config.z_dim,
            )
            for image in block_state.processed_image
        ]

        self.set_block_state(state, block_state)
        return components, state
