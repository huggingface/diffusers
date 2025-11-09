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

import numpy as np
import PIL
import regex as re
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLWan
from ...utils import is_ftfy_available, is_torchvision_available, logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import WanModularPipeline


if is_ftfy_available():
    import ftfy

if is_torchvision_available():
    from torchvision import transforms


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


def get_t5_prompt_embeds(
    text_encoder: UMT5EncoderModel,
    tokenizer: AutoTokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int,
    device: torch.device,
):
    dtype = text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    return prompt_embeds


def encode_image(
    image: PipelineImageInput,
    image_processor: CLIPImageProcessor,
    image_encoder: CLIPVisionModel,
    device: Optional[torch.device] = None,
):
    image = image_processor(images=image, return_tensors="pt").to(device)
    image_embeds = image_encoder(**image, output_hidden_states=True)
    return image_embeds.hidden_states[-2]


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
    video_tensor: torch.Tensor,
    vae: AutoencoderKLWan,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    latent_channels: int = 16,
):
    if not isinstance(video_tensor, torch.Tensor):
        raise ValueError(f"Expected video_tensor to be a tensor, got {type(video_tensor)}.")

    if isinstance(generator, list) and len(generator) != video_tensor.shape[0]:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but it is not same as number of images {video_tensor.shape[0]}."
        )

    video_tensor = video_tensor.to(device=device, dtype=dtype)

    if isinstance(generator, list):
        video_latents = [
            retrieve_latents(vae.encode(video_tensor[i : i + 1]), generator=generator[i], sample_mode="argmax")
            for i in range(video_tensor.shape[0])
        ]
        video_latents = torch.cat(video_latents, dim=0)
    else:
        video_latents = retrieve_latents(vae.encode(video_tensor), sample_mode="argmax")

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, latent_channels, 1, 1, 1)
        .to(video_latents.device, video_latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, latent_channels, 1, 1, 1).to(
        video_latents.device, video_latents.dtype
    )
    video_latents = (video_latents - latents_mean) * latents_std

    return video_latents


class WanTextEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Text Encoder step that generate text_embeddings to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", UMT5EncoderModel),
            ComponentSpec("tokenizer", AutoTokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0}),
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

        prompt_embeds = get_t5_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

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

            negative_prompt_embeds = get_t5_prompt_embeds(
                text_encoder=components.text_encoder,
                tokenizer=components.tokenizer,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
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


class WanImageResizeStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the image to the target area (height * width) while maintaining the aspect ratio."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("image", type_hint=PIL.Image.Image, required=True),
            InputParam("height", type_hint=int, default=480),
            InputParam("width", type_hint=int, default=832),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("resized_image", type_hint=PIL.Image.Image),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        max_area = block_state.height * block_state.width

        image = block_state.image
        aspect_ratio = image.height / image.width
        mod_value = components.vae_scale_factor_spatial * components.patch_size_spatial
        block_state.height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        block_state.width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        block_state.resized_image = image.resize((block_state.width, block_state.height))

        self.set_block_state(state, block_state)
        return components, state


class WanImageCropResizeStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Image Resize step that resize the last_image to the same size of first frame image with center crop."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "resized_image", type_hint=PIL.Image.Image, required=True, description="The resized first frame image"
            ),
            InputParam("last_image", type_hint=PIL.Image.Image, required=True, description="The last frameimage"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("resized_last_image", type_hint=PIL.Image.Image),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        height = block_state.resized_image.height
        width = block_state.resized_image.width
        image = block_state.last_image

        # Calculate resize ratio to match first frame dimensions
        resize_ratio = max(width / image.width, height / image.height)

        # Resize the image
        width = round(image.width * resize_ratio)
        height = round(image.height * resize_ratio)
        size = [width, height]
        resized_image = transforms.functional.center_crop(image, size)
        block_state.resized_last_image = resized_image

        self.set_block_state(state, block_state)
        return components, state


class WanImageEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Image Encoder step that generate image_embeds based on first frame image to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("image_processor", CLIPImageProcessor),
            ComponentSpec("image_encoder", CLIPVisionModel),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("image_embeds", type_hint=torch.Tensor, description="The image embeddings"),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        image = block_state.resized_image

        image_embeds = encode_image(
            image_processor=components.image_processor,
            image_encoder=components.image_encoder,
            image=image,
            device=device,
        )
        block_state.image_embeds = image_embeds
        self.set_block_state(state, block_state)
        return components, state


class WanFirstLastFrameImageEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Image Encoder step that generate image_embeds based on first and last frame images to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("image_processor", CLIPImageProcessor),
            ComponentSpec("image_encoder", CLIPVisionModel),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
            InputParam("resized_last_image", type_hint=PIL.Image.Image, required=True),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam("image_embeds", type_hint=torch.Tensor, description="The image embeddings"),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        first_frame_image = block_state.resized_image
        last_frame_image = block_state.resized_last_image

        image_embeds = encode_image(
            image_processor=components.image_processor,
            image_encoder=components.image_encoder,
            image=[first_frame_image, last_frame_image],
            device=device,
        )
        block_state.image_embeds = image_embeds
        self.set_block_state(state, block_state)
        return components, state


class WanVaeImageEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Vae Image Encoder step that generate condition_latents based on first frame image to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("num_frames"),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "first_frame_latents",
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
        if block_state.num_frames is not None and (
            block_state.num_frames < 1 or (block_state.num_frames - 1) % components.vae_scale_factor_temporal != 0
        ):
            raise ValueError(
                f"`num_frames` has to be greater than 0, and (num_frames - 1) must be divisible by {components.vae_scale_factor_temporal}, but got {block_state.num_frames}."
            )

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        image = block_state.resized_image

        device = components._execution_device
        dtype = torch.float32

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        num_frames = block_state.num_frames or components.default_num_frames

        image_tensor = components.video_processor.preprocess(image, height=height, width=width).to(
            device=device, dtype=dtype
        )

        if image_tensor.dim() == 4:
            image_tensor = image_tensor.unsqueeze(2)

        video_tensor = torch.cat(
            [
                image_tensor,
                image_tensor.new_zeros(image_tensor.shape[0], image_tensor.shape[1], num_frames - 1, height, width),
            ],
            dim=2,
        ).to(device=device, dtype=dtype)

        block_state.first_frame_latents = encode_vae_image(
            video_tensor=video_tensor,
            vae=components.vae,
            generator=block_state.generator,
            device=device,
            dtype=dtype,
            latent_channels=components.num_channels_latents,
        )

        self.set_block_state(state, block_state)
        return components, state


class WanFirstLastFrameVaeImageEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Vae Image Encoder step that generate condition_latents based on first and last frame images to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
            InputParam("resized_last_image", type_hint=PIL.Image.Image, required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("num_frames"),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "first_last_frame_latents",
                type_hint=torch.Tensor,
                description="video latent representation with the first and last frame images condition",
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
        if block_state.num_frames is not None and (
            block_state.num_frames < 1 or (block_state.num_frames - 1) % components.vae_scale_factor_temporal != 0
        ):
            raise ValueError(
                f"`num_frames` has to be greater than 0, and (num_frames - 1) must be divisible by {components.vae_scale_factor_temporal}, but got {block_state.num_frames}."
            )

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        first_frame_image = block_state.resized_image
        last_frame_image = block_state.resized_last_image

        device = components._execution_device
        dtype = torch.float32

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        num_frames = block_state.num_frames or components.default_num_frames

        first_image_tensor = components.video_processor.preprocess(first_frame_image, height=height, width=width).to(
            device=device, dtype=dtype
        )
        first_image_tensor = first_image_tensor.unsqueeze(2)

        last_image_tensor = components.video_processor.preprocess(last_frame_image, height=height, width=width).to(
            device=device, dtype=dtype
        )

        last_image_tensor = last_image_tensor.unsqueeze(2)

        video_tensor = torch.cat(
            [
                first_image_tensor,
                first_image_tensor.new_zeros(
                    first_image_tensor.shape[0], first_image_tensor.shape[1], num_frames - 2, height, width
                ),
                last_image_tensor,
            ],
            dim=2,
        ).to(device=device, dtype=dtype)

        block_state.first_last_frame_latents = encode_vae_image(
            video_tensor=video_tensor,
            vae=components.vae,
            generator=block_state.generator,
            device=device,
            dtype=dtype,
            latent_channels=components.num_channels_latents,
        )

        self.set_block_state(state, block_state)
        return components, state
