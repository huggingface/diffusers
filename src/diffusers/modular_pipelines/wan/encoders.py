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

import html

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
    prompt: str | list[str],
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
    device: torch.device | None = None,
):
    image = image_processor(images=image, return_tensors="pt").to(device)
    image_embeds = image_encoder(**image, output_hidden_states=True)
    return image_embeds.hidden_states[-2]


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
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam("max_sequence_length", default=512),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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
        device: torch.device | None = None,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: str | None = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            prepare_unconditional_embeds (`bool`):
                whether to use prepare unconditional embeddings or not
            negative_prompt (`str` or `list[str]`, *optional*):
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", type_hint=PIL.Image.Image, required=True),
            InputParam("height", type_hint=int, default=480),
            InputParam("width", type_hint=int, default=832),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "resized_image", type_hint=PIL.Image.Image, required=True, description="The resized first frame image"
            ),
            InputParam("last_image", type_hint=PIL.Image.Image, required=True, description="The last frameimage"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("image_processor", CLIPImageProcessor),
            ComponentSpec("image_encoder", CLIPVisionModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("image_processor", CLIPImageProcessor),
            ComponentSpec("image_encoder", CLIPVisionModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
            InputParam("resized_last_image", type_hint=PIL.Image.Image, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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


class WanVaeEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Vae Image Encoder step that generate condition_latents based on first frame image to guide the video generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("num_frames", type_hint=int, default=81),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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
        vae_dtype = components.vae.dtype

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
            dtype=vae_dtype,
            latent_channels=components.num_channels_latents,
        )

        self.set_block_state(state, block_state)
        return components, state


class WanVaceEncoderStep(ModularPipelineBlocks):
    model_name = "wan-vace"

    @property
    def description(self) -> str:
        return (
            "Vace Encoder step that preprocesses the control video, mask and reference images and encodes them "
            "into the conditioning latents used by the VACE control branch of the transformer"
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video",
                type_hint=list[PIL.Image.Image],
                description="The control video to condition the generation on. If not provided, an empty video is used.",
            ),
            InputParam(
                "mask",
                type_hint=list[PIL.Image.Image],
                description="The mask that defines which video regions to condition on (black) and which to generate (white). Can only be passed if `video` is passed as well.",
            ),
            InputParam(
                "reference_images",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                description="One or more reference images as extra conditioning for the generation.",
            ),
            InputParam(
                "conditioning_scale",
                type_hint=float | list[float] | torch.Tensor,
                default=1.0,
                description="The conditioning scale applied in each control layer of the model. If a float, it is applied uniformly to all layers; a list or tensor must have the same length as the number of control layers.",
            ),
            InputParam("height"),
            InputParam("width"),
            InputParam("num_frames", type_hint=int, default=81),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "vace_conditioning_latents",
                type_hint=torch.Tensor,
                description="The concatenated video and mask conditioning latents fed into the VACE control branch of the transformer",
            ),
            OutputParam(
                "num_reference_images",
                type_hint=int,
                description="Number of reference images prepended on the frame dimension of the conditioning latents",
            ),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        base = components.vae_scale_factor_spatial * components.patch_size_spatial
        if (block_state.height is not None and block_state.height % base != 0) or (
            block_state.width is not None and block_state.width % base != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {base} but are {block_state.height} and {block_state.width}."
            )
        if block_state.num_frames is not None and (
            block_state.num_frames < 1 or (block_state.num_frames - 1) % components.vae_scale_factor_temporal != 0
        ):
            raise ValueError(
                f"`num_frames` has to be greater than 0, and (num_frames - 1) must be divisible by {components.vae_scale_factor_temporal}, but got {block_state.num_frames}."
            )
        if isinstance(block_state.generator, list):
            raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")
        if block_state.video is not None:
            if block_state.mask is not None and len(block_state.video) != len(block_state.mask):
                raise ValueError(
                    f"Length of `video` {len(block_state.video)} and `mask` {len(block_state.mask)} do not match. Please make sure that"
                    " they have the same length."
                )
        elif block_state.mask is not None:
            raise ValueError("`mask` can only be passed if `video` is passed as well.")

    @staticmethod
    def preprocess_conditions(
        components,
        video,
        mask,
        reference_images,
        height,
        width,
        num_frames,
        dtype,
        device,
    ):
        if video is not None:
            base = components.vae_scale_factor_spatial * components.patch_size_spatial
            video_height, video_width = components.video_processor.get_default_height_width(video[0])

            if video_height * video_width > height * width:
                scale = min(width / video_width, height / video_height)
                video_height, video_width = int(video_height * scale), int(video_width * scale)

            if video_height % base != 0 or video_width % base != 0:
                logger.warning(
                    f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}. "
                )
                video_height = (video_height // base) * base
                video_width = (video_width // base) * base

            video = components.video_processor.preprocess_video(video, video_height, video_width)
            image_size = (video_height, video_width)  # Use the height/width of video (with possible rescaling)
        else:
            video = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            image_size = (height, width)  # Use the height/width provider by user

        if mask is not None:
            mask = components.video_processor.preprocess_video(mask, image_size[0], image_size[1])
            mask = torch.clamp((mask + 1) / 2, min=0, max=1)
        else:
            mask = torch.ones_like(video)

        video = video.to(dtype=dtype, device=device)
        mask = mask.to(dtype=dtype, device=device)

        # Make a list of list of images where the outer list corresponds to video batch size and the inner list
        # corresponds to list of conditioning images per video
        if reference_images is None or isinstance(reference_images, PIL.Image.Image):
            reference_images = [[reference_images] for _ in range(video.shape[0])]
        elif isinstance(reference_images, (list, tuple)) and isinstance(next(iter(reference_images)), PIL.Image.Image):
            reference_images = [reference_images]
        elif (
            isinstance(reference_images, (list, tuple))
            and isinstance(next(iter(reference_images)), list)
            and isinstance(next(iter(reference_images[0])), PIL.Image.Image)
        ):
            reference_images = reference_images
        else:
            raise ValueError(
                "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
                f"`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
            )

        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

        reference_images_preprocessed = []
        for i, reference_images_batch in enumerate(reference_images):
            preprocessed_images = []
            for j, image in enumerate(reference_images_batch):
                if image is None:
                    continue
                image = components.video_processor.preprocess(image, None, None)
                img_height, img_width = image.shape[-2:]
                scale = min(image_size[0] / img_height, image_size[1] / img_width)
                new_height, new_width = int(img_height * scale), int(img_width * scale)
                resized_image = torch.nn.functional.interpolate(
                    image, size=(new_height, new_width), mode="bilinear", align_corners=False
                ).squeeze(0)  # [C, H, W]
                top = (image_size[0] - new_height) // 2
                left = (image_size[1] - new_width) // 2
                canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
                canvas[:, top : top + new_height, left : left + new_width] = resized_image
                preprocessed_images.append(canvas)
            reference_images_preprocessed.append(preprocessed_images)

        return video, mask, reference_images_preprocessed

    @staticmethod
    def prepare_video_latents(components, video, mask, reference_images, generator, device):
        vae_dtype = components.vae.dtype
        video = video.to(dtype=vae_dtype)

        latents_mean = torch.tensor(components.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, components.vae.config.z_dim, 1, 1, 1
        )
        latents_std = 1.0 / torch.tensor(components.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, components.vae.config.z_dim, 1, 1, 1
        )

        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        inactive = video * (1 - mask)
        reactive = video * mask
        inactive = retrieve_latents(components.vae.encode(inactive), generator, sample_mode="argmax")
        reactive = retrieve_latents(components.vae.encode(reactive), generator, sample_mode="argmax")
        inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive, reactive], dim=1)

        latent_list = []
        for latent, reference_images_batch in zip(latents, reference_images):
            for reference_image in reference_images_batch:
                reference_image = reference_image.to(dtype=vae_dtype)
                reference_image = reference_image[None, :, None, :, :]  # [1, C, 1, H, W]
                reference_latent = retrieve_latents(
                    components.vae.encode(reference_image), generator, sample_mode="argmax"
                )
                reference_latent = ((reference_latent.float() - latents_mean) * latents_std).to(vae_dtype)
                reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
                reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
                latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
            latent_list.append(latent)
        return torch.stack(latent_list)

    @staticmethod
    def prepare_masks(components, mask, reference_images):
        transformer_patch_size = components.patch_size_spatial

        mask_list = []
        for mask_, reference_images_batch in zip(mask, reference_images):
            num_channels, num_frames, height, width = mask_.shape
            new_num_frames = (
                num_frames + components.vae_scale_factor_temporal - 1
            ) // components.vae_scale_factor_temporal
            new_height = (
                height // (components.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            )
            new_width = (
                width // (components.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            )
            mask_ = mask_[0, :, :, :]
            mask_ = mask_.view(
                num_frames,
                new_height,
                components.vae_scale_factor_spatial,
                new_width,
                components.vae_scale_factor_spatial,
            )
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
            mask_ = torch.nn.functional.interpolate(
                mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
            ).squeeze(0)
            num_ref_images = len(reference_images_batch)
            if num_ref_images > 0:
                mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
                mask_ = torch.cat([mask_padding, mask_], dim=1)
            mask_list.append(mask_)
        return torch.stack(mask_list)

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        device = components._execution_device
        dtype = torch.float32

        height = block_state.height or components.default_height
        width = block_state.width or components.default_width
        num_frames = block_state.num_frames or components.default_num_frames

        video, mask, reference_images = self.preprocess_conditions(
            components,
            block_state.video,
            block_state.mask,
            block_state.reference_images,
            height,
            width,
            num_frames,
            dtype,
            device,
        )
        if video.shape[0] != 1:
            raise ValueError(
                "Generating with more than one video is not yet supported. This may be supported in the future."
            )
        block_state.num_reference_images = len(reference_images[0])

        conditioning_latents = self.prepare_video_latents(
            components, video, mask, reference_images, block_state.generator, device
        )
        mask = self.prepare_masks(components, mask, reference_images)
        block_state.vace_conditioning_latents = torch.cat([conditioning_latents, mask], dim=1)

        conditioning_scale = block_state.conditioning_scale
        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * components.num_vace_layers
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != components.num_vace_layers:
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {components.num_vace_layers}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != components.num_vace_layers:
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match number of layers {components.num_vace_layers}."
                )
            conditioning_scale = conditioning_scale.to(device=device)
        block_state.conditioning_scale = conditioning_scale

        self.set_block_state(state, block_state)
        return components, state


class WanPrepareFirstFrameLatentsStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "step that prepares the masked first frame latents and add it to the latent condition"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("first_frame_latents", type_hint=torch.Tensor | None),
            InputParam("num_frames", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("image_condition_latents", type_hint=torch.Tensor | None),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        batch_size, _, _, latent_height, latent_width = block_state.first_frame_latents.shape

        mask_lat_size = torch.ones(batch_size, 1, block_state.num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, block_state.num_frames))] = 0

        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=components.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, components.vae_scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(block_state.first_frame_latents.device)
        block_state.image_condition_latents = torch.concat([mask_lat_size, block_state.first_frame_latents], dim=1)

        self.set_block_state(state, block_state)
        return components, state


class WanFirstLastFrameVaeEncoderStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Vae Image Encoder step that generate condition_latents based on first and last frame images to guide the video generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
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
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("resized_image", type_hint=PIL.Image.Image, required=True),
            InputParam("resized_last_image", type_hint=PIL.Image.Image, required=True),
            InputParam("height"),
            InputParam("width"),
            InputParam("num_frames", type_hint=int, default=81),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
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
        vae_dtype = components.vae.dtype

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
            dtype=vae_dtype,
            latent_channels=components.num_channels_latents,
        )

        self.set_block_state(state, block_state)
        return components, state


class WanPrepareFirstLastFrameLatentsStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "step that prepares the masked latents with first and last frames and add it to the latent condition"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("first_last_frame_latents", type_hint=torch.Tensor | None),
            InputParam("num_frames", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("image_condition_latents", type_hint=torch.Tensor | None),
        ]

    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        batch_size, _, _, latent_height, latent_width = block_state.first_last_frame_latents.shape

        mask_lat_size = torch.ones(batch_size, 1, block_state.num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, block_state.num_frames - 1))] = 0

        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=components.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, components.vae_scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(block_state.first_last_frame_latents.device)
        block_state.image_condition_latents = torch.concat(
            [mask_lat_size, block_state.first_last_frame_latents], dim=1
        )

        self.set_block_state(state, block_state)
        return components, state
