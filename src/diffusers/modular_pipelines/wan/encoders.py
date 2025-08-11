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
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLWan
from ...utils import is_ftfy_available, logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import PipelineBlock, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import WanModularPipeline


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


class WanTextEncoderStep(PipelineBlock):
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
    def expected_configs(self) -> List[ConfigSpec]:
        return []

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam("attention_kwargs"),
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
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str) and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}")

    @staticmethod
    def _get_t5_prompt_embeds(
        components,
        prompt: Union[str, List[str]],
        max_sequence_length: int,
        device: torch.device,
    ):
        dtype = components.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]

        text_inputs = components.tokenizer(
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
        prompt_embeds = components.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        return prompt_embeds

    @staticmethod
    def encode_prompt(
        components,
        prompt: str,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        prepare_unconditional_embeds: bool = True,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            prepare_unconditional_embeds (`bool`):
                whether to use prepare unconditional embeddings or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of text tokens to be used for the generation process.
        """
        device = device or components._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = WanTextEncoderStep._get_t5_prompt_embeds(components, prompt, max_sequence_length, device)

        if prepare_unconditional_embeds and negative_prompt_embeds is None:
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

            negative_prompt_embeds = WanTextEncoderStep._get_t5_prompt_embeds(
                components, negative_prompt, max_sequence_length, device
            )

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if prepare_unconditional_embeds:
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.prepare_unconditional_embeds = components.guider.num_conditions > 1
        block_state.device = components._execution_device

        # Encode input prompt
        (
            block_state.prompt_embeds,
            block_state.negative_prompt_embeds,
        ) = self.encode_prompt(
            components,
            block_state.prompt,
            block_state.device,
            1,
            block_state.prepare_unconditional_embeds,
            block_state.negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # Add outputs
        self.set_block_state(state, block_state)
        return components, state


class WanImageEncoderStep(PipelineBlock):
    model_name = "wan"

    @property
    def description(self) -> str:
        return "Image Encoder step to compute image embeddings to guide the video generation"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("image_encoder", CLIPVisionModel),
            ComponentSpec("image_processor", CLIPImageProcessor),
        ]

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return []

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "image",
                required=True,
                description="The input image to condition the generation on for first-frame conditioned video generation.",
            ),
            InputParam(
                "last_image",
                required=False,
                description="The last image to condition the generation on for last-frame conditioned video generation.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "image_embeds",
                type_hint=torch.Tensor,
                description="image embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.image, PipelineImageInput):
            raise ValueError(f"`image` has to be of type `PipelineImageInput` but is {type(block_state.image)}.")
        if block_state.last_image is not None and not isinstance(block_state.last_image, PipelineImageInput):
            raise ValueError(
                f"`last_image` has to be of type `PipelineImageInput` but is {type(block_state.last_image)}."
            )

    @staticmethod
    def encode_image(
        components,
        image: Union[PipelineImageInput, List[PipelineImageInput]],
        device: torch.device,
    ):
        image = components.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = components.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        # Get inputs and intermediates
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.prepare_unconditional_embeds = components.guider.num_conditions > 1
        block_state.device = components._execution_device

        # Encode input images
        image = block_state.image
        if block_state.last_image is not None:
            image = [block_state.image, block_state.last_image]

        block_state.image_embeds = self.encode_image(components, image, block_state.device)

        # Add outputs
        self.set_block_state(state, block_state)
        return components, state


class WanVaeEncoderStep(PipelineBlock):
    model_name = "wan"

    @property
    def description(self) -> str:
        return (
            "VAE encode step that encodes the input image/last_image to latents for conditioning the video generation"
        )

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
            InputParam("image", required=True),
            InputParam("last_image", required=False),
        ]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam("height", type_hint=int),
            InputParam("width", type_hint=int),
            InputParam("num_frames", type_hint=int),
            InputParam("batch_size", type_hint=int),
            InputParam("generator"),
            InputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latent_condition",
                type_hint=torch.Tensor,
                description="The latents representing the reference first-frame/last-frame for conditioned video generation.",
            ),
            OutputParam("num_channels_latents", type_hint=int),
        ]

    @staticmethod
    def _encode_vae_image(
        components: WanModularPipeline,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        image: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        last_image: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        latent_height = height // components.vae_scale_factor_spatial
        latent_width = width // components.vae_scale_factor_spatial

        latents_mean = (
            torch.tensor(components.vae.config.latents_mean)
            .view(1, components.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(components.vae.config.latents_std).view(
            1, components.vae.config.z_dim, 1, 1, 1
        ).to(device, dtype)

        image = image.unsqueeze(2)
        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=dtype)

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(components.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(components.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=components.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, components.vae_scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)
        latent_condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        return latent_condition

    @torch.no_grad()
    def __call__(self, components: WanModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.device = components._execution_device
        block_state.num_channels_latents = components.vae.config.z_dim

        block_state.image = components.video_processor.preprocess(
            block_state.image, height=block_state.height, width=block_state.width
        ).to(block_state.device, dtype=torch.float32)

        if block_state.last_image is not None:
            block_state.last_image = components.video_processor.preprocess(
                block_state.last_image, height=block_state.height, width=block_state.width
            ).to(block_state.device, dtype=torch.float32)

        block_state.latent_condition = self._encode_vae_image(
            components,
            block_state.batch_size,
            block_state.height,
            block_state.width,
            block_state.num_frames,
            block_state.image,
            block_state.device,
            block_state.dtype,
            block_state.last_image,
            block_state.generator,
        )

        self.set_block_state(state, block_state)

        return components, state
