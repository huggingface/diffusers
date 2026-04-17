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

import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import AutoencoderKLWan
from ...utils import is_ftfy_available, logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HeliosModularPipeline


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


def get_t5_prompt_embeds(
    text_encoder: UMT5EncoderModel,
    tokenizer: AutoTokenizer,
    prompt: str | list[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    """Encode text prompts into T5 embeddings for Helios.

    Args:
        text_encoder: The T5 text encoder model.
        tokenizer: The tokenizer for the text encoder.
        prompt: The prompt or prompts to encode.
        max_sequence_length: Maximum sequence length for tokenization.
        device: Device to place tensors on.
        dtype: Optional dtype override. Defaults to `text_encoder.dtype`.

    Returns:
        A tuple of `(prompt_embeds, attention_mask)` where `prompt_embeds` is the encoded text embeddings and
        `attention_mask` is a boolean mask.
    """
    dtype = dtype or text_encoder.dtype

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

    return prompt_embeds, text_inputs.attention_mask.bool()


class HeliosTextEncoderStep(ModularPipelineBlocks):
    model_name = "helios"

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text embeddings to guide the video generation"

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
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam.template("max_sequence_length"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
        ]

    @staticmethod
    def check_inputs(prompt, negative_prompt):
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if prompt is not None and negative_prompt is not None:
            prompt_list = [prompt] if isinstance(prompt, str) else prompt
            neg_list = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if type(prompt_list) is not type(neg_list):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if len(prompt_list) != len(neg_list):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(neg_list)}, but `prompt` has batch size"
                    f" {len(prompt_list)}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        prompt = block_state.prompt
        negative_prompt = block_state.negative_prompt
        max_sequence_length = block_state.max_sequence_length
        device = components._execution_device

        self.check_inputs(prompt, negative_prompt)

        # Encode prompt
        block_state.prompt_embeds, _ = get_t5_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode negative prompt
        block_state.negative_prompt_embeds = None
        if components.requires_unconditional_embeds:
            negative_prompt = negative_prompt or ""
            if isinstance(prompt, list) and isinstance(negative_prompt, str):
                negative_prompt = len(prompt) * [negative_prompt]

            block_state.negative_prompt_embeds, _ = get_t5_prompt_embeds(
                text_encoder=components.text_encoder,
                tokenizer=components.tokenizer,
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        self.set_block_state(state, block_state)
        return components, state


class HeliosImageVaeEncoderStep(ModularPipelineBlocks):
    """Encodes an input image into VAE latent space for image-to-video generation."""

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Image Encoder step that encodes an input image into VAE latent space, "
            "producing image_latents (first frame prefix) and fake_image_latents (history seed) "
            "for image-to-video generation."
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
            InputParam.template("image"),
            InputParam.template("height", default=384),
            InputParam.template("width", default=640),
            InputParam(
                "num_latent_frames_per_chunk",
                default=9,
                type_hint=int,
                description="Number of latent frames per temporal chunk.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("image_latents"),
            OutputParam(
                "fake_image_latents", type_hint=torch.Tensor, description="Fake image latents for history seeding"
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        vae = components.vae
        device = components._execution_device

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        # Preprocess image to 4D tensor (B, C, H, W)
        image = components.video_processor.preprocess(
            block_state.image, height=block_state.height, width=block_state.width
        )
        image_5d = image.unsqueeze(2).to(device=device, dtype=vae.dtype)  # (B, C, 1, H, W)

        # Encode image to get image_latents
        image_latents = vae.encode(image_5d).latent_dist.sample(generator=block_state.generator)
        image_latents = (image_latents - latents_mean) * latents_std

        # Encode fake video to get fake_image_latents
        min_frames = (block_state.num_latent_frames_per_chunk - 1) * components.vae_scale_factor_temporal + 1
        fake_video = image_5d.repeat(1, 1, min_frames, 1, 1)  # (B, C, min_frames, H, W)
        fake_latents_full = vae.encode(fake_video).latent_dist.sample(generator=block_state.generator)
        fake_latents_full = (fake_latents_full - latents_mean) * latents_std
        fake_image_latents = fake_latents_full[:, :, -1:, :, :]

        block_state.image_latents = image_latents.to(device=device, dtype=torch.float32)
        block_state.fake_image_latents = fake_image_latents.to(device=device, dtype=torch.float32)

        self.set_block_state(state, block_state)
        return components, state


class HeliosVideoVaeEncoderStep(ModularPipelineBlocks):
    """Encodes an input video into VAE latent space for video-to-video generation.

    Produces `image_latents` (first frame) and `video_latents` (remaining frames encoded in chunks).
    """

    model_name = "helios"

    @property
    def description(self) -> str:
        return (
            "Video Encoder step that encodes an input video into VAE latent space, "
            "producing image_latents (first frame) and video_latents (chunked video frames) "
            "for video-to-video generation."
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
            InputParam("video", required=True, description="Input video for video-to-video generation"),
            InputParam.template("height", default=384),
            InputParam.template("width", default=640),
            InputParam(
                "num_latent_frames_per_chunk",
                default=9,
                type_hint=int,
                description="Number of latent frames per temporal chunk.",
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("image_latents"),
            OutputParam("video_latents", type_hint=torch.Tensor, description="Encoded video latents (chunked)"),
        ]

    @torch.no_grad()
    def __call__(self, components: HeliosModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        vae = components.vae
        device = components._execution_device
        num_latent_frames_per_chunk = block_state.num_latent_frames_per_chunk

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        # Preprocess video
        video = components.video_processor.preprocess_video(
            block_state.video, height=block_state.height, width=block_state.width
        )
        video = video.to(device=device, dtype=vae.dtype)

        # Encode video into latents
        num_frames = video.shape[2]
        min_frames = (num_latent_frames_per_chunk - 1) * 4 + 1
        num_chunks = num_frames // min_frames
        if num_chunks == 0:
            raise ValueError(
                f"Video must have at least {min_frames} frames "
                f"(got {num_frames} frames). "
                f"Required: (num_latent_frames_per_chunk - 1) * 4 + 1 = ({num_latent_frames_per_chunk} - 1) * 4 + 1 = {min_frames}"
            )
        total_valid_frames = num_chunks * min_frames
        start_frame = num_frames - total_valid_frames

        # Encode first frame
        first_frame = video[:, :, 0:1, :, :]
        image_latents = vae.encode(first_frame).latent_dist.sample(generator=block_state.generator)
        image_latents = (image_latents - latents_mean) * latents_std

        # Encode remaining frames in chunks
        latents_chunks = []
        for i in range(num_chunks):
            chunk_start = start_frame + i * min_frames
            chunk_end = chunk_start + min_frames
            video_chunk = video[:, :, chunk_start:chunk_end, :, :]
            chunk_latents = vae.encode(video_chunk).latent_dist.sample(generator=block_state.generator)
            chunk_latents = (chunk_latents - latents_mean) * latents_std
            latents_chunks.append(chunk_latents)
        video_latents = torch.cat(latents_chunks, dim=2)

        block_state.image_latents = image_latents.to(device=device, dtype=torch.float32)
        block_state.video_latents = video_latents.to(device=device, dtype=torch.float32)

        self.set_block_state(state, block_state)
        return components, state
