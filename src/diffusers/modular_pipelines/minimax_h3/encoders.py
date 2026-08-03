# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline
from .packing_ref2va import (
    MiniMaxH3Reference,
    build_ref2va_presentation,
    sample_reference_video_frames,
    trim_reference_num_frames,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3TextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Encodes MiniMax-H3's presentation of a `t2va` / `fl2va` request: the prompt verbatim, preceded by a "
            '`"<Picture i>: "` label and a vision block per keyframe, with no chat template and no special tokens. '
            "The checkpoint is guidance-distilled, so there is no negative prompt and no unconditional branch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
            InputParam(
                name="keyframes",
                type_hint=list,
                description="The keyframes put onto the target canvas, in packed order (empty or None for `t2va`).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description=(
                    "The hidden state MiniMax-H3 conditions on, of shape `(1, num_text_tokens, 5120)`, read after the "
                    "50th decoder layer of the Qwen3-VL conditioner."
                ),
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description=(
                    "The per-row modality tag of every row of `prompt_embeds`; a vision block is tagged as video."
                ),
            ),
        ]

    @staticmethod
    def encode_prompt(
        text_encoder,
        tokenizer,
        processor,
        prompt: str,
        images: list | None = None,
        *,
        text_encoder_layer: int,# can you set the default?
        text_tag: int,# can you set the default?
        video_tag: int, # can you set the default?
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build MiniMax-H3's presentation of a request and encode it.

        The presentation is the verbatim prompt for `t2va`. Every keyframe prepends a `"<Picture i>: "` label and a
        vision block (`<|vision_start|>`, one `<|image_pad|>` per vision patch, `<|vision_end|>`) — no chat template
        and no special tokens. The rows of a vision block are tagged as *video* rather than text, which is what the
        transformer's AdaLN modulation keys off.

        Args:
            prompt (`str`): The prompt to encode.
            images (`list[PIL.Image.Image]`, *optional*):
                The keyframes, already prepared onto the target canvas, in packed order.
            device (`torch.device`, *optional*): The device to run the conditioner on.
            dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden states and the
            `(num_text_tokens,)` per-row modality tags.
        """

        num_layers = text_encoder.config.text_config.num_hidden_layers
        if num_layers <= text_encoder_layer:
            raise ValueError(
                f"MiniMax-H3 conditions on `hidden_states[{text_encoder_layer}]` of its Qwen3-VL "
                f"conditioner, which needs more than {text_encoder_layer} decoder layers, but "
                f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
                f"{text_encoder_layer} layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        pixel_values, image_grid_thw = None, None
        token_ids, token_tags = [], []
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge_size = processor.image_processor.merge_size**2
            for index in range(len(images)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vision_ids)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [text_tag] * len(prompt_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # Qwen3-VL lays its 3D rotary positions out per modality run, which it reads off the token type ids the
        # processor derives from the vision pad ids (`0` text, `1` image, `2` video).
        mm_token_type_ids = torch.tensor(
            processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
        )
        # `text_encoder.model` is a submodule, and a CPU-offload hook — accelerate's or the one the
        # `ComponentsManager` attaches — wraps the *top-level* module's `forward` alone, so calling the submodule
        # directly would leave the conditioner on the CPU. Fire the hook by hand instead of routing through
        # `text_encoder(...)`: MiniMax-H3 reads `hidden_states[50]` and never uses the language-model head, whose
        # vocabulary-wide projection over every token is all the top-level forward would add.
        # TODO: firing another module's offload hook by hand is not something a block should do. It is here
        # because MiniMax-H3 reads `hidden_states[50]` off `text_encoder.model` while the hook wraps only the
        # top-level `forward`. Needs a real answer — an opt-in on the conditioner, or a hook that follows
        # submodule calls.
        hook = getattr(text_encoder, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(text_encoder)
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None if pixel_values is None else pixel_values.to(device, text_encoder.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[text_encoder_layer].to(device=device, dtype=dtype)
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(block_state.prompt)}."
            )

        # `encode_prompt` defaults the embedding dtype to the denoiser's; a text encoder block has no denoiser of
        # its own — it is meant to run on its own — so it emits the conditioner's dtype, as every other model does.
        block_state.prompt_embeds, block_state.text_token_tags = self.encode_prompt(
            components.text_encoder,
            components.tokenizer,
            components.processor,
            block_state.prompt,
            block_state.keyframes,
            text_encoder_layer=components.text_encoder_layer,
            text_tag=components.text_tag,
            video_tag=components.video_tag,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3KeyframeVaeEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Encodes the `fl2va` keyframes into conditioning latents. They become the anchors of the whole denoising "
            "loop, which only ever writes the generated rows, so they are never updated again — the prepare-latents "
            "step noises them to MiniMax-H3's conditioning level and packs them."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLMiniMaxH3)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="keyframes",
                type_hint=list,
                required=True,
                description="The keyframes put onto the target canvas, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The normalized video conditioning latents, one `(1, latent_channels, 1, latent_height, "
                    "latent_width)` tensor per keyframe, in packed order."
                ),
            )
        ]

    @staticmethod
    def encode_keyframes(
        vae, images: list, pixel_mean: tuple, pixel_std: tuple, encode_seed: int, device: torch.device
    ) -> list[torch.Tensor]:
        r"""
        Encode the `fl2va` keyframes into normalized conditioning latents.

        A keyframe is a single frame, so `vae.encode` runs its spatial encoder alone with none of the 17-frame
        temporal chunking. The posterior is *sampled*, under a generator seeded with 42 independently of the request
        seed, and the sampled latent is rounded to float16 before being normalized, as in the reference
        implementation; both are part of reproducing the released model's conditioning.

        Args:
            vae (`AutoencoderKLMiniMaxH3`): The video VAE.
            images (`list[PIL.Image.Image]`):
                The keyframes, already prepared onto the target canvas, in packed order.
            pixel_mean (`tuple[float, float, float]`), pixel_std (`tuple[float, float, float]`):
                The video VAE's pixel convention, i.e. `components.pixel_mean` / `components.pixel_std`.
            encode_seed (`int`): Seed the posterior is sampled under, i.e. `components.keyframe_encode_seed`.
            device (`torch.device`): The device to run the VAE on.

        Returns:
            `list[torch.Tensor]`: one `(1, latent_channels, 1, latent_height, latent_width)` float32 CPU tensor per
            keyframe, in packed order. One entry per condition is what the prepare-latents step draws its noise
            against, so the list is the unit the request's generator is consumed in.
        """
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(pixel_std, device=device).view(1, -1, 1, 1, 1)

        keyframe_latents = []
        for image in images:
            pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            posterior = vae.encode(pixels, return_dict=False)[0]
            latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
            # The sampled latent is rounded to float16 before it is normalized: ~11 bits of every conditioning
            # latent, so the released model's conditioning cannot be reproduced without it.
            latents = latents.to(torch.float16).float().cpu()
            keyframe_latents.append((latents - latents_mean) / latents_std)
        return keyframe_latents

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.condition_latents = self.encode_keyframes(
            components.vae,
            block_state.keyframes,
            components.pixel_mean,
            components.pixel_std,
            components.keyframe_encode_seed,
            device,
        )

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VATextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    @property
    def description(self) -> str:
        return (
            "Encodes MiniMax-H3's presentation of a `ref2va` request: a label per reference, numbered per modality "
            '(`"<Picture i>: "` plus a vision block, `"<Audio j>: "` alone, `"<Video k>: "` plus one timestamped '
            "vision block per merged frame pair), then the prompt verbatim. The checkpoint is guidance-distilled, so "
            "there is no negative prompt and no unconditional branch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
            InputParam(
                name="prepared_references",
                type_hint=list[MiniMaxH3Reference],
                required=True,
                description="The prepared references, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description=(
                    "The hidden state MiniMax-H3 conditions on, of shape `(1, num_text_tokens, 5120)`, read after the "
                    "50th decoder layer of the Qwen3-VL conditioner."
                ),
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description=(
                    "The per-row modality tag of every row of `prompt_embeds`; a vision block is tagged as video."
                ),
            ),
        ]

    @staticmethod
    def encode_prompt(
        text_encoder,
        tokenizer,
        processor,
        prompt: str,
        references: list[MiniMaxH3Reference],
        *,
        text_encoder_layer: int, # can you add default
        text_tag: int, # can you add default
        video_tag: int, # can you add default
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build MiniMax-H3's presentation of a `ref2va` request and encode it.

        Every reference prepends a label, in request order and numbered per modality: `"<Picture i>: "` plus a vision
        block for an image, `"<Audio j>: "` alone for audio (a waveform never reaches the conditioner), and
        `"<Video k>: "` plus one timestamped vision block per merged frame pair for a video. A video that carries
        sound is labelled `"<Audio j>: "` *before* `"<Video k>: "`, mirroring the order its rows are packed in. The
        prompt then follows, verbatim, with no chat template and no special tokens. The rows of a vision block are
        tagged as *video* rather than text, which is what the transformer's AdaLN modulation keys off.

        Args:
            prompt (`str`): The prompt to encode.
            references (`list[MiniMaxH3Reference]`):
                The prepared references, in packed order, as returned by
                [`~MiniMaxH3Ref2VASetupStep.prepare_references`].
            device (`torch.device`, *optional*): The device to run the conditioner on.
            dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden states and the
            `(num_text_tokens,)` per-row modality tags.
        """

        num_layers = text_encoder.config.text_config.num_hidden_layers
        if num_layers <= text_encoder_layer:
            raise ValueError(
                f"MiniMax-H3 conditions on `hidden_states[{text_encoder_layer}]` of its Qwen3-VL "
                f"conditioner, which needs more than {text_encoder_layer} decoder layers, but "
                f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
                f"{text_encoder_layer} layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        merge_size = processor.image_processor.merge_size**2
        pixel_values, image_grid_thw, image_token_counts = None, None, []
        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            image_token_counts = [int(grid.prod()) // merge_size for grid in image_grid_thw]

        pixel_values_videos, video_grid_thw = None, None
        video_block_token_counts, video_block_timestamps = [], []
        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
            video_block_timestamps = [timestamps for _, timestamps in sampled]
            vision = processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
            )
            pixel_values_videos, video_grid_thw = vision["pixel_values_videos"], vision["video_grid_thw"]
            video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw]
            for timestamps, grid in zip(video_block_timestamps, video_grid_thw):
                if int(grid[0]) != len(timestamps):
                    raise ValueError(
                        f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax-H3 "
                        f"labels {len(timestamps)} of them."
                    )

        token_ids, token_tags = build_ref2va_presentation(
            tokenizer, prompt, references, image_token_counts, video_block_token_counts, video_block_timestamps
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # Qwen3-VL lays its 3D rotary positions out per modality run, which it reads off the token type ids the
        # processor derives from the vision pad ids (`0` text, `1` image, `2` video).
        mm_token_type_ids = torch.tensor(
            processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
        )
        # `text_encoder.model` is a submodule, and a CPU-offload hook — accelerate's or the one the
        # `ComponentsManager` attaches — wraps the *top-level* module's `forward` alone, so calling the submodule
        # directly would leave the conditioner on the CPU. Fire the hook by hand instead of routing through
        # `text_encoder(...)`: MiniMax-H3 reads `hidden_states[50]` and never uses the language-model head, whose
        # vocabulary-wide projection over every token is all the top-level forward would add.
        # TODO: firing another module's offload hook by hand is not something a block should do. It is here
        # because MiniMax-H3 reads `hidden_states[50]` off `text_encoder.model` while the hook wraps only the
        # top-level `forward`. Needs a real answer — an opt-in on the conditioner, or a hook that follows
        # submodule calls.
        hook = getattr(text_encoder, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(text_encoder)
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None if pixel_values is None else pixel_values.to(device, text_encoder.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
            pixel_values_videos=(
                None if pixel_values_videos is None else pixel_values_videos.to(device, text_encoder.dtype)
            ),
            video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(device),
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[text_encoder_layer].to(device=device, dtype=dtype)
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(block_state.prompt)}."
            )

        # `encode_prompt` defaults the embedding dtype to the denoiser's; a text encoder block has no denoiser of
        # its own — it is meant to run on its own — so it emits the conditioner's dtype, as every other model does.
        block_state.prompt_embeds, block_state.text_token_tags = self.encode_prompt(
            components.text_encoder,
            components.tokenizer,
            components.processor,
            block_state.prompt,
            block_state.prepared_references,
            text_encoder_layer=components.text_encoder_layer,
            text_tag=components.text_tag,
            video_tag=components.video_tag,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VAReferenceEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    @property
    def description(self) -> str:
        return (
            "Encodes the `ref2va` references — image and video references through the video VAE, soundtracks through "
            "the audio VAE. They are the anchors of the whole denoising loop, which only ever writes the generated "
            "rows; the prepare-latents step noises the visual ones to MiniMax-H3's conditioning level and packs them, "
            "while soundtracks ride along clean at `t = 1.0`. The latent geometry of every reference is resolved "
            "here, so this runs before the packed layout is built."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLMiniMaxH3),
            ComponentSpec("audio_vae", AutoencoderKLMiniMaxH3Audio),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="prepared_references",
                type_hint=list[MiniMaxH3Reference],
                required=True,
                description="The references normalized by the setup step, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The encoded video conditioning latents of the image and video references, one `(1, "
                    "latent_channels, num_latent_frames, latent_height, latent_width)` tensor each in packed order, "
                    "or None when the references carry none."
                ),
            ),
            OutputParam(
                "audio_condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The clean audio conditioning rows of the reference soundtracks, one `(num_audio_latents * 2, "
                    "audio_latent_channels)` tensor per audio-bearing reference in packed order. One entry per "
                    "reference rather than one concatenated block, because the packed layout is built from the row "
                    "count of each."
                ),
            ),
        ]

    @staticmethod
    def encode_references(
        components, references: list[MiniMaxH3Reference], device: torch.device | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        r"""
        Encode the references into conditioning latents.

        Image and video references go through the video VAE with the same recipe the `fl2va` keyframes use: the
        posterior is *sampled* under a generator seeded with 42 independently of the request seed, and the sampled
        latent is rounded to float16 before being normalized. An image is a single frame and is encoded by the spatial
        encoder alone, while a video reference goes through the 17-frames-per-5-latents temporal chunking. Reference
        soundtracks instead take the posterior *mean*, and are never sampled.

        The latent geometry every later block keys off is the shape of what this returns, so nothing has to be
        written back onto the references.

        Args:
            references (`list[MiniMaxH3Reference]`):
                The references normalized by the setup step, in packed order.
            device (`torch.device`, *optional*): The device to run the VAEs on.

        Returns:
            `tuple[list[torch.Tensor], list[torch.Tensor]]`: one `(1, latent_channels, num_latent_frames,
            latent_height, latent_width)` tensor per image and video reference, and one `(num_audio_latents * 2,
            audio_latent_channels)` tensor per audio-bearing reference, both in packed order and float32 on CPU.
        """
        latents_mean = torch.tensor(components.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(components.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
        audio_latents_mean = torch.tensor(components.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_latents_std = torch.tensor(components.audio_vae.config.latents_std).view(1, 1, -1)

        video_latents, audio_rows = [], []
        for reference in references:
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
                else:
                    frames = reference.frames[
                        : trim_reference_num_frames(
                            reference.frames.shape[0],
                            components.vae_frames_per_chunk,
                            components.vae_latents_per_chunk,
                        )
                    ]
                    pixels = torch.from_numpy(frames.copy()).to(device).permute(3, 0, 1, 2)[None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                # A single frame is encoded by the (tiled) spatial encoder alone; a video goes through the temporal
                # chunking, which is what turns `17 * n + 5` frames into `5 * n + 2` latent frames.
                posterior = components.vae.encode(pixels, return_dict=False)[0]
                latents = posterior.sample(
                    generator=torch.Generator().manual_seed(components.keyframe_encode_seed)
                )
                # The sampled latent is rounded to float16 before it is normalized: ~11 bits of every conditioning
                # latent, so the released model's conditioning cannot be reproduced without it.
                latents = latents.to(torch.float16).float().cpu()
                video_latents.append((latents - latents_mean) / latents_std)

            if reference.has_audio:
                posterior = components.audio_vae.encode(reference.audio.to(device)[:, None], return_dict=False)[0]
                # Channel-major rows: the two stereo channels are two batch items of the mono audio VAE.
                latents = posterior.mode().float().cpu().transpose(1, 2)
                normalized = (latents - audio_latents_mean) / audio_latents_std
                audio_rows.append(normalized.reshape(-1, components.audio_latent_channels))

        return video_latents, audio_rows

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        block_state.condition_latents, block_state.audio_condition_latents = self.encode_references(
            components, block_state.prepared_references, device=device
        )

        self.set_block_state(state, block_state)
        return components, state
