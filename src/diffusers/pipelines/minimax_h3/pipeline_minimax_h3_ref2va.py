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

import os
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel
from ...models.autoencoders.vae import DiagonalGaussianDistribution
from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    align_num_frames,
    audio_latent_num_frames,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    build_ref2va_packed_sequence,
    build_ref2va_presentation,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    reference_kind,
    reference_media_to_uint8,
    resample_reference_frames,
    resolve_reference_image_size,
    sample_reference_video_frames,
    trim_reference_num_frames,
)
from .pipeline_output import MiniMaxH3PipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import MiniMaxH3Ref2VAPipeline
        >>> from diffusers.pipelines.minimax_h3 import MiniMaxH3Reference
        >>> from diffusers.utils import load_image, load_video
        >>> from diffusers.utils.export_utils import encode_video

        >>> # One repository holds both checkpoint partitions: this pipeline loads `transformer_ref/`, while
        >>> # `MiniMaxH3Pipeline.from_pretrained` on the same repository loads `transformer/`.
        >>> pipe = MiniMaxH3Ref2VAPipeline.from_pretrained(
        ...     "MiniMaxAI/MiniMax-H3", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> subject_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )

        >>> # A single reference may name its medium directly.
        >>> output = pipe(
        ...     prompt="The subject walks toward camera, turning slightly, natural arm swing",
        ...     image=subject_image,
        ...     generator=torch.Generator().manual_seed(42),
        ... )

        >>> # Any ordered mix of image, video and audio references goes through `references`, and the order is the
        >>> # order the model reads them in. A reference takes a path or a URL and decodes it as it is built, rates
        >>> # included: a video brings its own frame rate and its soundtrack along, and an audio clip its sample rate.
        >>> output = pipe(
        ...     prompt="The character speaks in time with the reference recording, natural lip movement",
        ...     references=[
        ...         MiniMaxH3Reference(image=subject_image),
        ...         MiniMaxH3Reference(
        ...             video="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        ...         ),
        ...         MiniMaxH3Reference(audio="voice.wav"),
        ...     ],
        ... )

        >>> # In-memory media instead says which rates it carries, MiniMax-H3's own 24 fps and the audio VAE's own
        >>> # sample rate by default.
        >>> motion = load_video("motion_ref.mp4")  # decoded frames, no soundtrack, at the file's own rate
        >>> output = pipe(
        ...     prompt="The subject walks toward camera, matching the reference video's shot rhythm",
        ...     references=[
        ...         MiniMaxH3Reference(image=subject_image),
        ...         MiniMaxH3Reference(video=motion, fps=30.0),
        ...     ],
        ... )

        >>> encode_video(
        ...     output.frames[0], fps=24, output_path="output.mp4",
        ...     audio=output.audio[0], audio_sample_rate=pipe.audio_sampling_rate,
        ... )
        ```
"""


class MiniMaxH3Ref2VAPipeline(DiffusionPipeline):
    r"""
    Pipeline for joint video + audio generation from omni-references with MiniMax-H3, the `ref2va` task of the Ref2VA
    checkpoint.

    A request carries an ordered list of references — up to 9 images, 3 videos and 3 audio clips, 12 in total — and
    MiniMax-H3 packs one block per reference in front of the generated rows, all inside the single sequence its
    transformer runs full self-attention over. The order is semantic: it labels the references in the prompt
    presentation (`"<Picture 1>"`, `"<Audio 1>"`, `"<Video 1>"`) and it advances the shared audio/video rotary clock,
    so reordering the same references is a different request.

    Unlike the keyframes of [`MiniMaxH3Pipeline`], references do not bind the generated geometry: they are encoded at
    their own resolution and the target canvas defaults to MiniMax-H3's 16:9. The duration may be left to a reference
    soundtrack, but only when exactly one reference carries audio.

    The checkpoint is guidance-distilled: `guidance_scale` is fixed at 1.0, there is no negative prompt, and every
    step runs exactly one forward pass.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer_ref ([`MiniMaxH3Transformer3DModel`]):
            The packed-sequence transformer of the Ref2VA checkpoint partition. It is named `transformer_ref` so that
            one repository can hold both partitions, `transformer/` for [`MiniMaxH3Pipeline`] and `transformer_ref/`
            for this one, and either pipeline picks its own weights up with no extra argument.
        vae ([`AutoencoderKLMiniMaxH3`]):
            Video VAE, 16x spatially and 4x temporally. Also encodes the image and video references.
        audio_vae ([`AutoencoderKLMiniMaxH3Audio`]):
            Audio VAE, a waveform autoencoder at 32 kHz with 40 latents per second. Also encodes the reference
            soundtracks.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            The conditioner. MiniMax-H3 reads the hidden state after its 50th decoder layer, so the released 64-layer
            checkpoint is used with its language-model head unused; a checkpoint truncated below 51 layers is
            rejected, since the last hidden state of a truncated stack is post-norm and roughly an order of magnitude
            off in scale.
        tokenizer ([`Qwen2TokenizerFast`]):
            Tokenizer of the conditioner.
        processor ([`Qwen3VLProcessor`]):
            Processor of the conditioner, used to turn the image and video references into vision patches.
        scheduler ([`MiniMaxH3Scheduler`]):
            The video-latent schedule (`shift = 12.0` in the released checkpoint).
        audio_scheduler ([`MiniMaxH3Scheduler`]):
            The audio-latent schedule (`shift = 3.0` in the released checkpoint).
    """

    model_cpu_offload_seq = "text_encoder->transformer_ref->vae->audio_vae"
    _callback_tensor_inputs = ["latents", "audio_latents", "prompt_embeds"]

    def __init__(
        self,
        transformer_ref: MiniMaxH3Transformer3DModel,
        vae: AutoencoderKLMiniMaxH3,
        audio_vae: AutoencoderKLMiniMaxH3Audio,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: Qwen2TokenizerFast,
        processor: Qwen3VLProcessor,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
    ):
        super().__init__()

        self.register_modules(
            transformer_ref=transformer_ref,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 16
        )
        self.vae_latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) is not None else 24
        self.audio_sampling_rate = (
            self.audio_vae.config.sampling_rate if getattr(self, "audio_vae", None) is not None else 32000
        )
        self.audio_latent_channels = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 32
        )
        self.patch_size = (
            tuple(self.transformer_ref.config.patch_size)
            if getattr(self, "transformer_ref", None) is not None
            else (1, 2, 2)
        )
        # The video VAE decodes into ImageNet-normalized RGB over a [0, 1] base range, which the pipeline reverts
        # itself, so the processor must not denormalize a second time.
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, do_normalize=False)

    def check_inputs(self, prompt, prompt_embeds, text_token_tags, references, height, width, num_frames):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Pass one of `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(prompt)}."
            )
        if (prompt_embeds is None) != (text_token_tags is None):
            raise ValueError("`prompt_embeds` and `text_token_tags` have to be passed together.")
        if prompt_embeds is not None and prompt_embeds.shape[1] != text_token_tags.shape[0]:
            raise ValueError(
                f"`text_token_tags` must hold one tag per row of `prompt_embeds`, got {text_token_tags.shape[0]} tags "
                f"for {prompt_embeds.shape[1]} rows."
            )
        if (height is None) != (width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if height is not None and (height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE):
            raise ValueError(
                f"`height` and `width` must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, got {height}x{width}."
            )
        # The duration the request generates is the one of the *aligned* frame count, so that is what the ceiling has
        # to hold for: 346 frames would otherwise pass the check and then be rounded up to 362, i.e. 15.083 seconds.
        aligned_num_frames = None if num_frames is None else align_num_frames(num_frames)
        duration = None if aligned_num_frames is None else aligned_num_frames / MINIMAX_H3_FPS
        if duration is not None and not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates between {MINIMAX_H3_MIN_DURATION} and {MINIMAX_H3_MAX_DURATION} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS)} and "
                f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)}, got {num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )

        if not references:
            raise ValueError("`ref2va` needs at least one reference; use `MiniMaxH3Pipeline` for text-only requests.")
        kinds = [reference_kind(index, entry) for index, entry in enumerate(references)]
        for kind, limit in (
            ("image", MINIMAX_H3_MAX_REFERENCE_IMAGES),
            ("video", MINIMAX_H3_MAX_REFERENCE_VIDEOS),
            ("audio", MINIMAX_H3_MAX_REFERENCE_AUDIOS),
        ):
            if kinds.count(kind) > limit:
                raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
        if len(kinds) > MINIMAX_H3_MAX_REFERENCES:
            raise ValueError(
                f"MiniMax-H3 accepts at most {MINIMAX_H3_MAX_REFERENCES} references in total, got {len(kinds)}."
            )
        if set(kinds) == {"audio"}:
            raise ValueError(
                "An audio reference has to be paired with at least one image or video reference and cannot be used "
                "on its own."
            )

    def prepare_references(
        self, references: list[MiniMaxH3Reference], num_frames: int | None
    ) -> tuple[list[MiniMaxH3PreparedReference], int]:
        r"""
        Resolve the references and, if it was left open, the duration they imply.

        Every reference is prepared at its own resolution: an image is resized to a 2048 pixel short edge, a video is
        resampled onto MiniMax-H3's own 24 fps, rescaled onto the 768 pixel canvas of *its own* aspect ratio and
        truncated to the generated frame count, and a soundtrack is put on the audio VAE's sample rate and truncated to
        the generated duration. None of this touches the target canvas.

        A reference that left its `fps` or its `sample_rate` out is taken to already be at MiniMax-H3's own rate, and
        its frames or its samples then flow through untouched.

        A video reference goes through the two passes the reference implementation's `ffmpeg` decode applied, in the
        same order: the constant frame rate resample of `resample_reference_frames` and the LANCZOS rescale of
        `prepare_reference_frames`. Frames handed over at 24 fps and already at the canvas their own aspect ratio
        resolves to therefore reach the VAE untouched, which is the parity-exact route.

        Args:
            references (`list[MiniMaxH3Reference]`):
                The `references` argument of [`~MiniMaxH3Ref2VAPipeline.__call__`].
            num_frames (`int`, *optional*):
                The requested frame count, or `None` to derive it from the single audio-bearing reference.

        Returns:
            `tuple[list[MiniMaxH3PreparedReference], int]`: the prepared references, in packed order, and the frame
            count.
        """
        resolved = [
            MiniMaxH3PreparedReference(kind=reference_kind(index, entry), has_audio=entry.has_audio)
            for index, entry in enumerate(references)
        ]

        # The duration may be left open, but then exactly one reference may carry audio, or the request is ambiguous.
        if num_frames is None:
            audio_bearing = [index for index, reference in enumerate(resolved) if reference.has_audio]
            if len(audio_bearing) != 1:
                raise ValueError(
                    "`num_frames` may only be left to the references when exactly one of them carries audio, got "
                    f"{len(audio_bearing)}."
                )
            index = audio_bearing[0]
            sample_rate = references[index].sample_rate or self.audio_sampling_rate
            duration = references[index].audio.shape[-1] / sample_rate
            if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
                raise ValueError(
                    f"`references[{index}]` is {duration:g} seconds long, outside the "
                    f"{MINIMAX_H3_MIN_DURATION} to {MINIMAX_H3_MAX_DURATION} seconds MiniMax-H3 generates."
                )
            num_frames = align_num_frames(round(duration * MINIMAX_H3_FPS))
            # The duration the request generates is the one of the *aligned* frame count, so that is what the
            # ceiling has to hold for: a 14.99 second soundtrack rounds up to 362 frames, i.e. 15.083 seconds.
            if num_frames / MINIMAX_H3_FPS > MINIMAX_H3_MAX_DURATION:
                raise ValueError(
                    f"`references[{index}]` is {duration:g} seconds long, which rounds up to {num_frames} frames "
                    f"(`17 * n + 5`), i.e. {num_frames / MINIMAX_H3_FPS:g} seconds — past the "
                    f"{MINIMAX_H3_MAX_DURATION} seconds MiniMax-H3 generates. Pass `num_frames` to generate a "
                    "shorter video from this soundtrack."
                )
        num_frames = align_num_frames(num_frames)

        for reference, entry in zip(resolved, references):
            if reference.kind == "image":
                image = entry.image
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(reference_media_to_uint8(image))
                image = ImageOps.exif_transpose(image).convert("RGB")
                height, width = resolve_reference_image_size(*image.size)
                reference.image = prepare_reference_image(image, height, width)
            elif reference.kind == "video":
                frames = resample_reference_frames(reference_media_to_uint8(entry.video), float(entry.fps))
                reference.frames = prepare_reference_frames(frames, num_frames)
            if reference.has_audio:
                reference.waveform = prepare_reference_waveform(
                    entry.audio,
                    entry.sample_rate or self.audio_sampling_rate,
                    self.audio_sampling_rate,
                    max_duration=num_frames / MINIMAX_H3_FPS,
                )
        return resolved, num_frames

    def encode_prompt(
        self,
        prompt: str,
        references: list[MiniMaxH3PreparedReference],
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
            references (`list[MiniMaxH3PreparedReference]`):
                The prepared references, in packed order, as returned by
                [`~MiniMaxH3Ref2VAPipeline.prepare_references`].
            device (`torch.device`, *optional*): The device to run the conditioner on.
            dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden states and the
            `(num_text_tokens,)` per-row modality tags.
        """
        device = device or self._execution_device
        dtype = dtype or self.transformer_ref.dtype

        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"MiniMax-H3 conditions on `hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}]` of its Qwen3-VL "
                f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
                f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER} layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        merge_size = self.processor.image_processor.merge_size**2
        pixel_values, image_grid_thw, image_token_counts = None, None, []
        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            image_token_counts = [int(grid.prod()) // merge_size for grid in image_grid_thw]

        pixel_values_videos, video_grid_thw, video_block_token_counts = None, None, []
        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            sampled = [sample_reference_video_frames(reference.frames) for reference in videos]
            for reference, (_, block_timestamps) in zip(videos, sampled):
                reference.block_timestamps = block_timestamps
            vision = self.processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
            )
            pixel_values_videos, video_grid_thw = vision["pixel_values_videos"], vision["video_grid_thw"]
            video_block_token_counts = [int(grid[1]) * int(grid[2]) // merge_size for grid in video_grid_thw]
            for reference, grid in zip(videos, video_grid_thw):
                if int(grid[0]) != len(reference.block_timestamps):
                    raise ValueError(
                        f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax-H3 "
                        f"labels {len(reference.block_timestamps)} of them."
                    )

        token_ids, token_tags = build_ref2va_presentation(
            self.tokenizer, prompt, references, image_token_counts, video_block_token_counts
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # Qwen3-VL lays its 3D rotary positions out per modality run, which it reads off the token type ids the
        # processor derives from the vision pad ids (`0` text, `1` image, `2` video).
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
        )
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None if pixel_values is None else pixel_values.to(device, self.text_encoder.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
            pixel_values_videos=(
                None if pixel_values_videos is None else pixel_values_videos.to(device, self.text_encoder.dtype)
            ),
            video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(device),
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=device, dtype=dtype)
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    def encode_references(
        self, references: list[MiniMaxH3PreparedReference], device: torch.device | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        r"""
        Encode the references into packed conditioning rows, and resolve their latent geometry.

        Image and video references go through the video VAE with the same recipe the `fl2va` keyframes use: the
        posterior is *sampled* under a generator seeded with 42 independently of the request seed, and the sampled
        latent is rounded to float16 before being normalized. An image is a single frame and is encoded by the spatial
        encoder alone, while a video reference goes through the 17-frames-per-5-latents temporal chunking. Reference
        soundtracks instead take the posterior *mean*, and are never sampled.

        Args:
            references (`list[MiniMaxH3PreparedReference]`):
                The prepared references, in packed order. Their latent geometry is filled in here.
            device (`torch.device`, *optional*): The device to run the VAEs on.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(num_condition_video_rows, latent_channels * prod(patch_size))`
            video rows and the `(num_condition_audio_rows, audio_latent_channels)` audio rows, both float32 on CPU and
            both `None` when the references carry no such rows.
        """
        device = device or self._execution_device
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)

        video_rows, audio_rows = [], []
        for reference in references:
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
                else:
                    frames = reference.frames[: trim_reference_num_frames(reference.frames.shape[0])]
                    pixels = torch.from_numpy(frames.copy()).to(device).permute(3, 0, 1, 2)[None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                # A single frame is encoded by the (tiled) spatial encoder alone; a video goes through the temporal
                # chunking, which is what turns `17 * n + 5` frames into `5 * n + 2` latent frames.
                moments = self.vae._encode_clip(pixels) if reference.kind == "image" else self.vae._encode(pixels)
                posterior = DiagonalGaussianDistribution(moments)
                latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
                # The sampled latent is rounded to float16 before it is normalized: ~11 bits of every conditioning
                # latent, so the released model's conditioning cannot be reproduced without it.
                latents = latents.to(torch.float16).float().cpu()
                reference.num_latent_frames = latents.shape[2]
                reference.latent_height, reference.latent_width = latents.shape[3], latents.shape[4]
                video_rows.append(patchify_video_latents((latents - latents_mean) / latents_std, self.patch_size))

            if reference.has_audio:
                posterior = self.audio_vae.encode(reference.waveform.to(device)[:, None], return_dict=False)[0]
                # Channel-major rows: the two stereo channels are two batch items of the mono audio VAE.
                latents = posterior.mode().float().cpu().transpose(1, 2)
                reference.num_audio_latents = latents.shape[1]
                normalized = (latents - audio_latents_mean) / audio_latents_std
                audio_rows.append(normalized.reshape(-1, self.audio_latent_channels))

        return (
            torch.cat(video_rows) if video_rows else None,
            torch.cat(audio_rows) if audio_rows else None,
        )

    # Copied from diffusers.pipelines.minimax_h3.pipeline_minimax_h3.MiniMaxH3Pipeline.prepare_latents
    def prepare_latents(
        self,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Draw the initial noise of both modalities and pack it into transformer rows.

        A request draws every stream from the one generator it is given, and the order is part of what that generator
        reproduces: the conditioning noise of the keyframes or references first (one draw per condition, in
        [`~pipelines.minimax_h3.packing.keyframe_condition_noise`]), then the video noise here, as a latent tensor
        that is patchified afterwards, then the audio noise, directly in row layout. Passing `latents` or
        `audio_latents` skips its draw and shifts the ones after it.

        Args:
            num_latent_frames (`int`): Number of video latent frames.
            latent_height (`int`): Latent height.
            latent_width (`int`): Latent width.
            num_audio_latents (`int`): Number of audio latents per channel.
            device (`torch.device`): The device the rows are drawn on.
            generator (`torch.Generator`, *optional*): The generator of the request.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video noise of shape `(1, latent_channels, num_latent_frames, latent_height,
                latent_width)`, used instead of the draw.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio noise of shape `(2, audio_latent_channels, num_audio_latents)`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the video rows and the channel-major audio rows.
        """
        if latents is None:
            latents = randn_tensor(
                (1, self.vae_latent_channels, num_latent_frames, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(latents.to(torch.float32), self.patch_size)

        if audio_latents is None:
            audio_rows = randn_tensor(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, self.audio_latent_channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            audio_rows = audio_latents.to(torch.float32).permute(0, 2, 1).reshape(-1, self.audio_latent_channels)
        return video_rows.to(device), audio_rows.to(device)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | None = None,
        references: list[MiniMaxH3Reference] | None = None,
        image: str | os.PathLike | Image.Image | np.ndarray | torch.Tensor | None = None,
        video: str | os.PathLike | list[Image.Image] | np.ndarray | torch.Tensor | None = None,
        fps: float | None = None,
        audio: str | os.PathLike | torch.Tensor | None = None,
        sample_rate: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        text_token_tags: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents", "audio_latents"],
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str`, *optional*):
                The prompt to guide generation. MiniMax-H3 packs one request into one sequence, so only a single
                prompt is accepted.
            references (`list[MiniMaxH3Reference]`):
                The references to condition on, **in the order the model should read them**: it labels them in the
                prompt presentation and lays them out on the shared rotary clock, so a different order is a different
                request. Every [`MiniMaxH3Reference`] carries exactly one medium, which names its modality, and it is
                in-memory media by the time it reaches this pipeline: a path or a URL was decoded when the reference
                was built, so no pipeline of this model opens a media file.

                - `MiniMaxH3Reference(image=image)` — a subject, style or scene reference, at most 9 of them,
                - `MiniMaxH3Reference(video=video)` — a motion and camera reference, at most 3. Its `fps` is the rate
                  the frames carry, a decoded file's own or 24.0 for in-memory frames, which is MiniMax-H3's own clock:
                  anything else is resampled onto it by dropping and duplicating whole frames, exactly as the reference
                  implementation's decode did. The soundtrack may be passed alongside, as `audio`, and is then
                  conditioned on as this reference's own, packed immediately before its frames,
                - `MiniMaxH3Reference(audio=audio)` — a voice or music reference, at most 3, which never reaches the
                  conditioner and is encoded by the audio VAE alone. Its waveform is a `(channels, num_samples)` mono
                  or stereo tensor, at the `sample_rate` it says it carries or the audio VAE's own, and is resampled
                  onto the audio VAE's rate.

                At most 12 references in total, and audio references cannot be the only ones.
            image (`str`, `os.PathLike`, `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`, *optional*):
                A single image reference, the same request as `references=[MiniMaxH3Reference(image=image)]`. Mutually
                exclusive with `references`.
            video (`str`, `os.PathLike`, `list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`, *optional*):
                A single video reference, the same request as `references=[MiniMaxH3Reference(video=video)]`. Mutually
                exclusive with `references`.
            fps (`float`, *optional*):
                The frame rate `video` carries its frames at, only read for a single `video` reference: a `references`
                list carries the rate of every video on the reference itself. Left out, it is the rate a decoded file
                reports and MiniMax-H3's own 24 fps for in-memory frames.
            audio (`str`, `os.PathLike` or `torch.Tensor` of shape `(channels, num_samples)`, *optional*):
                A single audio reference, the same request as `references=[MiniMaxH3Reference(audio=audio)]`, or the
                soundtrack of a `video` passed with it. Mutually exclusive with `references`, and rejected on its own,
                as an audio reference has to be paired with an image or a video.
            sample_rate (`int`, *optional*):
                The rate `audio` carries its samples at. Left out, it is the rate a decoded file reports and, for an
                in-memory waveform, the audio VAE's own.
            height (`int`, *optional*):
                Height of the generated video, a multiple of 32. Defaults, together with `width`, to MiniMax-H3's own
                768-short-edge 16:9 canvas — references never bind the generated geometry.
            width (`int`, *optional*):
                Width of the generated video, a multiple of 32.
            num_frames (`int`, *optional*):
                Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the VAE can
                decode; the resulting duration must stay between 5 and 15 seconds. May be left out, but only when
                exactly one reference carries audio, in which case the duration is that soundtrack's.
            num_inference_steps (`int`, defaults to `50`):
                Number of denoising steps.
            generator (`torch.Generator`, *optional*):
                A generator to make generation deterministic. Every noise draw of a request — the reference
                conditioning noise, the video noise and the audio noise, in that order — comes off this one
                generator, so the same generator state reproduces a sample.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video noise, of shape `(1, 24, num_latent_frames, height // 16, width // 16)`.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio noise, of shape `(2, 32, num_audio_latents)`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed conditioning of shape `(1, num_text_tokens, 5120)`, as returned by
                [`~MiniMaxH3Ref2VAPipeline.encode_prompt`], which skips the conditioner. Must be passed together with
                `text_token_tags`, and must have been encoded from the very references passed as `references` — those
                are still needed here, for their conditioning latents.
            text_token_tags (`torch.Tensor`, *optional*):
                The per-row modality tags that go with `prompt_embeds`.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generated video: `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~pipelines.minimax_h3.MiniMaxH3PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Passed to the attention processors, and may carry a `scale` entry for the LoRA layers.
            callback_on_step_end (`Callable`, *optional*):
                Called at the end of every denoising step with `(self, step, timestep, callback_kwargs)`. The tensors
                it receives are named by `callback_on_step_end_tensor_inputs`; note that `latents` and `audio_latents`
                are the packed rows of the whole sequence during the loop, references included.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents", "audio_latents"]`):
                Which local tensors to pass to `callback_on_step_end`.

        Examples:

        Returns:
            [`~pipelines.minimax_h3.MiniMaxH3PipelineOutput`] or `tuple`:
                The generated video and its soundtrack.
        """
        for name in callback_on_step_end_tensor_inputs:
            if name not in self._callback_tensor_inputs:
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, got {name}."
                )

        # 0. A single-reference request may name its medium directly, which is the very same request as a one-item
        # `references` list: build that list here, so there is one code path from now on.
        if image is not None or video is not None or audio is not None:
            if references is not None:
                raise ValueError(
                    "Pass either `references` or a single reference as `image` / `video` / `audio`, not both."
                )
            references = [MiniMaxH3Reference(image=image, video=video, fps=fps, audio=audio, sample_rate=sample_rate)]

        # 1. Resolve the geometry and prepare the references. The canvas is MiniMax-H3's own 16:9 unless asked
        # otherwise, and the duration may come from a reference soundtrack.
        self.check_inputs(prompt, prompt_embeds, text_token_tags, references, height, width, num_frames)
        if height is None:
            height, width = resolve_canvas_size(16, 9)
        requested_num_frames = num_frames
        references, num_frames = self.prepare_references(references, num_frames)
        if requested_num_frames is not None and requested_num_frames != num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {requested_num_frames} up "
                f"to {num_frames}."
            )

        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        num_latent_frames = video_latent_num_frames(num_frames)
        num_audio_latents = audio_latent_num_frames(num_frames)

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        device = self._execution_device

        # 2. Encode the prompt, including a labelled vision block per image and video reference.
        if prompt_embeds is None:
            prompt_embeds, text_token_tags = self.encode_prompt(prompt, references, device=device)

        # 3. Encode the references, then noise the visual ones to their conditioning level. Audio references are the
        # exception: they ride along clean, at `t = 1.0`. The anchors are fixed for the whole loop, since the loop
        # only ever writes the target rows.
        condition_rows, audio_condition_rows = self.encode_references(references, device=device)
        if condition_rows is not None:
            noise = keyframe_condition_noise(
                tuple(
                    (reference.num_latent_frames, reference.latent_height, reference.latent_width)
                    for reference in references
                    if reference.kind != "audio"
                ),
                self.patch_size,
                self.vae_latent_channels,
                generator=generator,
                device=device,
            )
            condition_rows = self.scheduler.scale_noise(
                condition_rows.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise
            )
        if audio_condition_rows is not None:
            audio_condition_rows = audio_condition_rows.to(device)

        # 4. Build the packed layout: [text | reference blocks | target audio | target video].
        layout = build_ref2va_packed_sequence(
            text_token_tags,
            references,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
        )
        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)

        # 5. The two schedules, and the noise.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.audio_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, audio_timesteps = self.scheduler.timesteps, self.audio_scheduler.timesteps
        self._num_timesteps = len(timesteps)

        latents, audio_latents = self.prepare_latents(
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            device,
            generator,
            latents,
            audio_latents,
        )
        if condition_rows is not None:
            latents = torch.cat([condition_rows, latents])
        if audio_condition_rows is not None:
            audio_latents = torch.cat([audio_condition_rows, audio_latents])
        num_condition_rows = layout.num_condition_video_rows
        num_condition_audio_rows = layout.num_condition_audio_rows

        # 6. Denoise. One forward per step serves every modality and every noise level at once: the reference rows
        # ride along at their own timestep and are never updated. The row-to-timestep assignment is static per step,
        # so the whole plan is staged before the loop rather than rebuilt on every one of them.
        timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in build_row_timesteps(
                    layout, float(t), float(audio_t), max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG), 1.0
                )
            )
            for t, audio_t in zip(timesteps, audio_timesteps)
        ]

        with self.progress_bar(total=self._num_timesteps) as progress_bar:
            for i, (t, audio_t) in enumerate(zip(timesteps, audio_timesteps)):
                if self.interrupt:
                    continue

                unique_timesteps, timestep_indices = timestep_plan[i]
                noise_pred, audio_noise_pred = self.transformer_ref(
                    hidden_states=latents[None],
                    audio_hidden_states=audio_latents[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps,
                    timestep_indices=timestep_indices,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )

                latents[num_condition_rows:] = self.scheduler.step(
                    noise_pred[0, num_condition_rows:].float(),
                    t,
                    latents[num_condition_rows:],
                    return_dict=False,
                )[0]
                audio_latents[num_condition_audio_rows:] = self.audio_scheduler.step(
                    audio_noise_pred[0, num_condition_audio_rows:].float(),
                    audio_t,
                    audio_latents[num_condition_audio_rows:],
                    return_dict=False,
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[name] = locals()[name]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    audio_latents = callback_outputs.pop("audio_latents", audio_latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        # 7. Unpack the generated rows. The spatial tiling of the VAE covers the canvas exactly (tiles and overlaps
        # are both latent-aligned), so the decoded frames need no crop back to `latent_height * 16`.
        latents = unpatchify_video_tokens(
            latents[num_condition_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae_latent_channels,
            self.patch_size,
        )
        audio_latents = unpack_audio_tokens(audio_latents[num_condition_audio_rows:], num_audio_latents)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        latents = latents * latents_std + latents_mean
        audio_latents = audio_latents * audio_latents_std + audio_latents_mean

        if output_type == "latent":
            video, audio = latents, audio_latents
        else:
            # MiniMax-H3 decodes video under float16 autocast even though the VAE weights are float32.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                video = self.vae.decode(latents, return_dict=False)[0]
            pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            # The audio VAE is mono and takes the two stereo channels as two batch items.
            audio = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = audio.float().permute(1, 0, 2)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)
        return MiniMaxH3PipelineOutput(frames=video, audio=audio)
