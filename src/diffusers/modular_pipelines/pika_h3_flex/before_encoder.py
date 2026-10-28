# Copyright 2026 Mellis, Inc. (Pika), The MiniMax and HuggingFace Teams. All rights reserved.
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
from PIL import Image

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import (
    PikaH3FlexModularPipeline,
    align_num_frames,
    resolve_canvas_size,
    resolve_tier_rule,
)
from .references import (
    PikaH3FlexAudioReference,
    PikaH3FlexImageReference,
    PikaH3FlexReference,
    PikaH3FlexVideoReference,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PikaH3FlexResizeStep(ModularPipelineBlocks):
    model_name = "pika-h3-flex"

    @property
    def description(self) -> str:
        return (
            "Puts the `fl2va` keyframes onto the target canvas — Pika H3 Flex's 768-short-edge geometry for the "
            "aspect ratio of the first keyframe unless `height` and `width` say otherwise — the way the released "
            "package fits them: every keyframe is center-cropped to the canvas' aspect ratio at its own resolution, "
            "then resized onto the canvas, so nothing is ever stretched. The canvas resolved here is the one the "
            "whole request generates at."
        )

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        # The canvas Pika H3 Flex generates on by default, the 768p tier of its bucket table. Every block that
        # resolves one declares these, so a request at another tier passes `height` and `width`.
        return [ConfigSpec("canvas_short_edge", 768), ConfigSpec("canvas_max_pixels", 768 * 1344)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="image",
                type_hint=PIL.Image.Image,
                description=(
                    "Keyframe the video starts from. It is center-cropped to the canvas' aspect ratio and resized "
                    "onto it; the canvas by default is derived from its own aspect ratio."
                ),
            ),
            InputParam(
                name="last_image",
                type_hint=PIL.Image.Image,
                description=(
                    "Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. It is "
                    "center-cropped to the canvas' aspect ratio and resized onto it, like `image`."
                ),
            ),
            InputParam.template("height", description="Height of the generated video in pixels, a multiple of 32."),
            InputParam.template("width", description="Width of the generated video in pixels, a multiple of 32."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam(
                "keyframes",
                type_hint=list,
                description="The keyframes put onto the target canvas, in packed order.",
            ),
            OutputParam(
                "keyframe_anchors",
                type_hint=tuple,
                description=(
                    "Which end of the video every keyframe is anchored to, in packed order. Positional with "
                    "`keyframes`, so both are resolved here."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PikaH3FlexModularPipeline, state: PipelineState
    ) -> tuple[PikaH3FlexModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        keyframes = [keyframe for keyframe in (block_state.image, block_state.last_image) if keyframe is not None]
        block_state.keyframe_anchors = tuple(
            anchor
            for anchor, keyframe in (("first", block_state.image), ("last", block_state.last_image))
            if keyframe is not None
        )
        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(
                *keyframes[0].size,
                components.canvas_multiple,
                components.config.canvas_short_edge,
                components.config.canvas_max_pixels,
            )

        # The released package crops every keyframe to the canvas' aspect ratio at its source resolution — the crop
        # box rounds its size and centers it with `(source - crop) // 2` — and LANCZOS-resizes the crop onto the
        # canvas. `VaeImageProcessor`'s `resize_mode="crop"` resizes first and crops after, which lands on other
        # pixels, so the released arithmetic is kept.
        prepared = []
        aspect = block_state.width / block_state.height
        for keyframe in keyframes:
            source_width, source_height = keyframe.size
            if source_width / source_height > aspect:
                crop_height = source_height
                crop_width = max(1, min(source_width, round(source_height * aspect)))
            else:
                crop_width = source_width
                crop_height = max(1, min(source_height, round(source_width / aspect)))
            left = (source_width - crop_width) // 2
            top = (source_height - crop_height) // 2
            cropped = keyframe.crop((left, top, left + crop_width, top + crop_height))
            prepared.append(cropped.resize((block_state.width, block_state.height), Image.Resampling.LANCZOS))
        block_state.keyframes = prepared

        self.set_block_state(state, block_state)
        return components, state


class PikaH3FlexRef2VASetupStep(ModularPipelineBlocks):
    model_name = "pika-h3-flex"

    def __init__(
        self,
        max_images: int = 9,
        max_videos: int = 3,
        max_audios: int = 3,
        max_references: int = 12,
        min_reference_duration: float = 2.0,
        max_reference_duration: float = 15.0,
    ):
        r"""
        Resolve a `ref2va` request's plan and normalize its references onto Pika H3 Flex's own rates and resolutions.

        Args:
            max_images (`int`, defaults to 9): Image references a request may carry.
            max_videos (`int`, defaults to 3): Video references a request may carry.
            max_audios (`int`, defaults to 3): Audio references a request may carry.
            max_references (`int`, defaults to 12): References of any modality a request may carry in total.
            min_reference_duration (`float`, defaults to 2.0):
                Shortest video or standalone audio reference a request may carry, in seconds.
            max_reference_duration (`float`, defaults to 15.0):
                Longest video or standalone audio reference, in seconds, which is also the ceiling the video
                references' durations and the standalone audio references' durations each sum to.

        The limits are what the released package accepts; they bound nothing but this block's own validation.
        """
        self.max_images = max_images
        self.max_videos = max_videos
        self.max_audios = max_audios
        self.max_references = max_references
        self.min_reference_duration = min_reference_duration
        self.max_reference_duration = max_reference_duration
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Resolves the `ref2va` plan — the canvas (Pika H3 Flex's own 16:9 unless asked otherwise; references "
            "never bind the generated geometry) and the `17 * n + 5` frame count — and normalizes every reference "
            "onto the model's own rates and the resolutions of the canvas' tier: an image reference is encoded at the "
            "tier's reference short edge (1024 pixels below the 768 tier, 2048 from it upwards) and a video reference "
            "is put on the tier's own canvas rule."
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
    def expected_configs(self) -> list[ConfigSpec]:
        # The canvas Pika H3 Flex generates on by default, the 768p tier of its bucket table. The resolution a
        # reference is conditioned at is not configured here: it follows the tier of the canvas the request resolves.
        return [ConfigSpec("canvas_short_edge", 768), ConfigSpec("canvas_max_pixels", 768 * 1344)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="references",
                type_hint=list[PikaH3FlexReference],
                required=True,
                description=(
                    "The references to condition on, **in the order the model should read them**: the order labels "
                    "them in the prompt presentation and lays them out on the shared rotary clock, so a different "
                    "order is a different request. One dataclass per modality, all holding in-memory media — a "
                    "[`PikaH3FlexImageReference`] (at most 9), a [`PikaH3FlexVideoReference`] at its own `fps` (at "
                    "most 3, whose `audio` soundtrack is conditioned on as well), or a [`PikaH3FlexAudioReference`] "
                    "at its own `sample_rate` (at most 3) — for at most 12 references in total, and audio references "
                    "cannot be the only ones. Every video or standalone audio reference runs 2 to 15 seconds, and the "
                    "video durations and the standalone audio durations each sum to at most 15 seconds. These blocks "
                    "never open a media file: decode with each class's `from_file` classmethod, which brings the "
                    "rates along."
                ),
            ),
            InputParam.template("height", description="Height of the generated video in pixels, a multiple of 32."),
            InputParam.template("width", description="Width of the generated video in pixels, a multiple of 32."),
            InputParam(
                name="num_frames",
                type_hint=int,
                required=True,
                description=(
                    "Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video "
                    "VAE can decode; the resulting duration must stay between 5 and 15 seconds. To generate a video "
                    "as long as a reference soundtrack, pass `round(samples / sample_rate * 24)`."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5."),
            OutputParam(
                "normalized_references",
                type_hint=list[PikaH3FlexReference],
                description=(
                    "The references normalized onto Pika H3 Flex's own rates and the resolutions of the canvas' "
                    "tier, in packed order: the same public reference types the request passed in, with an image "
                    "resized to the tier's reference short edge, a video resampled onto 24 fps and onto the canvas "
                    "its own aspect ratio resolves to under the tier's rule, and a soundtrack put on the audio VAE's "
                    "sample rate and truncated to the generated duration."
                ),
            ),
        ]

    @staticmethod
    # Copied from diffusers.modular_pipelines.minimax_h3.before_encoder.MiniMaxH3Ref2VASetupStep._normalize_video_condition
    def _normalize_video_condition(
        frames,
        fps: float,
        num_frames: int,
        canvas_multiple: int,
        canvas_short_edge: int,
        canvas_max_pixels: int,
        target_fps: float,
    ) -> np.ndarray:
        r"""
        Normalize a video reference's frames: any accepted layout, onto `uint8` at 24 fps, truncated to the generated
        frame count, on the canvas its own aspect ratio resolves to.

        The two passes reproduce the reference implementation's `ffmpeg` decode, in the same order: the constant frame
        rate resample first (dropping and duplicating whole frames, as `ffmpeg`'s `fps` filter does), the LANCZOS
        rescale second. Frames handed over at 24 fps and already at the canvas their own aspect ratio resolves to flow
        through untouched, which is the parity-exact route: the reference rescaled with `ffmpeg`'s own LANCZOS while
        decoding, so only frames decoded at the canvas reproduce its pixels bit for bit.

        Args:
            frames (`list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`):
                The reference frames: a list of images, a `(num_frames, height, width, 3)` array or a `(num_frames, 3,
                height, width)` tensor, `uint8` or floating point over `[0, 1]`.
            fps (`float`): The frame rate `frames` carries.
            num_frames (`int`): The generated frame count the reference is truncated to.
            canvas_multiple (`int`): What both canvas axes round to, i.e. `components.canvas_multiple`.
            canvas_short_edge (`int`), canvas_max_pixels (`int`):
                The canvas rule the reference is put on, i.e. `components.config.canvas_short_edge` and
                `components.config.canvas_max_pixels` — the same one the generated video follows.
            target_fps (`float`): The rate the frames are resampled onto, i.e. `components.fps`.

        Returns:
            `np.ndarray` of shape `(num_frames, height, width, 3)`: the normalized `uint8` RGB frames.
        """
        # Any accepted layout onto `uint8` THWC. A `torch.Tensor` is channels-first, as everywhere else in
        # diffusers, and a `np.ndarray` channels-last; floating point values are read over `[0, 1]`.
        if isinstance(frames, list):
            frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
        if isinstance(frames, torch.Tensor):
            frames = frames.movedim(-3, -1).cpu().numpy()
        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError(
                f"A reference video must be `(num_frames, height, width, 3)` RGB frames, got {tuple(frames.shape)}."
            )

        # Onto MiniMax-H3's 24 fps grid: every frame is held until the slot of the next one, and the last one until
        # the slot the stream's end rounds to.
        if fps <= 0:
            raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
        if fps != target_fps:
            scale = target_fps / fps
            slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
            frames = np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)

        # Truncated to the generated frame count and put on the canvas of its *own* aspect ratio — the same rule the
        # target canvas follows, unlike an image reference.
        frames = frames[:num_frames]
        height, width = resolve_canvas_size(
            frames.shape[2], frames.shape[1], canvas_multiple, canvas_short_edge, canvas_max_pixels
        )
        if frames.shape[1:3] == (height, width):
            return frames
        return np.stack(
            [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
        )

    @staticmethod
    def _normalize_audio_condition(
        waveform: torch.Tensor, sample_rate: int, target_sample_rate: int, max_samples: int
    ) -> torch.Tensor:
        r"""
        Normalize a reference soundtrack onto the audio VAE's sample rate, as a stereo waveform.

        The released package resamples a soundtrack onto the audio VAE's rate first and truncates it there, to the
        sample count that covers the generated frames, which this mirrors: the resampling is a single `torchaudio` pass
        and the truncation is applied at `target_sample_rate`. A mono waveform is upmixed by repeating its channel.

        Args:
            waveform (`torch.Tensor` of shape `(channels, num_samples)`): The soundtrack, mono or stereo.
            sample_rate (`int`): The sample rate `waveform` carries its samples at.
            target_sample_rate (`int`): The audio VAE's sample rate, i.e. what the waveform is resampled to.
            max_samples (`int`): Truncate the reference to this many samples at `target_sample_rate`.

        Returns:
            `torch.Tensor` of shape `(2, num_samples)`: the float32 waveform.
        """
        waveform = torch.as_tensor(waveform)
        if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
            raise ValueError(
                "A reference soundtrack must be a `(channels, num_samples)` mono or stereo waveform, got "
                f"{tuple(waveform.shape)}."
            )
        waveform = waveform.to(torch.float32)
        if waveform.shape[0] != 2:
            waveform = waveform.expand(2, -1)
        if sample_rate != target_sample_rate:
            try:
                import torchaudio
            except ImportError as error:
                raise ImportError(
                    f"Resampling a Pika H3 Flex reference soundtrack from {sample_rate} Hz to {target_sample_rate} "
                    "Hz needs `torchaudio`. Pass a waveform already at the audio VAE's sample rate to do without it."
                ) from error
            waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        return waveform[:, :max_samples].contiguous()

    @torch.no_grad()
    def __call__(
        self, components: PikaH3FlexModularPipeline, state: PipelineState
    ) -> tuple[PikaH3FlexModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        # 1. Validate the request.
        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        multiple = components.canvas_multiple
        if block_state.height is not None and (block_state.height % multiple or block_state.width % multiple):
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got {block_state.height}x{block_state.width}."
            )
        if not block_state.references:
            raise ValueError("`ref2va` needs at least one reference; use the `t2va` workflow for text-only requests.")
        for index, entry in enumerate(block_state.references):
            if not isinstance(entry, PikaH3FlexReference):
                raise ValueError(
                    f"`references[{index}]` must be a [`PikaH3FlexImageReference`], [`PikaH3FlexVideoReference`] or "
                    f"[`PikaH3FlexAudioReference`], got {type(entry)}. Pika H3 Flex blocks never open media files, so "
                    "a request that holds paths decodes them first, with each class's `from_file` classmethod."
                )
        kinds = [entry.kind for entry in block_state.references]
        for kind, limit in (("image", self.max_images), ("video", self.max_videos), ("audio", self.max_audios)):
            if kinds.count(kind) > limit:
                raise ValueError(f"Pika H3 Flex accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
        if len(kinds) > self.max_references:
            raise ValueError(
                f"Pika H3 Flex accepts at most {self.max_references} references in total, got {len(kinds)}."
            )
        if set(kinds) == {"audio"}:
            raise ValueError(
                "An audio reference has to be paired with at least one image or video reference and cannot be used "
                "on its own."
            )
        # The released package bounds the duration of every video and standalone audio reference, and the sum of
        # each kind; a video's own soundtrack rides with the video and counts toward neither audio bound.
        durations = {"video": 0.0, "audio": 0.0}
        for entry in block_state.references:
            if entry.kind == "video":
                duration = len(entry.frames) / float(entry.fps)
            elif entry.kind == "audio":
                sample_rate = entry.sample_rate if entry.sample_rate is not None else components.audio_sampling_rate
                duration = torch.as_tensor(entry.audio).shape[-1] / sample_rate
            else:
                continue
            if not self.min_reference_duration <= duration <= self.max_reference_duration + 1e-6:
                raise ValueError(
                    f"Every {entry.kind} reference must run {self.min_reference_duration:g} to "
                    f"{self.max_reference_duration:g} seconds, got one of {duration:.2f} seconds."
                )
            durations[entry.kind] += duration
        for kind, total in durations.items():
            if total > self.max_reference_duration + 1e-6:
                raise ValueError(
                    f"The {kind} references must run at most {self.max_reference_duration:g} seconds in total, got "
                    f"{total:.2f} seconds."
                )

        # 2. Resolve the canvas and the frame count. The duration the request generates is the one of the *aligned*
        # frame count, so that is what the ceiling holds for: 346 frames would otherwise pass the check and then be
        # rounded up to 362, i.e. 15.083 seconds.
        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(
                16, 9, multiple, components.config.canvas_short_edge, components.config.canvas_max_pixels
            )
        aligned_num_frames = align_num_frames(
            block_state.num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk
        )
        duration = aligned_num_frames / components.fps
        if not components.min_duration <= duration <= components.max_duration:
            raise ValueError(
                f"Pika H3 Flex generates between {components.min_duration} and {components.max_duration} seconds at "
                f"{components.fps} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(components.min_duration * components.fps)} and "
                f"{int(components.max_duration * components.fps)}, got {block_state.num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )
        if aligned_num_frames != block_state.num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {block_state.num_frames} "
                f"up to {aligned_num_frames}."
            )
        block_state.num_frames = aligned_num_frames
        # The references are conditioned at the resolutions of the tier the *generated* canvas falls in.
        tier = resolve_tier_rule(block_state.height, block_state.width)
        max_samples = round(block_state.num_frames / components.fps * components.audio_sampling_rate)

        # 3. Normalize the media, in packed order.
        normalized = []
        for entry in block_state.references:
            waveform = None
            if entry.has_audio:
                sample_rate = entry.sample_rate
                if sample_rate is None:
                    sample_rate = components.audio_sampling_rate
                waveform = self._normalize_audio_condition(
                    entry.audio, sample_rate, components.audio_sampling_rate, max_samples
                )

            if entry.kind == "image":
                # Images are encoded at high detail — the tier's reference short edge, upscaling included and with
                # *no* area cap — unlike video references and the target itself, which share the tier's canvas rule.
                #
                # Any accepted layout onto a PIL image first, through the processor's own conversions: a
                # `torch.Tensor` is channels-first, as everywhere else in diffusers, and a `np.ndarray`
                # channels-last. Both carry floating point over `[0, 1]`, which is what `numpy_to_pil` scales
                # back up, so `uint8` is normalized before it gets there. Everything below is the geometry of a
                # decoded image, and `image_processor.resize` interpolates an array with `F.interpolate` rather
                # than the LANCZOS the released model was conditioned on, so nothing may reach it as one.
                image = entry.image
                if isinstance(image, torch.Tensor):
                    if image.dtype == torch.uint8:
                        image = image.float() / 255.0
                    image = components.image_processor.pt_to_numpy(image[None])[0]
                if isinstance(image, np.ndarray):
                    if image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError(
                            f"A reference image must be `(height, width, 3)` RGB pixels, got {tuple(image.shape)}."
                        )
                    if image.dtype == np.uint8:
                        image = image.astype(np.float32) / 255.0
                    image = components.image_processor.numpy_to_pil(image)[0]
                if image.size[0] <= 0 or image.size[1] <= 0:
                    raise ValueError(f"A reference image must have a positive size, got {image.size}.")
                width, height = image.size
                if width > 4 * height or height > 4 * width:
                    raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
                scale = tier.reference_short_edge / min(width, height)
                target_height = max(multiple, round(height * scale / multiple) * multiple)
                target_width = max(multiple, round(width * scale / multiple) * multiple)
                if image.size != (target_width, target_height):
                    image = components.image_processor.resize(image, height=target_height, width=target_width)
                normalized.append(PikaH3FlexImageReference(image=image))
            elif entry.kind == "video":
                frames = self._normalize_video_condition(
                    entry.frames,
                    float(entry.fps),
                    block_state.num_frames,
                    multiple,
                    tier.short_edge,
                    tier.max_pixels,
                    float(components.fps),
                )
                # Short of a whole VAE chunk after the first, the reference encodes to a single latent frame.
                min_frames = components.vae_frames_per_chunk + components.vae_latents_per_chunk
                if frames.shape[0] < min_frames:
                    raise ValueError(
                        f"A video reference must hold at least {min_frames} frames at {components.fps} fps once "
                        f"resampled, got {frames.shape[0]}."
                    )
                normalized.append(
                    PikaH3FlexVideoReference(
                        frames=frames,
                        fps=float(components.fps),
                        audio=waveform,
                        sample_rate=None if waveform is None else components.audio_sampling_rate,
                    )
                )
            else:
                normalized.append(PikaH3FlexAudioReference(audio=waveform, sample_rate=components.audio_sampling_rate))
        block_state.normalized_references = normalized

        self.set_block_state(state, block_state)
        return components, state
