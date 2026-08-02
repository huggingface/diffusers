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

import PIL
import torch
from PIL import Image, ImageOps

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline
from .packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    align_num_frames,
    audio_latent_num_frames,
    prepare_keyframe_image,
    resolve_canvas_size,
    video_latent_num_frames,
)
from .packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    reference_kind,
    reference_media_to_uint8,
    resample_reference_frames,
    resolve_reference_image_size,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _latent_geometry(components, height: int, width: int, num_frames: int) -> tuple[int, int, int, int]:
    r"""The latent geometry the packed layout, the noise draws and the decoders all key off."""
    ratio = components.vae_spatial_compression_ratio
    return video_latent_num_frames(num_frames), height // ratio, width // ratio, audio_latent_num_frames(num_frames)


def _latent_geometry_outputs() -> list[OutputParam]:
    r"""The declaration of what [`_latent_geometry`] resolves, shared by the two setup blocks."""
    return [
        OutputParam("num_latent_frames", type_hint=int, description="Number of generated video latent frames."),
        OutputParam("latent_height", type_hint=int, description="Height of the generated video latents."),
        OutputParam("latent_width", type_hint=int, description="Width of the generated video latents."),
        OutputParam("num_audio_latents", type_hint=int, description="Number of generated audio latents per channel."),
    ]


class MiniMaxH3SetupStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Resolves the plan shared by the `t2va` and `fl2va` tasks: the canvas (MiniMax-H3's own 768-short-edge "
            "geometry for the aspect ratio of the first keyframe, or 16:9 without keyframes), the `17 * n + 5` frame "
            "count the video VAE can decode, the latent geometry every later block keys off, and the keyframes put "
            "onto that canvas."
        )

    @staticmethod
    def _check_inputs(block_state) -> None:
        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if block_state.height is not None and (
            block_state.height % MINIMAX_H3_CANVAS_MULTIPLE or block_state.width % MINIMAX_H3_CANVAS_MULTIPLE
        ):
            raise ValueError(
                f"`height` and `width` must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, got "
                f"{block_state.height}x{block_state.width}."
            )
        # The duration the request generates is the one of the *aligned* frame count, so that is what the ceiling has
        # to hold for: 346 frames would otherwise pass the check and then be rounded up to 362, i.e. 15.083 seconds.
        aligned_num_frames = align_num_frames(block_state.num_frames)
        duration = aligned_num_frames / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates between {MINIMAX_H3_MIN_DURATION} and {MINIMAX_H3_MAX_DURATION} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS)} and "
                f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)}, got {block_state.num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="image",
                type_hint=PIL.Image.Image,
                description=(
                    "Keyframe the video starts from. It is *stretched* onto the target canvas, which by default is "
                    "derived from its own aspect ratio."
                ),
            ),
            InputParam(
                name="last_image",
                type_hint=PIL.Image.Image,
                description=(
                    "Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. Combined with "
                    "`image` it is the follower of the two and is cover-cropped onto the canvas."
                ),
            ),
            InputParam.template("height", description="Height of the generated video in pixels, a multiple of 32."),
            InputParam.template("width", description="Width of the generated video in pixels, a multiple of 32."),
            InputParam(
                name="num_frames",
                type_hint=int,
                default=124,
                description=(
                    "Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video "
                    "VAE can decode; the resulting duration must stay between 5 and 15 seconds."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5."),
            *_latent_geometry_outputs(),
            OutputParam(
                "keyframes",
                type_hint=list,
                description="The keyframes put onto the target canvas, in packed order (empty for `t2va`).",
            ),
            OutputParam(
                "keyframe_anchors",
                type_hint=tuple,
                description="Which end of the video every keyframe is anchored to, in packed order.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self._check_inputs(block_state)

        keyframes = [
            ImageOps.exif_transpose(keyframe).convert("RGB")
            for keyframe in (block_state.image, block_state.last_image)
            if keyframe is not None
        ]
        block_state.keyframe_anchors = tuple(
            anchor
            for anchor, keyframe in (("first", block_state.image), ("last", block_state.last_image))
            if keyframe is not None
        )
        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(*(keyframes[0].size if keyframes else (16, 9)))

        aligned_num_frames = align_num_frames(block_state.num_frames)
        if aligned_num_frames != block_state.num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {block_state.num_frames} "
                f"up to {aligned_num_frames}."
            )
            block_state.num_frames = aligned_num_frames

        (
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
        ) = _latent_geometry(components, block_state.height, block_state.width, block_state.num_frames)

        block_state.keyframes = [
            prepare_keyframe_image(keyframe, block_state.height, block_state.width, stretch=index == 0)
            for index, keyframe in enumerate(keyframes)
        ]
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VASetupStep(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    @property
    def description(self) -> str:
        return (
            "Resolves the `ref2va` plan: the canvas (MiniMax-H3's own 16:9 unless asked otherwise — references never "
            "bind the generated geometry), the references prepared at their own resolutions, the frame count they "
            "imply when it was left open, and the latent geometry every later block keys off."
        )

    @staticmethod
    def _check_inputs(components, block_state) -> None:
        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if block_state.height is not None and (
            block_state.height % MINIMAX_H3_CANVAS_MULTIPLE or block_state.width % MINIMAX_H3_CANVAS_MULTIPLE
        ):
            raise ValueError(
                f"`height` and `width` must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, got "
                f"{block_state.height}x{block_state.width}."
            )
        # The duration the request generates is the one of the *aligned* frame count, so that is what the ceiling has
        # to hold for: 346 frames would otherwise pass the check and then be rounded up to 362, i.e. 15.083 seconds.
        aligned_num_frames = None if block_state.num_frames is None else align_num_frames(block_state.num_frames)
        duration = None if aligned_num_frames is None else aligned_num_frames / MINIMAX_H3_FPS
        if duration is not None and not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates between {MINIMAX_H3_MIN_DURATION} and {MINIMAX_H3_MAX_DURATION} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS)} and "
                f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)}, got {block_state.num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )

        if not block_state.references:
            raise ValueError(
                "`ref2va` needs at least one reference; use `MiniMaxH3ModularPipeline` for text-only requests."
            )
        kinds = [reference_kind(index, entry) for index, entry in enumerate(block_state.references)]
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

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="references",
                type_hint=list[MiniMaxH3Reference],
                required=True,
                description=(
                    "The references to condition on, **in the order the model should read them**: the order labels "
                    "them in the prompt presentation and lays them out on the shared rotary clock, so a different "
                    "order is a different request. Every [`MiniMaxH3Reference`] carries exactly one medium, a path or "
                    "in-memory media — `image` (at most 9), `video` at its own `fps` (at most 3, whose `audio` "
                    "soundtrack is conditioned on as well), or `audio` at its own `sample_rate` (at most 3) — for at "
                    "most 12 references in total, and audio references cannot be the only ones. A path is decoded "
                    "when the reference is built, so these blocks only ever see pixels and samples."
                ),
            ),
            InputParam.template("height", description="Height of the generated video in pixels, a multiple of 32."),
            InputParam.template("width", description="Width of the generated video in pixels, a multiple of 32."),
            InputParam(
                name="num_frames",
                type_hint=int,
                description=(
                    "Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video "
                    "VAE can decode. May be left out, but only when exactly one reference carries audio, in which "
                    "case the duration is that soundtrack's."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5."),
            *_latent_geometry_outputs(),
            OutputParam(
                "prepared_references",
                type_hint=list[MiniMaxH3PreparedReference],
                description="The references prepared at their own resolutions, in packed order.",
            ),
        ]

    @staticmethod
    def prepare_references(
        components, references: list[MiniMaxH3Reference], num_frames: int | None
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
                The `references` input of a [`MiniMaxH3Ref2VABlocks`] request.
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
            sample_rate = references[index].sample_rate or components.audio_sampling_rate
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
                    entry.sample_rate or components.audio_sampling_rate,
                    components.audio_sampling_rate,
                    max_duration=num_frames / MINIMAX_H3_FPS,
                )
        return resolved, num_frames

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self._check_inputs(components, block_state)

        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(16, 9)

        requested_num_frames = block_state.num_frames
        block_state.prepared_references, block_state.num_frames = self.prepare_references(
            components, block_state.references, block_state.num_frames
        )
        if requested_num_frames is not None and requested_num_frames != block_state.num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {requested_num_frames} up "
                f"to {block_state.num_frames}."
            )

        (
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            block_state.num_audio_latents,
        ) = _latent_geometry(components, block_state.height, block_state.width, block_state.num_frames)

        self.set_block_state(state, block_state)
        return components, state
