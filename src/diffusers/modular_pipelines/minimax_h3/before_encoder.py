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

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline
from .packing import (
    align_num_frames,
    resolve_canvas_size,
)
from .packing_ref2va import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    reference_media_to_uint8,
    resample_reference_frames,
    resolve_reference_image_size,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3ResizeStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Puts the `fl2va` keyframes onto the target canvas — MiniMax-H3's own 768-short-edge geometry for the "
            "aspect ratio of the first keyframe unless `height` and `width` say otherwise. The canvas resolved here "
            "is the one the whole request generates at."
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
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        keyframes = [keyframe for keyframe in (block_state.image, block_state.last_image) if keyframe is not None]
        block_state.keyframe_anchors = tuple(
            anchor
            for anchor, keyframe in (("first", block_state.image), ("last", block_state.last_image))
            if keyframe is not None
        )
        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(
                *keyframes[0].size, components.canvas_multiple
            )

        prepared = []
        for index, keyframe in enumerate(keyframes):
            if keyframe.size == (block_state.width, block_state.height):
                prepared.append(keyframe)
            elif index == 0:
                # The geometry anchor is stretched onto the canvas. `resize_mode="default"` is exactly PIL's
                # `resize((width, height), LANCZOS)`, verified pixel-identical across aspect ratios.
                prepared.append(
                    components.image_processor.resize(keyframe, height=block_state.height, width=block_state.width)
                )
            else:
                # The follower is cover-cropped. `VaeImageProcessor`'s `resize_mode="crop"` is *not* a drop-in here:
                # it sizes with floor division and centres with `w // 2 - src_w // 2`, where MiniMax-H3 rounds and
                # centres with `(src_w - w) // 2`. The two agree on some aspect ratios and differ by a pixel on
                # others (106 of 218 sampled), which would move the conditioning latents off the reference
                # implementation, so the released model's arithmetic is kept.
                scale = max(block_state.width / keyframe.size[0], block_state.height / keyframe.size[1])
                resized_size = (
                    max(block_state.width, round(keyframe.size[0] * scale)),
                    max(block_state.height, round(keyframe.size[1] * scale)),
                )
                left = max(0, (resized_size[0] - block_state.width) // 2)
                top = max(0, (resized_size[1] - block_state.height) // 2)
                resized = keyframe.resize(resized_size, Image.Resampling.LANCZOS)
                prepared.append(resized.crop((left, top, left + block_state.width, top + block_state.height)))
        block_state.keyframes = prepared

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VASetupStep(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    def __init__(
        self, max_images: int = 9, max_videos: int = 3, max_audios: int = 3, max_references: int = 12
    ):
        r"""
        Resolve a `ref2va` request's plan.

        Args:
            max_images (`int`, defaults to 9): Image references a request may carry.
            max_videos (`int`, defaults to 3): Video references a request may carry.
            max_audios (`int`, defaults to 3): Audio references a request may carry.
            max_references (`int`, defaults to 12): References of any modality a request may carry in total.

        The four limits are what MiniMax-H3 documents for the released checkpoint; they bound nothing but this
        block's own validation, so a fine-tune that packs more can raise them.
        """
        self.max_images = max_images
        self.max_videos = max_videos
        self.max_audios = max_audios
        self.max_references = max_references
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Resolves the `ref2va` plan: the canvas (MiniMax-H3's own 16:9 unless asked otherwise — references never "
            "bind the generated geometry), the references prepared at their own resolutions, the frame count they "
            "imply when it was left open, and the latent geometry every later block keys off."
        )

    def _check_inputs(self, components, block_state) -> None:
        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        multiple = components.canvas_multiple
        if block_state.height is not None and (block_state.height % multiple or block_state.width % multiple):
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got "
                f"{block_state.height}x{block_state.width}."
            )
        # The duration the request generates is the one of the *aligned* frame count, so that is what the ceiling has
        # to hold for: 346 frames would otherwise pass the check and then be rounded up to 362, i.e. 15.083 seconds.
        aligned_num_frames = (
            None
            if block_state.num_frames is None
            else align_num_frames(
                block_state.num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk
            )
        )
        duration = None if aligned_num_frames is None else aligned_num_frames / components.fps
        if duration is not None and not components.min_duration <= duration <= components.max_duration:
            raise ValueError(
                f"MiniMax-H3 generates between {components.min_duration} and {components.max_duration} seconds at "
                f"{components.fps} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(components.min_duration * components.fps)} and "
                f"{int(components.max_duration * components.fps)}, got {block_state.num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )

        if not block_state.references:
            raise ValueError(
                "`ref2va` needs at least one reference; use `MiniMaxH3ModularPipeline` for text-only requests."
            )
        for index, entry in enumerate(block_state.references):
            if not isinstance(entry, MiniMaxH3Reference):
                raise ValueError(
                    f"`references[{index}]` must be a [`MiniMaxH3ImageReference`], [`MiniMaxH3VideoReference`] or "
                    f"[`MiniMaxH3AudioReference`], got {type(entry)}. MiniMax-H3 blocks never open media files, so a "
                    "request that holds paths decodes them first — with "
                    "[`~modular_pipelines.minimax_h3.decode_reference_video`] and "
                    "[`~modular_pipelines.minimax_h3.decode_reference_audio`], or by putting a "
                    "[`MiniMaxH3Ref2VALoadReferencesStep`] in front of these blocks."
                )
        kinds = [entry.kind for entry in block_state.references]
        for kind, limit in (
            ("image", self.max_images),
            ("video", self.max_videos),
            ("audio", self.max_audios),
        ):
            if kinds.count(kind) > limit:
                raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
        if len(kinds) > self.max_references:
            raise ValueError(
                f"MiniMax-H3 accepts at most {self.max_references} references in total, got {len(kinds)}."
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
                    "order is a different request. One dataclass per modality, all holding in-memory media — a "
                    "[`MiniMaxH3ImageReference`] (at most 9), a [`MiniMaxH3VideoReference`] at its own `fps` (at most "
                    "3, whose `audio` soundtrack is conditioned on as well), or a [`MiniMaxH3AudioReference`] at its "
                    "own `sample_rate` (at most 3) — for at most 12 references in total, and audio references cannot "
                    "be the only ones. These blocks never open a media file: decode with "
                    "[`~modular_pipelines.minimax_h3.decode_reference_video`] and "
                    "[`~modular_pipelines.minimax_h3.decode_reference_audio`], which bring the rates along, or put a "
                    "[`MiniMaxH3Ref2VALoadReferencesStep`] in front of these blocks."
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
            OutputParam(
                "prepared_references",
                type_hint=list[MiniMaxH3Reference],
                description=(
                    "The references normalized onto MiniMax-H3's own rates and resolutions, in packed order: the "
                    "same public reference types the request passed in, with an image resized to its own 2048 pixel "
                    "short edge, a video resampled onto 24 fps and onto the canvas its own aspect ratio resolves "
                    "to, and a soundtrack put on the audio VAE's sample rate and truncated to the generated "
                    "duration."
                ),
            ),
        ]

    @staticmethod
    def prepare_references(
        components, references: list[MiniMaxH3Reference], num_frames: int | None
    ) -> tuple[list[MiniMaxH3Reference], int]:
        r"""
        Resolve the references and, if it was left open, the duration they imply.

        Every reference is prepared at its own resolution: an image is resized to a 2048 pixel short edge, a video is
        resampled onto MiniMax-H3's own 24 fps, rescaled onto the 768 pixel canvas of *its own* aspect ratio and
        truncated to the generated frame count, and a soundtrack is put on the audio VAE's sample rate and truncated to
        the generated duration. None of this touches the target canvas.

        A reference that left its `fps` or its `sample_rate` out is taken to already be at MiniMax-H3's own rate, and
        its frames or its samples then flow through untouched. Decoding a file with
        [`~modular_pipelines.minimax_h3.decode_reference_video`] or
        [`~modular_pipelines.minimax_h3.decode_reference_audio`] fills both in from the container.

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
            `tuple[list[MiniMaxH3Reference], int]`: the references normalized onto MiniMax-H3's own rates and
            resolutions, in packed order and of the same public types they came in as, and the frame count.
        """
        # The duration may be left open, but then exactly one reference may carry audio, or the request is ambiguous.
        if num_frames is None:
            audio_bearing = [index for index, reference in enumerate(references) if reference.has_audio]
            if len(audio_bearing) != 1:
                raise ValueError(
                    "`num_frames` may only be left to the references when exactly one of them carries audio, got "
                    f"{len(audio_bearing)}."
                )
            index = audio_bearing[0]
            sample_rate = references[index].sample_rate
            if sample_rate is None:
                sample_rate = components.audio_sampling_rate
            duration = references[index].audio.shape[-1] / sample_rate
            if not components.min_duration <= duration <= components.max_duration:
                raise ValueError(
                    f"`references[{index}]` is {duration:g} seconds long, outside the "
                    f"{components.min_duration} to {components.max_duration} seconds MiniMax-H3 generates."
                )
            num_frames = align_num_frames(
                round(duration * components.fps), components.vae_frames_per_chunk, components.vae_latents_per_chunk
            )
            # The duration the request generates is the one of the *aligned* frame count, so that is what the
            # ceiling has to hold for: a 14.99 second soundtrack rounds up to 362 frames, i.e. 15.083 seconds.
            if num_frames / components.fps > components.max_duration:
                raise ValueError(
                    f"`references[{index}]` is {duration:g} seconds long, which rounds up to {num_frames} frames "
                    f"(`17 * n + 5`), i.e. {num_frames / components.fps:g} seconds — past the "
                    f"{components.max_duration} seconds MiniMax-H3 generates. Pass `num_frames` to generate a "
                    "shorter video from this soundtrack."
                )
        num_frames = align_num_frames(num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk)

        prepared = []
        for entry in references:
            waveform = None
            if entry.has_audio:
                sample_rate = entry.sample_rate
                if sample_rate is None:
                    sample_rate = components.audio_sampling_rate
                waveform = prepare_reference_waveform(
                    entry.audio,
                    sample_rate,
                    components.audio_sampling_rate,
                    max_duration=num_frames / components.fps,
                )

            if entry.kind == "image":
                image = ImageOps.exif_transpose(entry.image).convert("RGB")
                height, width = resolve_reference_image_size(*image.size, components.canvas_multiple)
                prepared.append(MiniMaxH3ImageReference(image=prepare_reference_image(image, height, width)))
            elif entry.kind == "video":
                frames = resample_reference_frames(reference_media_to_uint8(entry.frames), float(entry.fps))
                prepared.append(
                    MiniMaxH3VideoReference(
                        frames=prepare_reference_frames(frames, num_frames, components.canvas_multiple),
                        fps=float(components.fps),
                        audio=waveform,
                        sample_rate=None if waveform is None else components.audio_sampling_rate,
                    )
                )
            else:
                prepared.append(
                    MiniMaxH3AudioReference(audio=waveform, sample_rate=components.audio_sampling_rate)
                )
        return prepared, num_frames

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self._check_inputs(components, block_state)

        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(16, 9, components.canvas_multiple)

        requested_num_frames = block_state.num_frames
        block_state.prepared_references, block_state.num_frames = self.prepare_references(
            components, block_state.references, block_state.num_frames
        )
        if requested_num_frames is not None and requested_num_frames != block_state.num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {requested_num_frames} up "
                f"to {block_state.num_frames}."
            )

        self.set_block_state(state, block_state)
        return components, state
