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

from dataclasses import dataclass

from ...loaders import PikaH3FlexLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Per-row modality tags. They index the transformer's AdaLN table, so the values are a checkpoint contract.
PIKA_H3_FLEX_VIDEO_TAG = 0
PIKA_H3_FLEX_TEXT_TAG = 1
PIKA_H3_FLEX_AUDIO_TAG = 2

# Pika H3 Flex generates at MiniMax-H3's fixed 24 fps. The canvas it generates on is configurable rather than a
# constant here: the short edge and the area cap are the `canvas_short_edge` / `canvas_max_pixels` configs of the
# blocks that resolve a canvas, and the multiple both axes round to follows from the VAE and the transformer, so it is
# `PikaH3FlexModularPipeline.canvas_multiple` below. All three reach these helpers as arguments.
PIKA_H3_FLEX_FPS = 24
PIKA_H3_FLEX_MIN_ASPECT_RATIO = 1 / 4
PIKA_H3_FLEX_MAX_ASPECT_RATIO = 4

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second. Stereo is carried as two channel-major
# blocks of audio rows (and as two batch items at the audio VAE boundary, which is mono).
PIKA_H3_FLEX_AUDIO_LATENTS_PER_SECOND = 40
PIKA_H3_FLEX_AUDIO_CHANNELS = 2


@dataclass(frozen=True)
class PikaH3FlexTierRule:
    r"""
    The geometry rule of one resolution tier of the released bucket table.

    Pika H3 Flex generates on a table of canvases — one tier per short edge, eleven aspect-ratio buckets per tier — and
    conditions its `ref2va` references by the tier the *generated* canvas falls in: a video reference is put on the
    tier's own canvas rule, and an image reference is encoded at the tier's reference short edge, 1024 pixels below the
    768 tier and 2048 from it upwards.

    Attributes:
        short_edge (`int`): The short edge a canvas of this tier aims for.
        max_pixels (`int`): The area budget of a canvas of this tier.
        reference_short_edge (`int`): The short edge an image reference is encoded at for a canvas of this tier.
    """

    short_edge: int
    max_pixels: int
    reference_short_edge: int


# The tiers of the released bucket table, keyed by name. Only the 480p and 768p tiers are exposed by the released
# package; the others follow the same rule and are what a larger canvas resolves to.
PIKA_H3_FLEX_TIER_RULES = {
    "p480": PikaH3FlexTierRule(short_edge=480, max_pixels=832 * 480, reference_short_edge=1024),
    "p544": PikaH3FlexTierRule(short_edge=544, max_pixels=960 * 544, reference_short_edge=1024),
    "p768": PikaH3FlexTierRule(short_edge=768, max_pixels=1344 * 768, reference_short_edge=2048),
    "p1080": PikaH3FlexTierRule(short_edge=1088, max_pixels=1920 * 1088, reference_short_edge=2048),
    "p1440": PikaH3FlexTierRule(short_edge=1440, max_pixels=2560 * 1440, reference_short_edge=2048),
    "p2160": PikaH3FlexTierRule(short_edge=2176, max_pixels=3840 * 2176, reference_short_edge=2048),
}


def resolve_tier_rule(height: int, width: int) -> PikaH3FlexTierRule:
    r"""
    The tier rule a generated canvas falls in: the smallest tier whose short edge is at least the canvas' own, or the
    largest tier for a canvas above every rule. A 832x480 canvas is a `p480` bucket, 1344x768 a `p768` one.

    Args:
        height (`int`): Height of the generated canvas in pixels.
        width (`int`): Width of the generated canvas in pixels.

    Returns:
        [`PikaH3FlexTierRule`]: the rule of the tier.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"The canvas must be positive, got {height}x{width}.")
    short_edge = min(height, width)
    ordered = sorted(PIKA_H3_FLEX_TIER_RULES.values(), key=lambda rule: rule.short_edge)
    for rule in ordered:
        if rule.short_edge >= short_edge:
            return rule
    return ordered[-1]


# Copied from diffusers.modular_pipelines.minimax_h3.modular_pipeline.resolve_canvas_size with MINIMAX_H3->PIKA_H3_FLEX
def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int,
    short_edge: int,
    max_pixels: int,
    min_aspect_ratio: float = PIKA_H3_FLEX_MIN_ASPECT_RATIO,
    max_aspect_ratio: float = PIKA_H3_FLEX_MAX_ASPECT_RATIO,
) -> tuple[int, int]:
    r"""
    Resolve a display aspect ratio into a MiniMax-H3 canvas.

    The short edge starts at `short_edge`, the area is capped at `max_pixels` and both axes are then rounded to the
    nearest `canvas_multiple` — so the final area may end up slightly above the pre-rounding budget. Only the ratio of
    the first two arguments matters; pass either the aspect ratio (`16, 9`) or the source dimensions of a keyframe.

    Args:
        aspect_width (`float`): Width of the target ratio.
        aspect_height (`float`): Height of the target ratio.
        canvas_multiple (`int`):
            What both axes round to, i.e. `components.canvas_multiple` — 32 for the released checkpoint.
        short_edge (`int`):
            The short edge to aim for, i.e. `components.config.canvas_short_edge` — 768 for the released checkpoint.
        max_pixels (`int`):
            The area budget, i.e. `components.config.canvas_max_pixels` — `768 * 1344` for the released checkpoint.
        min_aspect_ratio (`float`, defaults to 1/4), max_aspect_ratio (`float`, defaults to 4):
            The ratios the released checkpoint was trained over, which the requested one has to fall between.

    Returns:
        `tuple[int, int]`: the `(height, width)` of the canvas.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / min_aspect_ratio:g} to {max_aspect_ratio:g}:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio

    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = canvas_multiple
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


# Copied from diffusers.modular_pipelines.minimax_h3.modular_pipeline.align_num_frames
def align_num_frames(num_frames: int, frames_per_chunk: int, latents_per_chunk: int) -> int:
    r"""
    Snap a frame count up to the next `frames_per_chunk * n + latents_per_chunk` the video VAE can encode.

    Args:
        num_frames (`int`): The requested number of frames.
        frames_per_chunk (`int`): Pixel frames the video VAE encodes per chunk, its `clip_length`.
        latents_per_chunk (`int`): Latent frames a chunk keeps, the VAE's `tokens_chunk_size`.

    Returns:
        `int`: The aligned number of frames.
    """
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % frames_per_chunk != latents_per_chunk:
        num_frames += 1
    return num_frames


# Copied from diffusers.modular_pipelines.minimax_h3.modular_pipeline.video_latent_num_frames
def video_latent_num_frames(num_frames: int, frames_per_chunk: int, latents_per_chunk: int) -> int:
    r"""
    The number of latent frames the video VAE produces for a `17 * n + 5` frame count.

    Args:
        num_frames (`int`): An aligned number of frames.

    Returns:
        `int`: The number of latent frames, `5 * n + 2`.
    """
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"`num_frames` must be of the form {frames_per_chunk} * n + {latents_per_chunk}, got {num_frames}."
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


# Copied from diffusers.modular_pipelines.minimax_h3.modular_pipeline.audio_latent_num_frames with MINIMAX_H3->PIKA_H3_FLEX
def audio_latent_num_frames(
    num_frames: int, fps: float = PIKA_H3_FLEX_FPS, latents_per_second: int = PIKA_H3_FLEX_AUDIO_LATENTS_PER_SECOND
) -> int:
    r"""
    The number of audio latents that covers a video of `num_frames` frames.

    Args:
        num_frames (`int`): The number of video frames.
        fps (`float`, defaults to 24): The rate those frames run at, i.e. `components.fps`.
        latents_per_second (`int`, defaults to 40): The audio VAE's latent rate.

    Returns:
        `int`: The number of audio latents, rounded at the latent grid.
    """
    return int(round(num_frames / fps * latents_per_second))


class PikaH3FlexModularPipeline(ModularPipeline, PikaH3FlexLoraLoaderMixin):
    """
    A ModularPipeline for joint video + audio generation with Pika H3 Flex, the post-trained six-evaluation MiniMax-H3:
    the `t2va` (text only) and `fl2va` (first and/or last keyframe) workflows against the `transformer/` checkpoint
    partition with the `fl2va` adapter, and the `ref2va` (omni-reference) workflow against `transformer_ref/` with the
    `ref2va` adapter.

    The base components are MiniMax-H3's, and the Pika H3 Flex release is a pair of rank-128 LoRA adapters on top of
    the two transformer partitions, so a pipeline is assembled in two steps: the components are loaded through a
    `modular_model_index.json` that names the base components and the two schedulers, and the adapter of the selected
    workflow is loaded with [`~loaders.PikaH3FlexLoraLoaderMixin.load_lora_weights`].

    ```py
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex-Early-Access", workflow="fl2va")
    pipe.load_components(dtype=torch.bfloat16)
    pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex-Early-Access", weight_name="fl2va/adapter.safetensors")
    ```

    Like MiniMax-H3, Pika H3 Flex denoises **one packed sequence** that holds the text conditioning, the keyframe or
    reference conditioning latents, the audio latents and the video latents at once, which is why the blocks pass a row
    layout around rather than per-modality tensors, and why the pipeline carries two schedulers (`shift = 12.0` for
    video, `shift = 3.0` for audio) that are stepped inside a single transformer call. Unlike MiniMax-H3, the two
    schedulers are [`PikaH3FlexScheduler`] re-noising ladders over a fixed six-knot schedule: there is no
    `num_inference_steps`, and every step draws fresh noise from the request's generator.

    The adapters are guidance-distilled like their base: there is no guider, no `negative_prompt` and no
    `guidance_scale`, and every step runs exactly one forward pass.

    This module carries the model facts every block keys off — the config-derived geometry as properties, the canvas,
    tier and frame-count arithmetic as functions — and the conditioning, encoding and noise contracts live on the
    blocks themselves.
    """

    default_blocks_name = "PikaH3FlexBlocks"

    @property
    def vae_spatial_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.spatial_compression_ratio
        return 16

    @property
    def vae_latent_channels(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.latent_channels
        return 24

    @property
    def vae_frames_per_chunk(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.clip_length
        return 17

    @property
    def vae_latents_per_chunk(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.tokens_chunk_size
        return 5

    @property
    def audio_sampling_rate(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.sampling_rate
        return 32000

    @property
    def audio_latent_channels(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.latent_channels
        return 32

    @property
    def patch_size(self):
        # One repository holds both checkpoint partitions — `transformer/` for `t2va`/`fl2va`, `transformer_ref/`
        # for `ref2va` — and a workflow loads only its own, so read whichever is present.
        for name in ("transformer", "transformer_ref"):
            if getattr(self, name, None) is not None:
                return tuple(getattr(self, name).config.patch_size)
        return (1, 2, 2)

    @property
    def canvas_multiple(self):
        r"""What the generated height and width have to be a multiple of, 32 for the released checkpoint."""
        # A canvas has to survive the VAE's spatial compression and still be a whole number of patch rows wide, so
        # the multiple is the product of the two.
        return self.vae_spatial_compression_ratio * self.patch_size[2]

    @property
    def fps(self):
        r"""MiniMax-H3's own frame rate. Everything it generates and conditions on is resampled onto it."""
        return PIKA_H3_FLEX_FPS

    @property
    def min_duration(self):
        r"""Shortest video Pika H3 Flex generates, in seconds."""
        return 5.0

    @property
    def max_duration(self):
        r"""Longest video Pika H3 Flex generates, in seconds."""
        return 15.0

    @property
    def audio_channels(self):
        r"""Channels of the generated soundtrack: MiniMax-H3 is stereo, packed channel-major."""
        return PIKA_H3_FLEX_AUDIO_CHANNELS

    @property
    def text_encoder_layer(self):
        r"""
        Which Qwen3-VL hidden state conditions the transformer.

        MiniMax-H3 reads `hidden_states[50]`, not the final one: the last layer is post-norm and is not the
        conditioning the released weights were trained against.
        """
        return 50

    @property
    def pixel_mean(self):
        r"""Per-channel mean the video VAE's input is normalized by, ImageNet's."""
        return (0.485, 0.456, 0.406)

    @property
    def pixel_std(self):
        r"""Per-channel standard deviation the video VAE's input is normalized by, ImageNet's."""
        return (0.229, 0.224, 0.225)

    @property
    def keyframe_encode_seed(self):
        r"""
        Seed the conditioning posterior is sampled under, independently of the request's own generator.

        Fixed at 42 in the reference implementation, so the same keyframe always encodes to the same anchor.
        """
        return 42

    @property
    def keyframe_noise_aug(self):
        r"""
        The `t` a visual conditioning anchor is held at: 0.999, just short of clean.

        The released model was trained with its anchors very slightly noised, so conditioning on exactly `t = 1.0` is
        off-distribution.
        """
        return 0.999

    @property
    def text_tag(self):
        r"""The modality tag of a text row of the packed sequence."""
        return PIKA_H3_FLEX_TEXT_TAG

    @property
    def video_tag(self):
        r"""The modality tag of a video row of the packed sequence, which a vision block's rows also carry."""
        return PIKA_H3_FLEX_VIDEO_TAG

    @property
    def audio_tag(self):
        r"""The modality tag of an audio row of the packed sequence."""
        return PIKA_H3_FLEX_AUDIO_TAG
