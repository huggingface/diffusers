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

from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Per-row modality tags. They index the transformer's AdaLN table, so the values are a checkpoint contract.
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

# MiniMax-H3 generates at a fixed 24 fps. The canvas it generates on is configurable rather than a constant here:
# the short edge and the area cap are the `canvas_short_edge` / `canvas_max_pixels` configs of the blocks that
# resolve a canvas, and the multiple both axes round to follows from the VAE and the transformer, so it is
# `MiniMaxH3ModularPipeline.canvas_multiple` below. All three reach these helpers as arguments.
MINIMAX_H3_FPS = 24
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second. Stereo is carried as two channel-major
# blocks of audio rows (and as two batch items at the audio VAE boundary, which is mono).
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int,
    short_edge: int,
    max_pixels: int,
    min_aspect_ratio: float = MINIMAX_H3_MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MINIMAX_H3_MAX_ASPECT_RATIO,
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


def audio_latent_num_frames(
    num_frames: int, fps: float = MINIMAX_H3_FPS, latents_per_second: int = MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
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


class MiniMaxH3ModularPipeline(ModularPipeline):
    """
    A ModularPipeline for joint video + audio generation with MiniMax-H3: the `t2va` (text only) and `fl2va` (first
    and/or last keyframe) workflows against the `transformer/` checkpoint partition, and the `ref2va` (omni-reference)
    workflow against `transformer_ref/`. One repository holds both partitions, and selecting a workflow loads only its
    own:

    ```py
    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", workflow="ref2va")
    ```

    MiniMax-H3 denoises **one packed sequence** that holds the text conditioning, the keyframe conditioning latents,
    the audio latents and the video latents at once, which is why the blocks pass a row layout around rather than
    per-modality tensors, and why the pipeline carries two schedulers (`shift = 12.0` for video, `shift = 3.0` for
    audio) that are stepped inside a single transformer call.

    The checkpoint is guidance-distilled: guidance is baked into the weights, so there is no guider, no
    `negative_prompt` and no `guidance_scale`, and every step runs exactly one forward pass.

    MiniMax-H3 is modular only: this pipeline and its blocks are the whole integration, there is no `DiffusionPipeline`
    half. This module carries the model facts every block keys off — the config-derived geometry as properties, the
    canvas and frame-count arithmetic as functions — and the conditioning, encoding and noise contracts live on the
    blocks themselves.

    ```py
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
    pipe.load_components(dtype=torch.bfloat16)
    ```

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "MiniMaxH3Blocks"

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
        return MINIMAX_H3_FPS

    @property
    def min_duration(self):
        r"""Shortest video MiniMax-H3 generates, in seconds."""
        return 5.0

    @property
    def max_duration(self):
        r"""Longest video MiniMax-H3 generates, in seconds."""
        return 15.0

    @property
    def audio_channels(self):
        r"""Channels of the generated soundtrack: MiniMax-H3 is stereo, packed channel-major."""
        return MINIMAX_H3_AUDIO_CHANNELS

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
        return MINIMAX_H3_TEXT_TAG

    @property
    def video_tag(self):
        r"""The modality tag of a video row of the packed sequence, which a vision block's rows also carry."""
        return MINIMAX_H3_VIDEO_TAG

    @property
    def audio_tag(self):
        r"""The modality tag of an audio row of the packed sequence."""
        return MINIMAX_H3_AUDIO_TAG
