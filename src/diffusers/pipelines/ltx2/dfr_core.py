# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

"""Shared DFR constants and helpers used by the LTX-2.5 DFR pipelines."""

import torch


# Keyframes carried between temporal rounds are pinned just short of fully clean so a tile can still settle its seam
# frame.
ANCHOR_KEYFRAME_STRENGTH = 0.95

# Ancestral noise fraction used by the temporal refine rounds. Their short schedule densifies detail rather than
# building structure, so a partly stochastic step is what fills in the freshly interpolated frames.
TEMPORAL_ANCESTRAL_ETA = 0.5

# RoPE time is `pixel_frame / fps`. The transformer is trained around 24/25/30 and 60 fps, not 48 or 120. A 120 fps
# time base halves every token's temporal span and the model can no longer lay out the VAE's pixel frames inside one
# latent -- it decodes as a motion spike at each latent border followed by a stall. 48 fps stretches the same span the
# other way. Condition at 60 in both cases and treat the decoded frames at the playback rate (120 fps: generate 2x
# frames at 60, mux as 120; 48 fps: generate the 24x2 canvas at 60, mux as 48). Playback fps is used for the returned
# frame count and the audio trim only.
MAX_CONDITIONING_FPS = 60.0
SNAP_CONDITIONING_FPS_ABOVE = 30.0


def _conditioning_fps(playback_fps: float) -> float:
    """Fps the transformer sees. Playback may be 48, 96, 120, ...; those rates only affect muxing."""
    return MAX_CONDITIONING_FPS if playback_fps > SNAP_CONDITIONING_FPS_ABOVE else playback_fps


# The epilogue's keyframes arrive as finished frames, rebuilt at the output resolution, so they are pinned fully clean.
EPILOGUE_KEYFRAME_STRENGTH = 1.0


def _audio_window_for_tile(
    audio_latents: torch.Tensor,
    pixel_start: int,
    tile_frames: int,
    playback_fps: float,
    source_seconds: float,
    conditioning_fps: float,
    audio_latents_per_second: float,
) -> torch.Tensor:
    """
    Cut the frozen stage-1 audio to one temporal tile's window and resample it to that tile's token count.

    The window is wall clock: `pixel_start / playback_fps` through `(pixel_start + tile_frames) / playback_fps`, as a
    fraction of `source_seconds`. Taking a fraction of the *canvas* instead would drift, because a refine round maps `N
    -> 2 (N - 1) + 1` while the frame rate doubles, so each round's canvas is a hair shorter than twice the last one
    and the tail tiles would pull audio from past their own playback. `conditioning_fps` only sizes the output token
    count, matching what the video side asks the transformer for.

    Returns the packed `(batch_size, tile_audio_frames, channels * mel_bins)` window and its frame count.
    """
    source_frames = audio_latents.shape[1]
    tile_audio_frames = round(tile_frames / conditioning_fps * audio_latents_per_second)
    start = pixel_start / playback_fps / source_seconds * source_frames
    span = tile_frames / playback_fps / source_seconds * source_frames
    positions = start + (span / tile_audio_frames) * torch.arange(
        tile_audio_frames, device=audio_latents.device, dtype=torch.float32
    )
    positions = positions.clamp(0, source_frames - 1)
    low = positions.floor().long()
    high = (low + 1).clamp(max=source_frames - 1)
    weight = (positions - low).to(audio_latents.dtype).view(1, -1, 1)
    return audio_latents[:, low] * (1 - weight) + audio_latents[:, high] * weight


def trim_canvas(latents: torch.Tensor, num_frames: int, temporal_compression_ratio: int) -> torch.Tensor:
    """Drop the padded tail of a DFR canvas so it matches `num_frames` pixel frames.

    The canvas is padded to a whole number of keyframe segments; this keeps the tokens that cover `num_frames` and is
    the last step before VAE decode. `output_type="latent"` returns the padded grid so a slot on the pad (e.g. pixel 96
    on an 81→97 canvas) is not dropped.
    """
    keep = (num_frames - 1) // temporal_compression_ratio + 1
    return latents[:, :, :keep]
