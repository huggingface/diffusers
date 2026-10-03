# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Optional action HUD postprocessing for Echo-WM videos."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ...utils import is_av_available
from .action import _parse_action


DEFAULT_IMAGE_CRF = 33
LTX2_5_IMAGE_CRF = 18

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


# Copied from diffusers.pipelines.ltx2.utils.resolve_default_image_crf
def resolve_default_image_crf(text_encoder: Any) -> int:
    """Return the image-conditioning H.264 CRF that matches the loaded text-encoder generation.

    LTX-2.5 uses a Gemma 4 (`gemma4_unified` / `gemma4`) text encoder and was trained with CRF 18; earlier generations
    use Gemma 3 and CRF 33. Mirrors `detect_params(...).default_image_crf` in ltx-pipelines, which keys off checkpoint
    `model_version`.
    """
    model_type = getattr(getattr(text_encoder, "config", None), "model_type", None)
    if model_type in ("gemma4_unified", "gemma4"):
        return LTX2_5_IMAGE_CRF
    return DEFAULT_IMAGE_CRF


# Copied from diffusers.pipelines.ltx2.utils.apply_image_conditioning_crf
def apply_image_conditioning_crf(image: np.ndarray, crf: int) -> np.ndarray:
    """Re-compress a single RGB image at ``crf`` so conditioning matches training compression.

    Port of ltx-pipelines `media_io.preprocess`. ``crf=0`` skips re-compression. ``crf`` must be resolved before
    calling (never ``None``).
    """
    if crf is None:
        raise ValueError(
            "Image conditioning CRF is unresolved (crf=None). Resolve it first via "
            "`resolve_default_image_crf(text_encoder)` or pass an explicit `crf` on the condition."
        )
    if crf == 0:
        return image
    if not is_av_available():
        raise ImportError(
            "PyAV is required to apply image-conditioning H.264 CRF re-compression. "
            "Install it with `pip install av`, or pass `crf=0` to skip re-compression."
        )
    import av

    if image.dtype != np.uint8:
        raise ValueError(
            f"Image conditioning CRF expects a uint8 RGB array, got dtype={image.dtype}. "
            "Pass a PIL image / uint8 array, or set `crf=0` to skip re-compression."
        )

    with BytesIO() as output_file:
        container = av.open(output_file, "w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
            # Round to nearest multiple of 2 for compatibility with video codecs.
            height = image.shape[0] // 2 * 2
            width = image.shape[1] // 2 * 2
            image = image[:height, :width]
            stream.height = height
            stream.width = width
            av_frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
        finally:
            container.close()
        video_bytes = output_file.getvalue()

    with BytesIO(video_bytes) as video_file:
        container = av.open(video_file)
        try:
            stream = next(s for s in container.streams if s.type == "video")
            frame = next(container.decode(stream))
        finally:
            container.close()
        return frame.to_ndarray(format="rgb24")


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _expand_action(action: str, num_frames: int) -> list[frozenset[str]]:
    frame_actions = []
    for keys, duration in _parse_action(action):
        frame_actions.extend([frozenset(keys)] * duration)

    if len(frame_actions) < num_frames:
        frame_actions.extend([frame_actions[-1]] * (num_frames - len(frame_actions)))
    return frame_actions[:num_frames]


class _ActionOverlayRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.key_size = max(32, int(height * 0.08))
        self.key_gap = max(4, int(self.key_size * 0.15))
        self.key_radius = max(4, int(self.key_size * 0.2))
        self.font = _load_font(int(self.key_size * 0.5))
        self.key_tiles = self._build_key_tiles()

    def _build_key_tiles(self) -> dict[tuple[str, bool], Image.Image]:
        tiles = {}
        for key in "WASD":
            for pressed in (False, True):
                fill = (255, 255, 255, 200) if pressed else (0, 0, 0, 100)
                outline = (255, 255, 255, 255) if pressed else (255, 255, 255, 60)
                text_color = (0, 0, 0, 220) if pressed else (255, 255, 255, 180)
                tile = Image.new("RGBA", (self.key_size, self.key_size), (0, 0, 0, 0))
                draw = ImageDraw.Draw(tile)
                draw.rounded_rectangle(
                    [0, 0, self.key_size - 1, self.key_size - 1],
                    radius=self.key_radius,
                    fill=fill,
                    outline=outline,
                    width=max(1, int(self.key_size * 0.03)),
                )
                box = draw.textbbox((0, 0), key, font=self.font)
                text_width, text_height = box[2] - box[0], box[3] - box[1]
                draw.text(
                    ((self.key_size - text_width) / 2, (self.key_size - text_height) / 2 - 2),
                    key,
                    fill=text_color,
                    font=self.font,
                )
                tiles[(key, pressed)] = tile
        return tiles

    @staticmethod
    def _draw_arrow(draw: ImageDraw.ImageDraw, cx: int, cy: int, direction: str, active: bool, size: int) -> None:
        if direction == "right":
            points = [(cx - size * 0.55, cy - size), (cx + size * 0.65, cy), (cx - size * 0.55, cy + size)]
        elif direction == "left":
            points = [(cx + size * 0.55, cy - size), (cx - size * 0.65, cy), (cx + size * 0.55, cy + size)]
        elif direction == "up":
            points = [(cx - size, cy + size * 0.55), (cx, cy - size * 0.65), (cx + size, cy + size * 0.55)]
        else:
            points = [(cx - size, cy - size * 0.55), (cx, cy + size * 0.65), (cx + size, cy - size * 0.55)]
        draw.polygon(points, fill=(255, 255, 255, 210) if active else (255, 255, 255, 72))

    def _draw_joystick(self, canvas: Image.Image, cx: int, cy: int, keys: frozenset[str]) -> None:
        yaw = float("l" in keys) - float("j" in keys)
        pitch = float("i" in keys) - float("k" in keys)
        radius = max(30, int((self.key_size * 2 + self.key_gap) * 0.47))

        shadow = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).ellipse(
            [cx - radius - 14, cy - radius - 14, cx + radius + 14, cy + radius + 14], fill=(0, 0, 0, 88)
        )
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(max(8, int(radius * 0.16)))))

        draw = ImageDraw.Draw(canvas)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(7, 9, 13, 104),
            outline=(255, 255, 255, 95),
            width=max(1, int(radius * 0.035)),
        )
        draw.line([cx - radius * 0.63, cy, cx + radius * 0.63, cy], fill=(255, 255, 255, 56), width=1)
        draw.line([cx, cy - radius * 0.63, cx, cy + radius * 0.63], fill=(255, 255, 255, 56), width=1)
        marker_offset = int(radius * 0.78)
        marker_size = max(7, int(radius * 0.16))
        self._draw_arrow(draw, cx + marker_offset, cy, "right", "l" in keys, marker_size)
        self._draw_arrow(draw, cx - marker_offset, cy, "left", "j" in keys, marker_size)
        self._draw_arrow(draw, cx, cy - marker_offset, "up", "i" in keys, marker_size)
        self._draw_arrow(draw, cx, cy + marker_offset, "down", "k" in keys, marker_size)

        knob_x = int(cx + yaw * radius * 0.48)
        knob_y = int(cy - pitch * radius * 0.48)
        draw.line([cx, cy, knob_x, knob_y], fill=(255, 255, 255, 120), width=max(1, int(radius * 0.025)))
        knob_radius = max(7, int(radius * 0.13))
        draw.ellipse(
            [knob_x - knob_radius, knob_y - knob_radius, knob_x + knob_radius, knob_y + knob_radius],
            fill=(255, 255, 255, 230),
            outline=(255, 255, 255, 255),
        )

    def render(self, keys: frozenset[str]) -> Image.Image:
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        margin = int(self.height * 0.05)
        cluster_width = self.key_size * 3 + self.key_gap * 2
        cluster_height = self.key_size * 2 + self.key_gap
        start_x = margin
        start_y = self.height - margin - cluster_height

        positions = {
            "W": (start_x + self.key_size + self.key_gap, start_y),
            "A": (start_x, start_y + self.key_size + self.key_gap),
            "S": (start_x + self.key_size + self.key_gap, start_y + self.key_size + self.key_gap),
            "D": (start_x + 2 * (self.key_size + self.key_gap), start_y + self.key_size + self.key_gap),
        }
        for key, position in positions.items():
            canvas.alpha_composite(self.key_tiles[(key, key.lower() in keys)], dest=position)

        joystick_radius = max(30, int(cluster_height * 0.47))
        joystick_x = start_x + cluster_width + self.key_size + joystick_radius
        self._draw_joystick(canvas, joystick_x, start_y + cluster_height // 2, keys)
        return canvas


def apply_action_overlay(video: Sequence[Image.Image], action: str) -> list[Image.Image]:
    """
    Add a WASD/IJKL HUD to decoded PIL frames without modifying the source frames.

    Args:
        video (`Sequence[PIL.Image.Image]`):
            Decoded video frames, such as one video returned by an Echo-WM pipeline with `output_type="pil"`.
        action (`str`):
            Echo-WM action program in the same comma-separated `<keys>-<frames>` format accepted by the pipeline.

    Returns:
        `list[PIL.Image.Image]`: A new list of RGB frames with the action HUD composited onto them.
    """
    if len(video) == 0:
        raise ValueError("`video` must contain at least one frame.")
    if not all(isinstance(frame, Image.Image) for frame in video):
        raise TypeError("`video` must be a sequence of PIL images. Use the pipeline's default `output_type='pil'`.")

    renderer = _ActionOverlayRenderer(*video[0].size)
    frame_actions = _expand_action(action, len(video))
    output = []
    for frame, keys in zip(video, frame_actions):
        rendered = frame.convert("RGBA")
        rendered.alpha_composite(renderer.render(keys))
        output.append(rendered.convert("RGB"))
    return output
