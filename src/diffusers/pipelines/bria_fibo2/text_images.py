# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
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

"""
Images of the text a Fibo 2.0 caption asks for, drawn the way Fibo 2.0's training drew them: each text on its own, in
one line, black on white. Pillow's built-in font draws the text. A character it lacks is drawn with the first of the
fallback fonts that has it: training used Aileron, the full version of the built-in font, then DejaVu Sans.
"""

import functools
from typing import Sequence

import PIL.Image
from PIL import ImageDraw, ImageFont

from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# The size the text is drawn at, and the number of characters longer texts are split into lines of
FONT_SIZE = 20
MAX_LINE_CHARS = 256
_PADDING = 2
_LINE_SPACING = 4


@functools.lru_cache(maxsize=None)
def load_font(path: str) -> ImageFont.FreeTypeFont:
    # With the basic layout, as Pillow's built-in font, so the text doesn't set differently where libraqm is installed
    return ImageFont.truetype(path, FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)


def draw_text_image(text: str, fallback_fonts: Sequence[ImageFont.FreeTypeFont] = ()) -> PIL.Image.Image:
    """
    Draws a text the way Fibo 2.0's training drew each `text_render` entry's text.

    The text is drawn in one line, its whitespace collapsed, black on white with 2 pixels of padding. A text longer
    than 256 characters is split into lines of 256. A character Pillow's built-in font lacks is drawn with the first of
    `fallback_fonts` that has it.
    """
    font = _get_default_font()
    line = " ".join(text.split())
    lines = [line[i : i + MAX_LINE_CHARS] for i in range(0, len(line), MAX_LINE_CHARS)] or [""]
    if all(ord(character) < 128 or _has_glyph(font, character) for character in line):
        return _draw_one_font(lines, font)
    if not fallback_fonts:
        logger.warning(
            "A text to draw has characters Pillow's built-in font lacks, which are drawn as it draws them. Pass the"
            " `fonts` Fibo 2.0 was trained with to draw them as training did."
        )
    return _draw_mixed(lines, [font, *fallback_fonts])


@functools.lru_cache(maxsize=None)
def _get_default_font() -> ImageFont.FreeTypeFont:
    try:
        font = ImageFont.load_default(FONT_SIZE)
    except TypeError:  # Pillow before 10.1 can't size its built-in font
        font = None
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise ImportError("Drawing the images of a caption's text needs Pillow 10.1 or later, built with FreeType.")
    return font


@functools.lru_cache(maxsize=4096)
def _has_glyph(font: ImageFont.FreeTypeFont, character: str) -> bool:
    # A font without a glyph for the character draws its "notdef" box, as it does for U+FFFE, which no font has
    try:
        notdef = font.getmask("￾")
        mask = font.getmask(character)
    except Exception:  # as in training, a font that can't say is taken to have the glyph
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def _font_runs(text: str, fonts: Sequence[ImageFont.FreeTypeFont]) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    # The text split into runs of characters drawn with the same font: the first font with a glyph for the character,
    # or the built-in font, the first, when no font has one
    runs = []
    for character in text:
        font = fonts[0]
        if not (ord(character) < 128 or _has_glyph(fonts[0], character)):
            font = next((candidate for candidate in fonts[1:] if _has_glyph(candidate, character)), fonts[0])
        if runs and runs[-1][1] is font:
            runs[-1][0] += character
        else:
            runs.append([character, font])
    return [(run, font) for run, font in runs]


def _draw_one_font(lines: list[str], font: ImageFont.FreeTypeFont) -> PIL.Image.Image:
    # Each line is placed by its bounding box, and is at least as tall as "Ag"
    draw = ImageDraw.Draw(PIL.Image.new("RGB", (1, 1), "white"))
    min_box = draw.textbbox((0, 0), "Ag", font=font)
    min_line_height = max(1, min_box[3] - min_box[1])
    line_sizes = []
    max_width = 1
    total_height = 0
    for line in lines:
        box = draw.textbbox((0, 0), line or " ", font=font)
        line_height = max(min_line_height, box[3] - box[1])
        line_sizes.append((box, line_height))
        max_width = max(max_width, box[2] - box[0])
        total_height += line_height
    total_height += _LINE_SPACING * (len(lines) - 1)

    image = PIL.Image.new("RGB", (max_width + _PADDING * 2, total_height + _PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = _PADDING
    for line, (box, line_height) in zip(lines, line_sizes):
        draw.text((_PADDING - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
        y += line_height + _LINE_SPACING
    return image


def _draw_mixed(lines: list[str], fonts: Sequence[ImageFont.FreeTypeFont]) -> PIL.Image.Image:
    # Each run is drawn with its own font, on the line's baseline, and each line is as tall as the built-in font's
    # ascent and descent
    ascent, descent = fonts[0].getmetrics()
    line_height = max(1, ascent + descent)
    runs_per_line = [_font_runs(line or " ", fonts) for line in lines]
    width = max(max(1, round(sum(font.getlength(run) for run, font in runs))) for runs in runs_per_line)
    height = line_height * len(lines) + _LINE_SPACING * (len(lines) - 1)

    image = PIL.Image.new("RGB", (width + _PADDING * 2, height + _PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = _PADDING
    for runs in runs_per_line:
        x = float(_PADDING)
        for run, font in runs:
            draw.text((x, y + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
            x += font.getlength(run)
        y += line_height + _LINE_SPACING
    return image
