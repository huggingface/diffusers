# Copyright 2026 Pruna AI and The HuggingFace Team. All rights reserved.
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

import PIL.Image
import torch

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import PH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _size_at_area(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
    r"""
    The `(width, height)` of an image resized to about `pixels` at its own aspect ratio, with both sides multiples of
    `multiple`. An image with sides that are already multiples of `multiple` and an area within 20% of `pixels` keeps
    its size.
    """
    if width % multiple == 0 and height % multiple == 0 and 0.8 * pixels <= width * height <= 1.2 * pixels:
        return width, height
    ratio = width / height
    return (
        max(multiple, round(math.sqrt(pixels * ratio) / multiple) * multiple),
        max(multiple, round(math.sqrt(pixels / ratio) / multiple) * multiple),
    )


MAX_EDIT_IMAGES = 8


def round_canvas(height: int, width: int, multiple: int) -> tuple[int, int]:
    r"""`height` and `width` rounded to the nearest multiples of `multiple`, with a warning when that changes them."""
    rounded = tuple(max(multiple, round(side / multiple) * multiple) for side in (height, width))
    if rounded != (height, width):
        logger.warning(
            f"`height` and `width` are rounded to multiples of {multiple}: {height}x{width} becomes "
            f"{rounded[0]}x{rounded[1]}."
        )
    return rounded


def _size_in_area_range(width: int, height: int, min_pixels: int, max_pixels: int, multiple: int) -> tuple[int, int]:
    r"""
    The `(width, height)` of an image that keeps its size, or is scaled at its own aspect ratio to `min_pixels` or
    `max_pixels` when its area is outside that range, with both sides rounded to multiples of `multiple`.
    """
    area = width * height
    scale = math.sqrt(min(max(area, min_pixels), max_pixels) / area)
    return tuple(max(multiple, round(side * scale / multiple) * multiple) for side in (width, height))


class PH3ImageEditResizeStep(ModularPipelineBlocks):
    model_name = "p-h3-image"

    @property
    def description(self) -> str:
        return (
            "Resizes the images of an edit. For the VAE, each keeps its size unless its area is outside "
            "`reference_min_pixels` to `reference_max_pixels`; for the text encoder, each is resized to about "
            "`vision_pixels`. Both keep the aspect ratio and round the sides to multiples of 32. Without `height` "
            "and `width`, the edit is generated at the size of the first image, scaled up to `canvas_min_pixels` if "
            "smaller. Passed `height` and `width` change the canvas only; the images are never resized to it."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "do_normalize": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("reference_min_pixels", 1024**2),
            ConfigSpec("reference_max_pixels", 8 * 1024**2),
            ConfigSpec("canvas_min_pixels", 4 * 1024**2),
            ConfigSpec("vision_pixels", 1_000_000),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="image",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                required=True,
                description=(
                    f"The image to edit, or a list of at most {MAX_EDIT_IMAGES} images: the first is edited, the "
                    "others are references."
                ),
            ),
            InputParam.template(
                "height",
                description=(
                    "Height of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for "
                    "`text2image`, which works best at 8 megapixels. An edit defaults to the first image's size, at "
                    "least `canvas_min_pixels`, and works best at 4 or 8 megapixels."
                ),
            ),
            InputParam.template(
                "width",
                description=(
                    "Width of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for "
                    "`text2image`, which works best at 8 megapixels. An edit defaults to the first image's size, at "
                    "least `canvas_min_pixels`, and works best at 4 or 8 megapixels."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved height of the edited image in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the edited image in pixels."),
            OutputParam(
                "reference_images",
                type_hint=list[PIL.Image.Image],
                description="The images of the edit resized for the VAE.",
            ),
            OutputParam(
                "vision_images",
                type_hint=list[PIL.Image.Image],
                description="The images of the edit resized for the text encoder.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: PH3ImageModularPipeline, state: PipelineState
    ) -> tuple[PH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        images = block_state.image if isinstance(block_state.image, list) else [block_state.image]
        images = [image.convert("RGB") for image in images]
        if len(images) > MAX_EDIT_IMAGES:
            raise ValueError(f"An edit takes at most {MAX_EDIT_IMAGES} images, got {len(images)}.")

        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        multiple = components.canvas_multiple
        min_pixels, max_pixels = components.config.reference_min_pixels, components.config.reference_max_pixels
        reference_sizes = [_size_in_area_range(*image.size, min_pixels, max_pixels, multiple) for image in images]
        if block_state.height is not None:
            block_state.height, block_state.width = round_canvas(block_state.height, block_state.width, multiple)
        else:
            canvas_min_pixels = max(components.config.canvas_min_pixels, min_pixels)
            reference_sizes[0] = _size_in_area_range(*images[0].size, canvas_min_pixels, max_pixels, multiple)
            block_state.width, block_state.height = reference_sizes[0]
        vision_sizes = [_size_at_area(*image.size, components.config.vision_pixels, multiple) for image in images]

        def resize(image, size):
            if image.size == size:
                return image
            return components.image_processor.resize(image, height=size[1], width=size[0])

        block_state.reference_images = [resize(image, size) for image, size in zip(images, reference_sizes)]
        block_state.vision_images = [resize(image, size) for image, size in zip(images, vision_sizes)]

        self.set_block_state(state, block_state)
        return components, state
