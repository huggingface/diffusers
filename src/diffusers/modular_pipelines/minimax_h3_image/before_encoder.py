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
from PIL import ImageOps

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ImageModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def bound_short_edge(
    width: int, height: int, min_short_edge: int, max_short_edge: int, multiple: int
) -> tuple[int, int]:
    r"""
    The size a MiniMax-H3-Image reference is resized to for one of its consumers.

    A short edge within `[min_short_edge, max_short_edge]` keeps the original size. Otherwise the image is scaled,
    keeping its ratio, so that the short edge lands on the nearer bound, and both sides round to `multiple`.

    Returns:
        `tuple[int, int]`: the `(width, height)` to resize to.
    """
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )


class MiniMaxH3ImageEditResizeStep(ModularPipelineBlocks):
    model_name = "minimax-h3-image"

    @property
    def description(self) -> str:
        return (
            "Resizes the reference images of an edit, once for the VAE and once for the text encoder. Each copy "
            "keeps the original pixels when its short edge is within the consumer's bounds, and is otherwise scaled "
            "to the nearer bound with both sides rounded to multiples of 32. The references are never resized to the "
            "size of the generated image."
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
            ConfigSpec("reference_min_short_edge", 1024),
            ConfigSpec("reference_max_short_edge", 2048),
            ConfigSpec("vision_min_short_edge", 1024),
            ConfigSpec("vision_max_short_edge", 1024),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="image",
                type_hint=PIL.Image.Image | list[PIL.Image.Image],
                required=True,
                description=(
                    "The reference image of the edit, or a list of them. The prompt refers to them in order as "
                    "`<Picture 1>`, `<Picture 2>`, ..."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_images",
                type_hint=list[PIL.Image.Image],
                description="The reference images resized for the VAE.",
            ),
            OutputParam(
                "vision_images",
                type_hint=list[PIL.Image.Image],
                description="The reference images resized for the text encoder.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ImageModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ImageModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        images = block_state.image if isinstance(block_state.image, list) else [block_state.image]
        # References are read upright: the EXIF orientation is applied before anything else looks at the size.
        images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
        multiple = components.canvas_multiple
        for image in images:
            if min(image.size) < multiple:
                raise ValueError(f"A reference image needs at least {multiple} pixels on each side, got {image.size}.")

        def resize(image, min_short_edge, max_short_edge):
            size = bound_short_edge(*image.size, min_short_edge, max_short_edge, multiple)
            if image.size == size:
                return image
            return components.image_processor.resize(image, height=size[1], width=size[0])

        config = components.config
        block_state.reference_images = [
            resize(image, config.reference_min_short_edge, config.reference_max_short_edge) for image in images
        ]
        block_state.vision_images = [
            resize(image, config.vision_min_short_edge, config.vision_max_short_edge) for image in images
        ]

        self.set_block_state(state, block_state)
        return components, state
