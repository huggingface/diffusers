# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import torch
from PIL import Image

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


def prepare_condition_images(image_processor, images, output_resolution):
    images = images if isinstance(images, list) else [images]
    if not images:
        raise ValueError("Provide at least one condition image.")
    resized, tensors = [], []
    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise ValueError("Condition images must be PIL images or numpy arrays in a flat list.")
        image = image.convert("RGBA")
        ratio = image.width / image.height
        width = round(math.sqrt(output_resolution**2 * ratio) / 32) * 32
        height = round(math.sqrt(output_resolution**2 / ratio) / 32) * 32
        if min(width, height) < 32:
            raise ValueError("The condition image aspect ratio produces a dimension below 32 pixels.")
        resized.append(image_processor.resize(image, width=width, height=height))
        tensors.append(image_processor.preprocess(image, width=width, height=height).unsqueeze(2))
    return resized, tensors


class QwenImage21ProcessImagesStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Resize condition images for the vision encoder and RGBA VAE."

    @property
    def expected_components(self):
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self):
        return [
            InputParam.template("image", required=True),
            InputParam(
                "output_resolution",
                default=1024,
                type_hint=int,
                description="Target side length used to resize condition images.",
            ),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "condition_images",
                type_hint=list,
                description="Resized RGBA images for joint vision and text encoding.",
            ),
            OutputParam("vae_images", type_hint=list, description="Normalized RGBA condition tensors."),
            OutputParam("height", type_hint=int, description="Output height in pixels."),
            OutputParam("width", type_hint=int, description="Output width in pixels."),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.condition_images, block_state.vae_images = prepare_condition_images(
            components.image_processor, block_state.image, block_state.output_resolution
        )
        block_state.height = block_state.height or block_state.condition_images[-1].height
        block_state.width = block_state.width or block_state.condition_images[-1].width
        self.set_block_state(state, block_state)
        return components, state


class QwenImage21ProcessInpaintStep(QwenImage21ProcessImagesStep):
    @property
    def description(self):
        return "Prepare one source image, its repaint mask, and optional additional reference images."

    @property
    def expected_components(self):
        return super().expected_components + [
            ComponentSpec(
                "mask_processor",
                VaeImageProcessor,
                config=FrozenDict(
                    {"vae_scale_factor": 16, "do_normalize": False, "do_binarize": True, "do_convert_grayscale": True}
                ),
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self):
        return super().inputs + [
            InputParam.template("mask_image", required=True),
            InputParam(
                "reference_images",
                type_hint=list,
                description="Additional PIL or numpy reference images, shared by every prompt. The source is always the first condition image.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return super().intermediate_outputs + [
            OutputParam(
                "source_image",
                type_hint=torch.Tensor,
                description="Normalized source RGBA image at the output resolution.",
            ),
            OutputParam(
                "processed_mask",
                type_hint=torch.Tensor,
                description="Binary repaint mask at the output resolution; white is repainted.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        source = block_state.image
        if isinstance(source, np.ndarray):
            source = Image.fromarray(source)
        if not isinstance(source, Image.Image):
            raise ValueError(
                "Inpainting requires one source PIL or numpy image; pass extra images as `reference_images`."
            )
        references = block_state.reference_images
        if references is not None and not isinstance(references, list):
            raise ValueError("`reference_images` must be a flat list of images.")
        block_state.condition_images, block_state.vae_images = prepare_condition_images(
            components.image_processor, [source] + (references or []), block_state.output_resolution
        )
        height = block_state.height or block_state.condition_images[0].height
        width = block_state.width or block_state.condition_images[0].width
        if min(height, width) < 32 or height % 32 or width % 32:
            raise ValueError("`height` and `width` must be positive multiples of 32.")
        block_state.height, block_state.width = height, width
        block_state.source_image = components.image_processor.preprocess(
            source.convert("RGBA"), height=height, width=width
        )
        block_state.processed_mask = components.mask_processor.preprocess(
            block_state.mask_image, height=height, width=width
        )
        if block_state.processed_mask.shape[0] != 1:
            raise ValueError("Inpainting requires one mask shared by the prompt batch.")
        self.set_block_state(state, block_state)
        return components, state
