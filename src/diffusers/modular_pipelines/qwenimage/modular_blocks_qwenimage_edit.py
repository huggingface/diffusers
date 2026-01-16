# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional

import PIL.Image
import torch

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    QwenImageCreateMaskLatentsStep,
    QwenImageEditRoPEInputsStep,
    QwenImagePrepareLatentsStep,
    QwenImagePrepareLatentsWithStrengthStep,
    QwenImageSetTimestepsStep,
    QwenImageSetTimestepsWithStrengthStep,
)
from .decoders import (
    QwenImageAfterDenoiseStep,
    QwenImageDecoderStep,
    QwenImageInpaintProcessImagesOutputStep,
    QwenImageProcessImagesOutputStep,
)
from .denoise import (
    QwenImageEditDenoiseStep,
    QwenImageEditInpaintDenoiseStep,
)
from .encoders import (
    QwenImageEditInpaintProcessImagesInputStep,
    QwenImageEditProcessImagesInputStep,
    QwenImageEditResizeStep,
    QwenImageEditTextEncoderStep,
    QwenImageVaeEncoderStep,
)
from .inputs import (
    QwenImageAdditionalInputsStep,
    QwenImageTextInputsStep,
)


logger = logging.get_logger(__name__)


# ====================
# 1. TEXT ENCODER
# ====================


class QwenImageEditVLEncoderStep(SequentialPipelineBlocks):
    """VL encoder that takes both image and text prompts."""

    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditResizeStep(),
        QwenImageEditTextEncoderStep(),
    ]
    block_names = ["resize", "encode"]

    @property
    def description(self) -> str:
        return "QwenImage-Edit VL encoder step that encode the image and text prompts together."


# ====================
# 2. VAE ENCODER
# ====================


# Edit VAE encoder
class QwenImageEditVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditResizeStep(),
        QwenImageEditProcessImagesInputStep(),
        QwenImageVaeEncoderStep(),
    ]
    block_names = ["resize", "preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Vae encoder step that encode the image inputs into their latent representations."


# Edit Inpaint VAE encoder
class QwenImageEditInpaintVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditResizeStep(),
        QwenImageEditInpaintProcessImagesInputStep(),
        QwenImageVaeEncoderStep(input_name="processed_image", output_name="image_latents"),
    ]
    block_names = ["resize", "preprocess", "encode"]

    @property
    def description(self) -> str:
        return (
            "This step is used for processing image and mask inputs for QwenImage-Edit inpaint tasks. It:\n"
            " - resize the image for target area (1024 * 1024) while maintaining the aspect ratio.\n"
            " - process the resized image and mask image.\n"
            " - create image latents."
        )


# Auto VAE encoder
class QwenImageEditAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [QwenImageEditInpaintVaeEncoderStep, QwenImageEditVaeEncoderStep]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["mask_image", "image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            "This is an auto pipeline block.\n"
            " - `QwenImageEditInpaintVaeEncoderStep` (edit_inpaint) is used when `mask_image` is provided.\n"
            " - `QwenImageEditVaeEncoderStep` (edit) is used when `image` is provided.\n"
            " - if `mask_image` or `image` is not provided, step will be skipped."
        )


# ====================
# 3. DENOISE (input -> prepare_latents -> set_timesteps -> prepare_rope_inputs -> denoise -> after_denoise)
# ====================


# assemble input steps
class QwenImageEditInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageAdditionalInputsStep(image_latent_inputs=["image_latents"]),
    ]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the edit denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs.\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


class QwenImageEditInpaintInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageAdditionalInputsStep(
            image_latent_inputs=["image_latents"], additional_batch_inputs=["processed_mask_image"]
        ),
    ]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the edit inpaint denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs.\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


# assemble prepare latents steps
class QwenImageEditInpaintPrepareLatentsStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [QwenImagePrepareLatentsWithStrengthStep(), QwenImageCreateMaskLatentsStep()]
    block_names = ["add_noise_to_latents", "create_mask_latents"]

    @property
    def description(self) -> str:
        return (
            "This step prepares the latents/image_latents and mask inputs for the edit inpainting denoising step. It:\n"
            " - Add noise to the image latents to create the latents input for the denoiser.\n"
            " - Create the patchified latents `mask` based on the processed mask image.\n"
        )


# Qwen Image Edit (image2image) core denoise step
class QwenImageEditCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditInputStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsStep(),
        QwenImageEditRoPEInputsStep(),
        QwenImageEditDenoiseStep(),
        QwenImageAfterDenoiseStep(),
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_rope_inputs",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return "Core denoising workflow for QwenImage-Edit edit (img2img) task."


# Qwen Image Edit (inpainting) core denoise step
class QwenImageEditInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditInpaintInputStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsWithStrengthStep(),
        QwenImageEditInpaintPrepareLatentsStep(),
        QwenImageEditRoPEInputsStep(),
        QwenImageEditInpaintDenoiseStep(),
        QwenImageAfterDenoiseStep(),
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_inpaint_latents",
        "prepare_rope_inputs",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return "Core denoising workflow for QwenImage-Edit edit inpaint task."


# Auto core denoise step for QwenImage Edit
class QwenImageEditAutoCoreDenoiseStep(ConditionalPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditInpaintCoreDenoiseStep,
        QwenImageEditCoreDenoiseStep,
    ]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["processed_mask_image", "image_latents"]
    default_block_name = "edit"

    def select_block(self, processed_mask_image=None, image_latents=None) -> Optional[str]:
        if processed_mask_image is not None:
            return "edit_inpaint"
        elif image_latents is not None:
            return "edit"
        return None

    @property
    def description(self):
        return (
            "Auto core denoising step that selects the appropriate workflow based on inputs.\n"
            " - `QwenImageEditInpaintCoreDenoiseStep` when `processed_mask_image` is provided\n"
            " - `QwenImageEditCoreDenoiseStep` when `image_latents` is provided\n"
            "Supports edit (img2img) and edit inpainting tasks for QwenImage-Edit."
        )


# ====================
# 4. DECODE
# ====================


# Decode step (standard)
class QwenImageEditDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [QwenImageDecoderStep(), QwenImageProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image."


# Inpaint decode step
class QwenImageEditInpaintDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [QwenImageDecoderStep(), QwenImageInpaintProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image, optionally apply the mask overlay to the original image."


# Auto decode step
class QwenImageEditAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [QwenImageEditInpaintDecodeStep, QwenImageEditDecodeStep]
    block_names = ["inpaint_decode", "decode"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self):
        return (
            "Decode step that decode the latents into images.\n"
            "This is an auto pipeline block.\n"
            " - `QwenImageEditInpaintDecodeStep` (inpaint) is used when `mask` is provided.\n"
            " - `QwenImageEditDecodeStep` (edit) is used when `mask` is not provided.\n"
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                name="latents", type_hint=torch.Tensor, description="The latents generated by the denoising step"
            ),
        ]


# ====================
# 5. AUTO BLOCKS & PRESETS
# ====================

EDIT_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditVLEncoderStep()),
        ("vae_encoder", QwenImageEditAutoVaeEncoderStep()),
        ("denoise", QwenImageEditAutoCoreDenoiseStep()),
        ("decode", QwenImageEditAutoDecodeStep()),
    ]
)


class QwenImageEditAutoBlocks(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = EDIT_AUTO_BLOCKS.values()
    block_names = EDIT_AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Auto Modular pipeline for edit (img2img) and edit inpaint tasks using QwenImage-Edit.\n"
            "- for edit (img2img) generation, you need to provide `image`\n"
            "- for edit inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop`\n"
        )

    @property
    def outputs(self):
        return [
            OutputParam(name="images", type_hint=List[List[PIL.Image.Image]], description="The generated images"),
        ]
