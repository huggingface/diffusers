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

from typing import Optional

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict
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
    QwenImageEditResizeStep,
    QwenImageTextEncoderStep,
    QwenImageProcessImagesInputStep,
    QwenImageVaeEncoderStep,
    QwenImageLayeredGetImagePromptStep,
    QwenImageLayeredPermuteLatentsStep,
)
from .inputs import (
    QwenImageInputsDynamicStep,
    QwenImageTextInputsStep,
)


logger = logging.get_logger(__name__)


# ====================
# 1. TEXT ENCODER
# ====================

class QwenImageLayeredTextEncoderStep(SequentialPipelineBlocks):
    """Text encoder that takes text prompt, will generate a prompt based on image if not provided."""
    model_name = "qwenimage-layered"
    block_classes = [
        QwenImageEditResizeStep(),
        QwenImageLayeredGetImagePromptStep(),
        QwenImageTextEncoderStep(),
    ]
    block_names = ["resize", "get_image_prompt", "encode"]

    @property
    def description(self) -> str:
        return "QwenImage-Layered Text encoder step that encode the text prompt, will generate a prompt based on image if not provided."


# ====================
# 2. VAE ENCODER
# ====================

# Edit VAE encoder
class QwenImageLayeredVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage-layered"
    block_classes = [
        QwenImageEditResizeStep(),
        QwenImageProcessImagesInputStep(),
        QwenImageVaeEncoderStep(),
        QwenImageLayeredPermuteLatentsStep(),
    ]
    block_names = ["resize", "preprocess", "encode", "permute"]

    @property
    def description(self) -> str:
        return "Vae encoder step that encode the image inputs into their latent representations."





# ====================
# 3. DENOISE - input -> prepare_latents -> set_timesteps -> prepare_rope_inputs -> denoise -> after_denoise
# ====================

# Edit input step
class QwenImageEditInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageInputsDynamicStep(image_latent_inputs=["image_latents"]),
    ]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the edit denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs.\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


# Edit Inpaint input step
class QwenImageEditInpaintInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageInputsDynamicStep(image_latent_inputs=["image_latents"], additional_batch_inputs=["processed_mask_image"]),
    ]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the edit inpaint denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs.\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


# Edit Inpaint prepare latents step
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


# 1. Edit (img2img) core denoise
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


# 2. Edit Inpaint core denoise
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


# Auto core denoise step
class QwenImageEditAutoCoreDenoiseStep(ConditionalPipelineBlocks):
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