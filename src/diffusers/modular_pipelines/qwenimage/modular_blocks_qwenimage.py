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

from typing import List

import PIL.Image
import torch

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    QwenImageControlNetBeforeDenoiserStep,
    QwenImageCreateMaskLatentsStep,
    QwenImagePrepareLatentsStep,
    QwenImagePrepareLatentsWithStrengthStep,
    QwenImageRoPEInputsStep,
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
    QwenImageControlNetDenoiseStep,
    QwenImageDenoiseStep,
    QwenImageInpaintControlNetDenoiseStep,
    QwenImageInpaintDenoiseStep,
)
from .encoders import (
    QwenImageControlNetVaeEncoderStep,
    QwenImageInpaintProcessImagesInputStep,
    QwenImageProcessImagesInputStep,
    QwenImageTextEncoderStep,
    QwenImageVaeEncoderStep,
)
from .inputs import (
    QwenImageAdditionalInputsStep,
    QwenImageControlNetInputsStep,
    QwenImageTextInputsStep,
)


logger = logging.get_logger(__name__)


# ====================
# 1. VAE ENCODER
# ====================


class QwenImageInpaintVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [QwenImageInpaintProcessImagesInputStep(), QwenImageVaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return (
            "This step is used for processing image and mask inputs for inpainting tasks. It:\n"
            " - Resizes the image to the target size, based on `height` and `width`.\n"
            " - Processes and updates `image` and `mask_image`.\n"
            " - Creates `image_latents`."
        )


class QwenImageImg2ImgVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = [QwenImageProcessImagesInputStep(), QwenImageVaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


# Auto VAE encoder
class QwenImageAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintVaeEncoderStep, QwenImageImg2ImgVaeEncoderStep]
    block_names = ["inpaint", "img2img"]
    block_trigger_inputs = ["mask_image", "image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block.\n"
            + " - `QwenImageInpaintVaeEncoderStep` (inpaint) is used when `mask_image` is provided.\n"
            + " - `QwenImageImg2ImgVaeEncoderStep` (img2img) is used when `image` is provided.\n"
            + " - if `mask_image` or `image` is not provided, step will be skipped."
        )


# optional controlnet vae encoder
class QwenImageOptionalControlNetVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [QwenImageControlNetVaeEncoderStep]
    block_names = ["controlnet"]
    block_trigger_inputs = ["control_image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block.\n"
            + " - `QwenImageControlNetVaeEncoderStep` (controlnet) is used when `control_image` is provided.\n"
            + " - if `control_image` is not provided, step will be skipped."
        )


# ====================
# 2. DENOISE (input -> prepare_latents -> set_timesteps -> prepare_rope_inputs -> denoise -> after_denoise)
# ====================


# assemble input steps
class QwenImageImg2ImgInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [QwenImageTextInputsStep(), QwenImageAdditionalInputsStep(image_latent_inputs=["image_latents"])]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return "Input step that prepares the inputs for the img2img denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


class QwenImageInpaintInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageAdditionalInputsStep(
            image_latent_inputs=["image_latents"], additional_batch_inputs=["processed_mask_image"]
        ),
    ]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return "Input step that prepares the inputs for the inpainting denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents` and `processed_mask_image`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


# assemble prepare latents steps
class QwenImageInpaintPrepareLatentsStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [QwenImagePrepareLatentsWithStrengthStep(), QwenImageCreateMaskLatentsStep()]
    block_names = ["add_noise_to_latents", "create_mask_latents"]

    @property
    def description(self) -> str:
        return (
            "This step prepares the latents/image_latents and mask inputs for the inpainting denoising step. It:\n"
            " - Add noise to the image latents to create the latents input for the denoiser.\n"
            " - Create the pachified latents `mask` based on the processedmask image.\n"
        )


# assemble denoising steps


# Qwen Image (text2image)
class QwenImageCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsStep(),
        QwenImageRoPEInputsStep(),
        QwenImageDenoiseStep(),
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
        return "step that denoise noise into image for text2image task. It includes the denoise loop, as well as prepare the inputs (timesteps, latents, rope inputs etc.)."


# Qwen Image (inpainting)
class QwenImageInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageInpaintInputStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsWithStrengthStep(),
        QwenImageInpaintPrepareLatentsStep(),
        QwenImageRoPEInputsStep(),
        QwenImageInpaintDenoiseStep(),
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
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for inpaint task."


# Qwen Image (image2image)
class QwenImageImg2ImgCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageImg2ImgInputStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsWithStrengthStep(),
        QwenImagePrepareLatentsWithStrengthStep(),
        QwenImageRoPEInputsStep(),
        QwenImageDenoiseStep(),
        QwenImageAfterDenoiseStep(),
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_img2img_latents",
        "prepare_rope_inputs",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for img2img task."


# Qwen Image (text2image) with controlnet
class QwenImageControlNetCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageControlNetInputsStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsStep(),
        QwenImageRoPEInputsStep(),
        QwenImageControlNetBeforeDenoiserStep(),
        QwenImageControlNetDenoiseStep(),
        QwenImageAfterDenoiseStep(),
    ]
    block_names = [
        "input",
        "controlnet_input",
        "prepare_latents",
        "set_timesteps",
        "prepare_rope_inputs",
        "controlnet_before_denoise",
        "controlnet_denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return "step that denoise noise into image for text2image task. It includes the denoise loop, as well as prepare the inputs (timesteps, latents, rope inputs etc.)."


# Qwen Image (inpainting) with controlnet
class QwenImageControlNetInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageInpaintInputStep(),
        QwenImageControlNetInputsStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsWithStrengthStep(),
        QwenImageInpaintPrepareLatentsStep(),
        QwenImageRoPEInputsStep(),
        QwenImageControlNetBeforeDenoiserStep(),
        QwenImageInpaintControlNetDenoiseStep(),
        QwenImageAfterDenoiseStep(),
    ]
    block_names = [
        "input",
        "controlnet_input",
        "prepare_latents",
        "set_timesteps",
        "prepare_inpaint_latents",
        "prepare_rope_inputs",
        "controlnet_before_denoise",
        "controlnet_denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for inpaint task."


# Qwen Image (image2image) with controlnet
class QwenImageControlNetImg2ImgCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageImg2ImgInputStep(),
        QwenImageControlNetInputsStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsWithStrengthStep(),
        QwenImagePrepareLatentsWithStrengthStep(),
        QwenImageRoPEInputsStep(),
        QwenImageControlNetBeforeDenoiserStep(),
        QwenImageControlNetDenoiseStep(),
        QwenImageAfterDenoiseStep(),
    ]
    block_names = [
        "input",
        "controlnet_input",
        "prepare_latents",
        "set_timesteps",
        "prepare_img2img_latents",
        "prepare_rope_inputs",
        "controlnet_before_denoise",
        "controlnet_denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for img2img task."


# Auto denoise step for QwenImage
class QwenImageAutoCoreDenoiseStep(ConditionalPipelineBlocks):
    block_classes = [
        QwenImageCoreDenoiseStep,
        QwenImageInpaintCoreDenoiseStep,
        QwenImageImg2ImgCoreDenoiseStep,
        QwenImageControlNetCoreDenoiseStep,
        QwenImageControlNetInpaintCoreDenoiseStep,
        QwenImageControlNetImg2ImgCoreDenoiseStep,
    ]
    block_names = [
        "text2image",
        "inpaint",
        "img2img",
        "controlnet_text2image",
        "controlnet_inpaint",
        "controlnet_img2img",
    ]
    block_trigger_inputs = ["control_image_latents", "processed_mask_image", "image_latents"]
    default_block_name = "text2image"

    def select_block(self, control_image_latents=None, processed_mask_image=None, image_latents=None):
        if control_image_latents is not None:
            if processed_mask_image is not None:
                return "controlnet_inpaint"
            elif image_latents is not None:
                return "controlnet_img2img"
            else:
                return "controlnet_text2image"
        else:
            if processed_mask_image is not None:
                return "inpaint"
            elif image_latents is not None:
                return "img2img"
            else:
                return "text2image"

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `QwenImageCoreDenoiseStep` (text2image) for text2image tasks.\n"
            + " - `QwenImageInpaintCoreDenoiseStep` (inpaint) for inpaint tasks.\n"
            + " - `QwenImageImg2ImgCoreDenoiseStep` (img2img) for img2img tasks.\n"
            + " - `QwenImageControlNetCoreDenoiseStep` (controlnet_text2image) for text2image tasks with controlnet.\n"
            + " - `QwenImageControlNetInpaintCoreDenoiseStep` (controlnet_inpaint) for inpaint tasks with controlnet.\n"
            + " - `QwenImageControlNetImg2ImgCoreDenoiseStep` (controlnet_img2img) for img2img tasks with controlnet.\n"
            + "This step support text-to-image, image-to-image, inpainting, and controlnet tasks for QwenImage:\n"
            + " - for image-to-image generation, you need to provide `image_latents`\n"
            + " - for inpainting, you need to provide `processed_mask_image` and `image_latents`\n"
            + " - to run the controlnet workflow, you need to provide `control_image_latents`\n"
            + " - for text-to-image generation, all you need to provide is prompt embeddings"
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                name="latents", type_hint=torch.Tensor, description="The latents generated by the denoising step"
            ),
        ]


# ====================
# 3. DECODE
# ====================


# standard decode step works for most tasks except for inpaint
class QwenImageDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [QwenImageDecoderStep(), QwenImageProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image."


# Inpaint decode step
class QwenImageInpaintDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [QwenImageDecoderStep(), QwenImageInpaintProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image, optional apply the mask overally to the original image."


# Auto decode step for QwenImage
class QwenImageAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintDecodeStep, QwenImageDecodeStep]
    block_names = ["inpaint_decode", "decode"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self):
        return (
            "Decode step that decode the latents into images. \n"
            " This is an auto pipeline block that works for inpaint/text2image/img2img tasks, for both QwenImage and QwenImage-Edit.\n"
            + " - `QwenImageInpaintDecodeStep` (inpaint) is used when `mask` is provided.\n"
            + " - `QwenImageDecodeStep` (text2image/img2img) is used when `mask` is not provided.\n"
        )


# ====================
# 4. AUTO BLOCKS & PRESETS
# ====================
AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageAutoVaeEncoderStep()),
        ("controlnet_vae_encoder", QwenImageOptionalControlNetVaeEncoderStep()),
        ("denoise", QwenImageAutoCoreDenoiseStep()),
        ("decode", QwenImageAutoDecodeStep()),
    ]
)


class QwenImageAutoBlocks(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using QwenImage.\n"
            + "- for image-to-image generation, you need to provide `image`\n"
            + "- for inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop` \n"
            + "- to run the controlnet workflow, you need to provide `control_image`\n"
            + "- for text-to-image generation, all you need to provide is `prompt`"
        )

    @property
    def outputs(self):
        return [
            OutputParam(name="images", type_hint=List[List[PIL.Image.Image]]),
        ]
