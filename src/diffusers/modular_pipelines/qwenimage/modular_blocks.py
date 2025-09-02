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

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict
from .before_denoise import (
    QwenImageControlNetInputsStep,
    QwenImageEditRoPEInputsStep,
    QwenImageInpaintPrepareLatentsStep,
    QwenImagePackLatentsDynamicStep,
    QwenImagePrepareLatentsStep,
    QwenImageRoPEInputsStep,
    QwenImageSetTimestepsStep,
    QwenImageSetTimestepsWithStrengthStep,
)
from .decoders import QwenImageDecodeDynamicStep, QwenImageInpaintDecodeStep
from .denoise import (
    QwenImageControlNetDenoiseStep,
    QwenImageDenoiseStep,
    QwenImageEditDenoiseStep,
    QwenImageEditInpaintDenoiseStep,
    QwenImageInpaintControlNetDenoiseStep,
    QwenImageInpaintDenoiseStep,
    QwenImageLoopBeforeDenoiserControlNet,
)
from .encoders import (
    QwenImageEditInpaintVaeEncoderStep,
    QwenImageEditResizeStep,
    QwenImageEditTextEncoderStep,
    QwenImageInpaintVaeEncoderStep,
    QwenImageTextEncoderStep,
    QwenImageVaeEncoderDynamicStep,
)
from .inputs import QwenImageInputsDynamicStep


logger = logging.get_logger(__name__)


# 1. vae encoders


class QwenImageOptionalInpaintVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintVaeEncoderStep]
    block_names = ["inpaint"]
    block_trigger_inputs = ["mask_image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block.\n"
            + " - `QwenImageInpaintVaeEncoderStep` (inpaint) is used when `mask_image` is provided.\n"
            + " - if `mask_image` is not provided, step will be skipped."
        )


class QwenImageOptionalControlNetVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [QwenImageVaeEncoderDynamicStep(input_name="control_image", output_name="control_image_latents")]
    block_names = ["controlnet"]
    block_trigger_inputs = ["control_image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block.\n"
            + " - `QwenImageVaeEncoderDynamicStep` (controlnet) is used when `control_image` is provided.\n"
            + " - if `control_image` is not provided, step will be skipped."
        )


class QwenImageEditAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [
        QwenImageEditInpaintVaeEncoderStep,
        QwenImageVaeEncoderDynamicStep(input_name="image", output_name="image_latents", do_resize=False),
    ]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["mask_image", "image"]

    @property
    def description(self):
        return (
            "Auto pipeline block that works for edit and edit_inpaint tasks.\n"
            + " - `QwenImageEditInpaintVaeEncoderStep` (edit_inpaint) is used when `mask_image` is provided.\n"
            + " - `QwenImageEditVaeEncoderStep` (edit) is used when `image` is provided.\n"
            + " - if `mask_image` or `image` is not provided, step will be skipped."
        )

# 2. before denoise

# QwenImage
# - text2image
class QwenImageBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageInputsDynamicStep(),
        QwenImageRoPEInputsStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsStep(),
    ]
    block_names = ["inputs", "prepare_rope_inputs", "prepare_latents", "set_timesteps"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for text2image tasks.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `QwenImageInputsDynamicStep` is used to adjust the batch size of the model inputs\n"
            + " - `QwenImageRoPEInputsStep` is used to prepare the RoPE inputs\n"
            + " - `QwenImagePrepareLatentsStep` is used to prepare the latents\n"
            + " - `QwenImageSetTimestepsStep` is used to set the timesteps\n"
        )

# QwenImage
# - inpaint
class QwenImageInpaintBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"]),
        QwenImageRoPEInputsStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsWithStrengthStep(),
        QwenImageInpaintPrepareLatentsStep(),
    ]
    block_names = ["inputs", "prepare_rope_inputs", "prepare_latents", "set_timesteps", "prepare_inpaint_latents"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for inpaint tasks.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `QwenImageInputsDynamicStep` is used to adjust the batch size of the model inputs\n"
            + " - `QwenImageRoPEInputsStep` is used to prepare the RoPE inputs\n"
            + " - `QwenImagePrepareLatentsStep` is used to generate the initial noise\n"
            + " - `QwenImageSetTimestepsWithStrengthStep` is used to set the timesteps\n"
            + " - `QwenImageInpaintPrepareLatentsStep` is used to prepare the inpaint latents\n"
        )


# QwenImage
# - text2image & inpaint
class QwenImageAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintBeforeDenoiseStep, QwenImageBeforeDenoiseStep]
    block_names = ["inpaint", "text2image"]
    block_trigger_inputs = ["mask_image", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for text2image and inpaint tasks.\n"
            + "This is an auto pipeline block that works for text2img, inpainting tasks.\n"
            + " - `QwenImageInpaintBeforeDenoiseStep` (inpaint) is used when `mask_image` is provided.\n"
            + " - `QwenImageBeforeDenoiseStep` (text2img) is used when `mask_image` is not provided.\n"
        )


# controlnet 
#(currently only available for qwenimage, not for qwenimage-edit)
class QwenImageOptionalControlNetInputsStep(AutoPipelineBlocks):
    block_classes = [QwenImageControlNetInputsStep]
    block_names = ["controlnet"]
    block_trigger_inputs = ["control_image"]

    @property
    def description(self):
        return (
            "Auto pipeline block that works for controlnet tasks.\n"
            + " - `QwenImageControlNetInputsStep` is used when `control_image` is provided.\n"
            + " - if `control_image` is not provided, step will be skipped."
        )


# QwenImage-Edit
# - img2img
class QwenImageEditBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"

    block_classes = [
        QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"]),
        QwenImageEditRoPEInputsStep(),
        QwenImagePackLatentsDynamicStep("image_latents"),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsStep(),
    ]

    block_names = [
        "inputs", 
        "prepare_rope_inputs", 
        "prepare_image_latents", 
        "prepare_latents", 
        "set_timesteps"
    ]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for edit tasks.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `QwenImageInputsDynamicStep` is used to adjust the batch size of the model inputs\n"
            + " - `QwenImageEditRoPEInputsStep` is used to prepare the RoPE inputs\n"
            + " - `QwenImagePackLatentsDynamicStep` is used to pack the image latents\n"
            + " - `QwenImagePrepareLatentsStep` is used to prepare the latents\n"
            + " - `QwenImageSetTimestepsStep` is used to set the timesteps\n"
        )


# QwenImage-Edit
# - inpaint
class QwenImageEditInpaintBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"

    block_classes = [
        QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"]),
        QwenImageEditRoPEInputsStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsWithStrengthStep(),
        QwenImageInpaintPrepareLatentsStep()
    ]

    block_names = [
        "inputs", 
        "prepare_rope_inputs", 
        "prepare_latents", 
        "set_timesteps",
        "prepare_inpaint_latents"
    ]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for edit inpaint tasks.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `QwenImageInputsDynamicStep` is used to adjust the batch size of the model inputs\n"
            + " - `QwenImageEditRoPEInputsStep` is used to prepare the RoPE inputs\n"
            + " - `QwenImagePrepareLatentsStep` is used to prepare the latents\n"
            + " - `QwenImageSetTimestepsWithStrengthStep` is used to set the timesteps\n"
            + " - `QwenImageInpaintPrepareLatentsStep` is used to prepare the inpaint latents\n"
        )


# QwenImage-edit
# - img2img & inpaint
class QwenImageEditAutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [QwenImageEditInpaintBeforeDenoiseStep, QwenImageEditBeforeDenoiseStep]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["mask_image", "image"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for edit (img2img) and edit inpaint tasks.\n"
            + "This is an auto pipeline block that works for edit (img2img) and edit inpaint tasks.\n"
            + " - `QwenImageEditInpaintBeforeDenoiseStep` (inpaint) is used when `mask_image` is provided.\n"
            + " - `QwenImageEditBeforeDenoiseStep` (img2img) is used when `mask_image` is not provided and `image` is provided.\n"
            + " - if `mask_image` or `image` is not provided, step will be skipped."
        )

# 3. denoise

# Controlnet
class QwenImageControlNetAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintControlNetDenoiseStep, QwenImageControlNetDenoiseStep]
    block_names = ["inpaint_denoise", "denoise"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self):
        return (
            "Auto pipeline block that works for inpaint and text2image tasks.\n"
            + " - `QwenImageInpaintControlNetDenoiseStep` (inpaint) is used when `mask` is provided.\n"
            + " - `QwenImageControlNetDenoiseStep` (text2image) is used when `mask` is not provided.\n"
        )


class QwenImageAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        QwenImageControlNetAutoDenoiseStep,
        QwenImageInpaintDenoiseStep,
        QwenImageDenoiseStep,
    ]
    block_names = ["controlnet_denoise", "inpaint_denoise", "denoise"]
    block_trigger_inputs = ["control_image_latents", "mask", None]

    @property
    def description(self):
        return (
            "Auto pipeline block that works for inpaint and text2image tasks. It also works with controlnet\n"
            + " - `QwenImageControlNetAutoDenoiseStep` (controlnet) is used when `control_image_latents` is provided.\n"
            + " - `QwenImageInpaintDenoiseStep` (inpaint) is used when `mask` is provided.\n"
            + " - `QwenImageDenoiseStep` (text2image) is used when `mask` is not provided.\n"
        )


# QwenImage-Edit
# - img2img & inpaint
class QwenImageEditAutoDenoiseStep(AutoPipelineBlocks):

    model_name = "qwenimage-edit"

    block_classes = [QwenImageEditInpaintDenoiseStep, QwenImageEditDenoiseStep]
    block_names = ["inpaint_denoise", "denoise"]
    block_trigger_inputs = ["mask", "image_latents"]
    @property
    def description(self):
        return (
            "Auto pipeline block that works for edit (img2img) and edit inpaint tasks.\n"
            + " - `QwenImageEditInpaintDenoiseStep` (inpaint) is used when `mask` is provided.\n"
            + " - `QwenImageEditDenoiseStep` (img2img) is used when `mask` is not provided and `image_latents` is provided.\n"
            + " - if `mask` or `image_latents` is not provided, step will be skipped."
        )

# 4. decode
class QwenImageAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintDecodeStep, QwenImageDecodeDynamicStep()]
    block_names = ["inpaint_decode", "decode"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self):
        return (
            "Auto pipeline block that works for inpaint and text2image tasks.\n"
            + " - `QwenImageInpaintDecodeStep` (inpaint) is used when `mask` is provided.\n"
            + " - `QwenImageDecodeDynamicStep` (text2image) is used when `mask` is not provided.\n"
        )


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("input", QwenImageInputsDynamicStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("denoise", QwenImageDenoiseStep()),
        ("decode", QwenImageDecodeDynamicStep()),
    ]
)


INPAINT_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageInpaintVaeEncoderStep()),
        ("input", QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"])),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("denoise", QwenImageInpaintDenoiseStep()),
        ("decode", QwenImageInpaintDecodeStep()),
    ]
)

CONTROLNET_BLOCKS = InsertableDict(
    [
        (
            "controlnet_vae_encoder",
            QwenImageVaeEncoderDynamicStep(input_name="control_image", output_name="control_image_latents"),
        ),
        ("controlnet_before_denoise", QwenImageControlNetInputsStep()),
        (
            "controlnet_denoise_loop_before",
            QwenImageLoopBeforeDenoiserControlNet(),
        ),  # insert before the denoiseloop_denoiser
    ]
)

AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageOptionalInpaintVaeEncoderStep()),
        ("controlnet_vae_encoder", QwenImageOptionalControlNetVaeEncoderStep()),
        ("before_denoise", QwenImageAutoBeforeDenoiseStep()),
        ("controlnet_inputs", QwenImageOptionalControlNetInputsStep()),
        ("denoise", QwenImageAutoDenoiseStep()),
        ("decode", QwenImageAutoDecodeStep()),
    ]
)


EDIT_BLOCKS = InsertableDict(
    [
        ("image_resize", QwenImageEditResizeStep()),
        ("text_encoder", QwenImageEditTextEncoderStep()),
        ("vae_encoder",QwenImageVaeEncoderDynamicStep(input_name="image", output_name="image_latents", do_resize=False)),
        ("input", QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"])),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
        ("prepare_image_latents", QwenImagePackLatentsDynamicStep("image_latents")),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("denoise", QwenImageEditDenoiseStep()),
        ("decode", QwenImageDecodeDynamicStep()),
    ]
)


EDIT_INPAINT_BLOCKS = InsertableDict(
    [
        ("image_resize", QwenImageEditResizeStep()),
        ("text_encoder", QwenImageEditTextEncoderStep()),
        ("vae_encoder", QwenImageEditInpaintVaeEncoderStep()),
        ("input", QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"])),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("denoise", QwenImageEditInpaintDenoiseStep()),
        ("decode", QwenImageInpaintDecodeStep()),
    ]
)


EDIT_AUTO_BLOCKS = InsertableDict(
    [
        ("image_resize", QwenImageEditResizeStep()),
        ("text_encoder", QwenImageEditTextEncoderStep()),
        ("vae_encoder", QwenImageEditAutoVaeEncoderStep()),
        ("before_denoise", QwenImageEditAutoBeforeDenoiseStep()),
        ("denoise", QwenImageEditAutoDenoiseStep()),
        ("decode", QwenImageAutoDecodeStep()),
    ]
)

ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "edit": EDIT_BLOCKS,
    "edit_inpaint": EDIT_INPAINT_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
    "auto": AUTO_BLOCKS,
    "edit_auto": EDIT_AUTO_BLOCKS,
}


# Auto Pipelines Blocks

# QwenImage
class QwenImageAutoBlocks(SequentialPipelineBlocks):

    model_name = "qwenimage"

    block_classes = [
        QwenImageTextEncoderStep(),
        QwenImageOptionalInpaintVaeEncoderStep(),
        QwenImageOptionalControlNetVaeEncoderStep(),
        QwenImageAutoBeforeDenoiseStep(),
        QwenImageOptionalControlNetInputsStep(),
        QwenImageAutoDenoiseStep(),
        QwenImageAutoDecodeStep(),
    ]
    block_names = [
        "text_encoder",
        "vae_encoder",
        "controlnet_vae_encoder",
        "before_denoise",
        "controlnet_inputs",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using QwenImage.\n"
            + "- for image-to-image generation, you need to provide `image`\n"
            + "- for inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop` \n"
            + "- to run the controlnet workflow, you need to provide `control_image`\n"
            + "- for text-to-image generation, all you need to provide is `prompt`"
        )



# QwenImage-Edit
class QwenImageEditAutoBlocks(SequentialPipelineBlocks):

    model_name = "qwenimage-edit"

    block_classes = [
        QwenImageEditResizeStep(),
        QwenImageEditTextEncoderStep(),
        QwenImageEditAutoVaeEncoderStep(),
        QwenImageEditAutoBeforeDenoiseStep(),
        QwenImageEditAutoDenoiseStep(),
        QwenImageAutoDecodeStep(),
    ]
    block_names = [
        "image_resize",
        "text_encoder",
        "vae_encoder",
        "before_denoise",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for edit (img2img) and edit inpaint tasks using QwenImage-Edit.\n"
            + "- for edit (img2img) generation, you need to provide `image`\n"
            + "- for edit inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop` \n"
        )