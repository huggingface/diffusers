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
    QwenImageControlNetBeforeDenoiserStep,
    QwenImageEditRoPEInputsStep,
    QwenImageCreateMaskLatentsStep,
    QwenImagePrepareLatentsStep,
    QwenImagePrepareLatentsWithStrengthStep,
    QwenImageRoPEInputsStep,
    QwenImageSetTimestepsStep,
    QwenImageSetTimestepsWithStrengthStep,
)
from .decoders import QwenImageDecoderStep, QwenImageInpaintProcessImagesOutputStep, QwenImageProcessImagesOutputStep
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
    QwenImageInpaintProcessImagesInputStep,
    QwenImageProcessImagesInputDynamicStep,
    QwenImageResizeDynamicStep,
    QwenImageEditResizeDynamicStep,
    QwenImageEditTextEncoderStep,
    QwenImageTextEncoderStep,
    QwenImageVaeEncoderDynamicStep,
    QwenImageControlNetVaeEncoderStep,
)
from .inputs import QwenImageTextInputsStep, QwenImageImageInputsDynamicStep, QwenImageBatchInputsDynamicStep, QwenImageControlNetInputsStep


logger = logging.get_logger(__name__)

# 1. QwenImage
## 1.1 basic block presets: text2image, inpaint, controlnet

### 1.1.1 text2image

# YiYi TODO: refactor the modular pipeline blocks to initialize with InsertableDict directly
QwenImageDecodeBlocks = InsertableDict([
    ("decode", QwenImageDecoderStep()),
    ("postprocess", QwenImageProcessImagesOutputStep()),
])

class QwenImageDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageDecodeBlocks.values()
    block_names = QwenImageDecodeBlocks.keys()

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image."


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("input", QwenImageTextInputsStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
        ("denoise", QwenImageDenoiseStep()),
        ("decode", QwenImageDecodeStep()),
    ]
)


### 1.1.2 inpaint

#### vae encoder
QwenImageInpaintVaeEncoderBlocks = InsertableDict([
    ("resize", QwenImageResizeDynamicStep(input_name="image", output_name="resized_image")  ), # image -> resized_image
    ("preprocess", QwenImageInpaintProcessImagesInputStep), # resized_image, mask_image -> processed_image, processed_mask_image, mask_overlay_kwargs
    ("encode", QwenImageVaeEncoderDynamicStep()), # processed_image -> image_latents
])

class QwenImageInpaintVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageInpaintVaeEncoderBlocks.values()
    block_names = QwenImageInpaintVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return (
            "This step is used for processing image and mask inputs for inpainting tasks. It:\n"
            " - Resizes the image to the target size, based on `height` and `width`.\n"
            " - Processes and updates `image` and `mask_image`.\n"
            " - Creates `image_latents`."
        )


#### inputs
QwenImageInpaintInputBlocks = InsertableDict([
    ("text_inputs", QwenImageTextInputsStep()), # default step to process text embeddings
    ("additional_inputs", QwenImageBatchInputsDynamicStep(batch_inputs=["image_latents", "processed_mask_image"])), # expand batch dimension for image_latents & processed_mask_image
    ("image_latents_inputs", QwenImageImageInputsDynamicStep()), # update height/width based image_latents, patchify image_latents
])

class QwenImageInpaintInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageInpaintInputBlocks.values()
    block_names = QwenImageInpaintInputBlocks.keys()
    @property
    def description(self):
        return "Input step that prepares the inputs for the inpainting denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents` and `processed_mask_image`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


# prepare latents
QwenImageInpaintPrepareLatentsBlocks = InsertableDict([
    ("add_noise_to_latents", QwenImagePrepareLatentsWithStrengthStep()),
    ("create_mask_latents", QwenImageCreateMaskLatentsStep()),
])

class QwenImageInpaintPrepareLatentsStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageInpaintPrepareLatentsBlocks.values()
    block_names = QwenImageInpaintPrepareLatentsBlocks.keys()

    @property
    def description(self) -> str:
        return (
            "This step prepares the latents/image_latents and mask inputs for the inpainting denoising step. It:\n"
            " - Add noise to the image latents to create the latents input for the denoiser.\n"
            " - Create the pachified latents `mask` based on the processedmask image.\n"
        )


#### decode
QwenImageInpaintDecodeBlocks = InsertableDict([
    ("decode", QwenImageDecoderStep()),
    ("postprocess", QwenImageInpaintProcessImagesOutputStep()),
])

class QwenImageInpaintDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageInpaintDecodeBlocks.values()
    block_names = QwenImageInpaintDecodeBlocks.keys()

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image, optional apply the mask overally to the original image."


INPAINT_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageInpaintVaeEncoderStep()),
        ("input", QwenImageInpaintInputStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
        ("denoise", QwenImageInpaintDenoiseStep()),
        ("decode", QwenImageInpaintDecodeStep()),
    ]
)

### 1.1.3 img2img
QwenImageImg2ImgVaeEncoderBlocks = InsertableDict([
    ("resize", QwenImageResizeDynamicStep()),
    ("preprocess", QwenImageProcessImagesInputDynamicStep()),
    ("encode", QwenImageVaeEncoderDynamicStep()),
])

class QwenImageImg2ImgVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = QwenImageImg2ImgVaeEncoderBlocks.values()
    block_names = QwenImageImg2ImgVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."

### 1.1.4 controlnet (currently only available for qwenimage, not for qwenimage-edit)

#### vae encoder



#### inputs

CONTROLNET_BLOCKS = InsertableDict(
    [
        ("controlnet_vae_encoder",QwenImageControlNetVaeEncoderStep()), # vae encoder step for control_image
        ("controlnet_inputs", QwenImageControlNetInputsStep()), # additional input step for controlnet
        ("controlnet_before_denoise", QwenImageControlNetBeforeDenoiserStep()), # before denoise step (after set_timesteps step)
        ("controlnet_denoise_loop_before", QwenImageLoopBeforeDenoiserControlNet()),  # controlnet loop step (insert before the denoiseloop_denoiser)
    ]
)


## 1.2 auto pipeline blocks for QwenImage
### 1.2.1 auto encoders 

#### inpaint
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

# controlnet
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


# 1.2.2 auto inputs

class QwenImageAutoInputStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintInputStep, QwenImageTextInputsStep]
    block_names = ["inpaint", "text2image"]
    block_trigger_inputs = ["processed_mask_image", None]
    @property
    def description(self):
        return (
            "Input step that prepare the inputs for the denoising step."
            + "This is an auto pipeline block.\n"
            + " - `QwenImageInpaintInputStep` (inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageTextInputsStep` (text2image) is used when `processed_mask_image` is not provided.\n"
        )

class QwenImageOptionalControlNetInputStep(AutoPipelineBlocks):
    block_classes = [QwenImageControlNetInputsStep]
    block_names = ["controlnet"]
    block_trigger_inputs = ["control_image_latents"]
    @property
    def description(self):
        return (
            "Controlnet input step that prepare the control_image_latents input.\n"
            + "This is an auto pipeline block.\n"
            + " - `QwenImageControlNetInputsStep` (controlnet) is used when `control_image_latents` is provided.\n"
            + " - if `control_image_latents` is not provided, step will be skipped."
            )


# 1.2.3 auto before denoise step

QwenImageText2ImageBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
    ]
)

QwenImageInpaintBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
    ]
)


class QwenImageAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        SequentialPipelineBlocks.from_blocks_dict(QwenImageInpaintBeforeDenoiseBlocks), 
        SequentialPipelineBlocks.from_blocks_dict(QwenImageText2ImageBeforeDenoiseBlocks),
        ]
    block_names = ["inpaint", "text2image"]
    block_trigger_inputs = ["processed_mask_image", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step.\n"
            + "This is an auto pipeline block that works for text2img, inpainting tasks.\n"
            + " - To use inpaint workflow, make sure `processed_mask_image` is provided.\n"
            + " - text2image task will be run when `processed_mask_image` is not provided.\n"
        )

class QwenImageOptionalControlNetBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [QwenImageControlNetBeforeDenoiserStep]
    block_names = ["controlnet"]
    block_trigger_inputs = ["control_image_latents"]
    @property
    def description(self):
        return (
            "Controlnet before denoise step that prepare the controlnet input.\n"
            + "This is an auto pipeline block.\n"
            + " - `QwenImageControlNetBeforeDenoiserStep` (controlnet) is used when `control_image_latents` is provided.\n"
            + " - if `control_image_latents` is not provided, step will be skipped."
            )


# 1.2.4 auto denoise 

# controlnet
class QwenImageControlNetAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintControlNetDenoiseStep, QwenImageControlNetDenoiseStep]
    block_names = ["inpaint_denoise", "denoise"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self):
        return (
            "Auto pipeline block that works for inpaint and text2image tasks with controlnet.\n"
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
            + " - `QwenImageInpaintDenoiseStep` (inpaint) is used when `mask` is provided and `control_image_latents` is not provided.\n"
            + " - `QwenImageDenoiseStep` (text2image) is used when `mask` is not provided and `control_image_latents` is not provided.\n"
        )


### 1.2.5 auto decode

class QwenImageAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintDecodeStep, QwenImageDecodeStep]
    block_names = ["inpaint_decode", "decode"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self):
        return (
            "Auto pipeline block that works for inpaint and text2image tasks.\n"
            + " - `QwenImageInpaintDecodeStep` (inpaint) is used when `mask` is provided.\n"
            + " - `QwenImageDecodeStep` (text2image) is used when `mask` is not provided.\n"
        )


### 1.3 auto block & presets
AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageOptionalInpaintVaeEncoderStep()),
        ("controlnet_vae_encoder", QwenImageOptionalControlNetVaeEncoderStep()),
        ("input", QwenImageAutoInputStep()),
        ("controlnet_input", QwenImageOptionalControlNetInputStep()),
        ("before_denoise", QwenImageAutoBeforeDenoiseStep()),
        ("controlnet_before_denoise", QwenImageOptionalControlNetBeforeDenoiseStep()),
        ("denoise", QwenImageAutoDenoiseStep()),
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


# 2. QwenImage-Edit
## 2.1 basic block presets: edit, edit inpaint

### 2.1.1 edit

#### vl encoder: take both image and text prompts
QwenImageEditVLEncoderBlocks = InsertableDict([
    ("resize", QwenImageEditResizeDynamicStep()),
    ("encode", QwenImageEditTextEncoderStep()),
])

class QwenImageEditVLEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditVLEncoderBlocks.values()
    block_names = QwenImageEditVLEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "QwenImage-Edit VL encoder step that encode the image an text prompts together."


#### vae encoder
QwenImageEditVaeEncoderBlocks = InsertableDict([
    ("resize", QwenImageEditResizeDynamicStep()), # edit has a different resize step
    ("preprocess", QwenImageProcessImagesInputDynamicStep()), # resized_image -> processed_image
    ("encode", QwenImageVaeEncoderDynamicStep()), # processed_image -> image_latents
])

class QwenImageEditVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditVaeEncoderBlocks.values()
    block_names = QwenImageEditVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Vae encoder step that encode the image inputs into their latent representations."


#### input
QwenImageEditInputBlocks = InsertableDict([
    ("text_inputs", QwenImageTextInputsStep()), # default step to process text embeddings
    ("additional_inputs", QwenImageBatchInputsDynamicStep("image_latents")), # expand batch dimension for image_latents
    ("image_latents_inputs", QwenImageImageInputsDynamicStep("image_latents")), # update height/width based image_latents, patchify image_latents
])

class QwenImageEditInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditInputBlocks.values()
    block_names = QwenImageEditInputBlocks.keys()
    @property
    def description(self):
        return "Input step that prepares the inputs for the edit denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs: \n"
        " - `image_latents`.\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


EDIT_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditVLEncoderStep()),
        ("vae_encoder", QwenImageEditVaeEncoderStep()),
        ("input", QwenImageEditInputStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
        ("denoise", QwenImageEditDenoiseStep()),
        ("decode", QwenImageDecodeStep()),
    ]
)



### 2.1.2 edit inpaint

#### vae encoder: the difference from regular inpaint is the resize step
QwenImageEditInpaintVaeEncoderBlocks = InsertableDict([
    ("resize", QwenImageEditResizeDynamicStep()), # image -> resized_image
    ("preprocess", QwenImageInpaintProcessImagesInputStep), # resized_image, mask_image -> processed_image, processed_mask_image, mask_overlay_kwargs
    ("encode", QwenImageVaeEncoderDynamicStep(input_name="processed_image", output_name="image_latents")), # processed_image -> image_latents
])

class QwenImageEditInpaintVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditInpaintVaeEncoderBlocks.values()
    block_names = QwenImageEditInpaintVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return (
            "This step is used for processing image and mask inputs for QwenImage-Edit inpaint tasks. It:\n"
            " - resize the image for target area (1024 * 1024) while maintaining the aspect ratio.\n"
            " - process the resized image and mask image.\n"
            " - create image latents."
        )


EDIT_INPAINT_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditVLEncoderStep()),
        ("vae_encoder", QwenImageEditInpaintVaeEncoderStep()),
        ("input", QwenImageInpaintInputStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
        ("denoise", QwenImageEditInpaintDenoiseStep()),
        ("decode", QwenImageInpaintDecodeStep()),
    ]
)


## 2.2 auto blocks

### 2.2.1 encoders

class QwenImageEditAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [
        QwenImageEditInpaintVaeEncoderStep,
        QwenImageEditVaeEncoderStep,
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


### 2.2.2 inputs
class QwenImageEditAutoInputStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintInputStep, QwenImageEditInputStep]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["processed_mask_image", "image"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the edit denoising step.\n"
            + " It is an auto pipeline block that works for edit and edit_inpaint tasks.\n"
            + " - `QwenImageInpaintInputStep` (edit_inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageEditInputStep` (edit) is used when `image_latents` is provided.\n"
            + " - if `processed_mask_image` or `image_latents` is not provided, step will be skipped."
        )

### 2.2.3 before denoise

#### edit
QwenImageEditBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep())
    ]
)

#### edit inpaint
QwenImageEditInpaintBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
    ]
)


class QwenImageEditAutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        SequentialPipelineBlocks.from_blocks_dict(QwenImageEditInpaintBeforeDenoiseBlocks),
        SequentialPipelineBlocks.from_blocks_dict(QwenImageEditBeforeDenoiseBlocks),
    ]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["processed_mask_image", "image_latents"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step.\n"
            + "This is an auto pipeline block that works for edit (img2img) and edit inpaint tasks.\n"
            + " - to run the edit_inpaint workflow, make sure `processed_mask_image` is provided.\n"
            + " - edit workflow will run when `image_latents` is provided and `processed_mask_image` is not provided.\n"
            + " - if `image_latents` or `processed_mask_image` is not provided, step will be skipped."
        )


### 2.2.4 denoise

class QwenImageEditAutoDenoiseStep(AutoPipelineBlocks):
    model_name = "qwenimage-edit"

    block_classes = [QwenImageEditInpaintDenoiseStep, QwenImageEditDenoiseStep]
    block_names = ["inpaint_denoise", "denoise"]
    block_trigger_inputs = ["processed_mask_image", "image_latents"]

    @property
    def description(self):
        return (
            "Denoise step that iteratively denoise the latents. \n"
            + "This block supports edit (img2img) and edit inpaint tasks for QwenImage Edit."
            + " - `QwenImageEditInpaintDenoiseStep` (inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageEditDenoiseStep` (img2img) is used when `image_latents` is provided.\n"
            + " - if `processed_mask_image` or `image_latents` is not provided, step will be skipped."
        )

### 2.3 auto blocks & presets

EDIT_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditVLEncoderStep()),
        ("vae_encoder", QwenImageEditAutoVaeEncoderStep()),
        ("input", QwenImageEditAutoInputStep()),
        ("before_denoise", QwenImageEditAutoBeforeDenoiseStep()),
        ("denoise", QwenImageEditAutoDenoiseStep()),
        ("decode", QwenImageAutoDecodeStep()),
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
            + "- for edit (img2img) generation, you need to provide `image`\n"
            + "- for edit inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop` \n"
        )


# 3. all block presets supported in QwenImage & QwenImage-Edit


ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "edit": EDIT_BLOCKS,
    "edit_inpaint": EDIT_INPAINT_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
    "auto": AUTO_BLOCKS,
    "edit_auto": EDIT_AUTO_BLOCKS,
}
