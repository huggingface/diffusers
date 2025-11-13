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
    QwenImageCreateMaskLatentsStep,
    QwenImageEditRoPEInputsStep,
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
    QwenImageControlNetVaeEncoderStep,
    QwenImageEditPlusProcessImagesInputStep,
    QwenImageEditPlusResizeDynamicStep,
    QwenImageEditPlusTextEncoderStep,
    QwenImageEditResizeDynamicStep,
    QwenImageEditTextEncoderStep,
    QwenImageInpaintProcessImagesInputStep,
    QwenImageProcessImagesInputStep,
    QwenImageTextEncoderStep,
    QwenImageVaeEncoderDynamicStep,
)
from .inputs import QwenImageControlNetInputsStep, QwenImageInputsDynamicStep, QwenImageTextInputsStep


logger = logging.get_logger(__name__)

# 1. QwenImage

## 1.1 QwenImage/text2image

#### QwenImage/decode
#### (standard decode step works for most tasks except for inpaint)
QwenImageDecodeBlocks = InsertableDict(
    [
        ("decode", QwenImageDecoderStep()),
        ("postprocess", QwenImageProcessImagesOutputStep()),
    ]
)


class QwenImageDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageDecodeBlocks.values()
    block_names = QwenImageDecodeBlocks.keys()

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image."


#### QwenImage/text2image presets
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


## 1.2 QwenImage/inpaint

#### QwenImage/inpaint vae encoder
QwenImageInpaintVaeEncoderBlocks = InsertableDict(
    [
        (
            "preprocess",
            QwenImageInpaintProcessImagesInputStep,
        ),  # image, mask_image -> processed_image, processed_mask_image, mask_overlay_kwargs
        ("encode", QwenImageVaeEncoderDynamicStep()),  # processed_image -> image_latents
    ]
)


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


#### QwenImage/inpaint inputs
QwenImageInpaintInputBlocks = InsertableDict(
    [
        ("text_inputs", QwenImageTextInputsStep()),  # default step to process text embeddings
        (
            "additional_inputs",
            QwenImageInputsDynamicStep(
                image_latent_inputs=["image_latents"], additional_batch_inputs=["processed_mask_image"]
            ),
        ),
    ]
)


class QwenImageInpaintInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageInpaintInputBlocks.values()
    block_names = QwenImageInpaintInputBlocks.keys()

    @property
    def description(self):
        return "Input step that prepares the inputs for the inpainting denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents` and `processed_mask_image`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


# QwenImage/inpaint prepare latents
QwenImageInpaintPrepareLatentsBlocks = InsertableDict(
    [
        ("add_noise_to_latents", QwenImagePrepareLatentsWithStrengthStep()),
        ("create_mask_latents", QwenImageCreateMaskLatentsStep()),
    ]
)


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


#### QwenImage/inpaint decode
QwenImageInpaintDecodeBlocks = InsertableDict(
    [
        ("decode", QwenImageDecoderStep()),
        ("postprocess", QwenImageInpaintProcessImagesOutputStep()),
    ]
)


class QwenImageInpaintDecodeStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageInpaintDecodeBlocks.values()
    block_names = QwenImageInpaintDecodeBlocks.keys()

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image, optional apply the mask overally to the original image."


#### QwenImage/inpaint presets
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


## 1.3 QwenImage/img2img

#### QwenImage/img2img vae encoder
QwenImageImg2ImgVaeEncoderBlocks = InsertableDict(
    [
        ("preprocess", QwenImageProcessImagesInputStep()),
        ("encode", QwenImageVaeEncoderDynamicStep()),
    ]
)


class QwenImageImg2ImgVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = QwenImageImg2ImgVaeEncoderBlocks.values()
    block_names = QwenImageImg2ImgVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


#### QwenImage/img2img inputs
QwenImageImg2ImgInputBlocks = InsertableDict(
    [
        ("text_inputs", QwenImageTextInputsStep()),  # default step to process text embeddings
        ("additional_inputs", QwenImageInputsDynamicStep(image_latent_inputs=["image_latents"])),
    ]
)


class QwenImageImg2ImgInputStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageImg2ImgInputBlocks.values()
    block_names = QwenImageImg2ImgInputBlocks.keys()

    @property
    def description(self):
        return "Input step that prepares the inputs for the img2img denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


#### QwenImage/img2img presets
IMAGE2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageImg2ImgVaeEncoderStep()),
        ("input", QwenImageImg2ImgInputStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_img2img_latents", QwenImagePrepareLatentsWithStrengthStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
        ("denoise", QwenImageDenoiseStep()),
        ("decode", QwenImageDecodeStep()),
    ]
)


## 1.4 QwenImage/controlnet

#### QwenImage/controlnet presets
CONTROLNET_BLOCKS = InsertableDict(
    [
        ("controlnet_vae_encoder", QwenImageControlNetVaeEncoderStep()),  # vae encoder step for control_image
        ("controlnet_inputs", QwenImageControlNetInputsStep()),  # additional input step for controlnet
        (
            "controlnet_before_denoise",
            QwenImageControlNetBeforeDenoiserStep(),
        ),  # before denoise step (after set_timesteps step)
        (
            "controlnet_denoise_loop_before",
            QwenImageLoopBeforeDenoiserControlNet(),
        ),  # controlnet loop step (insert before the denoiseloop_denoiser)
    ]
)


## 1.5 QwenImage/auto encoders


#### for inpaint and img2img tasks
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


# for controlnet tasks
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


## 1.6 QwenImage/auto inputs


# text2image/inpaint/img2img
class QwenImageAutoInputStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintInputStep, QwenImageImg2ImgInputStep, QwenImageTextInputsStep]
    block_names = ["inpaint", "img2img", "text2image"]
    block_trigger_inputs = ["processed_mask_image", "image_latents", None]

    @property
    def description(self):
        return (
            "Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size, and patchified. \n"
            " This is an auto pipeline block that works for text2image/inpaint/img2img tasks.\n"
            + " - `QwenImageInpaintInputStep` (inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageImg2ImgInputStep` (img2img) is used when `image_latents` is provided.\n"
            + " - `QwenImageTextInputsStep` (text2image) is used when both `processed_mask_image` and `image_latents` are not provided.\n"
        )


# controlnet
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


## 1.7 QwenImage/auto before denoise step
# compose the steps into a BeforeDenoiseStep for text2image/img2img/inpaint tasks before combine into an auto step

#  QwenImage/text2image before denoise
QwenImageText2ImageBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
    ]
)


class QwenImageText2ImageBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageText2ImageBeforeDenoiseBlocks.values()
    block_names = QwenImageText2ImageBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for text2image task."


# QwenImage/inpaint before denoise
QwenImageInpaintBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
    ]
)


class QwenImageInpaintBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageInpaintBeforeDenoiseBlocks.values()
    block_names = QwenImageInpaintBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for inpaint task."


# QwenImage/img2img before denoise
QwenImageImg2ImgBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_img2img_latents", QwenImagePrepareLatentsWithStrengthStep()),
        ("prepare_rope_inputs", QwenImageRoPEInputsStep()),
    ]
)


class QwenImageImg2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageImg2ImgBeforeDenoiseBlocks.values()
    block_names = QwenImageImg2ImgBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for img2img task."


# auto before_denoise step for text2image, inpaint, img2img tasks
class QwenImageAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        QwenImageInpaintBeforeDenoiseStep,
        QwenImageImg2ImgBeforeDenoiseStep,
        QwenImageText2ImageBeforeDenoiseStep,
    ]
    block_names = ["inpaint", "img2img", "text2image"]
    block_trigger_inputs = ["processed_mask_image", "image_latents", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step.\n"
            + "This is an auto pipeline block that works for text2img, inpainting, img2img tasks.\n"
            + " - `QwenImageInpaintBeforeDenoiseStep` (inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageImg2ImgBeforeDenoiseStep` (img2img) is used when `image_latents` is provided.\n"
            + " - `QwenImageText2ImageBeforeDenoiseStep` (text2image) is used when both `processed_mask_image` and `image_latents` are not provided.\n"
        )


# auto before_denoise step for controlnet tasks
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


## 1.8 QwenImage/auto denoise


# auto denoise step for controlnet tasks: works for all tasks with controlnet
class QwenImageControlNetAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintControlNetDenoiseStep, QwenImageControlNetDenoiseStep]
    block_names = ["inpaint_denoise", "denoise"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self):
        return (
            "Controlnet step during the denoising process. \n"
            " This is an auto pipeline block that works for inpaint and text2image/img2img tasks with controlnet.\n"
            + " - `QwenImageInpaintControlNetDenoiseStep` (inpaint) is used when `mask` is provided.\n"
            + " - `QwenImageControlNetDenoiseStep` (text2image/img2img) is used when `mask` is not provided.\n"
        )


# auto denoise step for everything: works for all tasks with or without controlnet
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
            "Denoise step that iteratively denoise the latents. \n"
            " This is an auto pipeline block that works for inpaint/text2image/img2img tasks. It also works with controlnet\n"
            + " - `QwenImageControlNetAutoDenoiseStep` (controlnet) is used when `control_image_latents` is provided.\n"
            + " - `QwenImageInpaintDenoiseStep` (inpaint) is used when `mask` is provided and `control_image_latents` is not provided.\n"
            + " - `QwenImageDenoiseStep` (text2image/img2img) is used when `mask` is not provided and `control_image_latents` is not provided.\n"
        )


## 1.9 QwenImage/auto decode
# auto decode step for inpaint and text2image tasks


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


class QwenImageCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = [
        QwenImageAutoInputStep,
        QwenImageOptionalControlNetInputStep,
        QwenImageAutoBeforeDenoiseStep,
        QwenImageOptionalControlNetBeforeDenoiseStep,
        QwenImageAutoDenoiseStep,
    ]
    block_names = ["input", "controlnet_input", "before_denoise", "controlnet_before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `QwenImageAutoInputStep` (input) standardizes the inputs for the denoising step.\n"
            + " - `QwenImageOptionalControlNetInputStep` (controlnet_input) prepares the controlnet input.\n"
            + " - `QwenImageAutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            + " - `QwenImageOptionalControlNetBeforeDenoiseStep` (controlnet_before_denoise) prepares the controlnet input for the denoising step.\n"
            + " - `QwenImageAutoDenoiseStep` (denoise) iteratively denoises the latents.\n"
            + "This step support text-to-image, image-to-image, inpainting, and controlnet tasks for QwenImage:\n"
            + " - for image-to-image generation, you need to provide `image_latents`\n"
            + " - for inpainting, you need to provide `processed_mask_image` and `image_latents`\n"
            + " - to run the controlnet workflow, you need to provide `control_image_latents`\n"
            + " - for text-to-image generation, all you need to provide is prompt embeddings"
        )


## 1.10 QwenImage/auto block & presets
AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageAutoVaeEncoderStep()),
        ("controlnet_vae_encoder", QwenImageOptionalControlNetVaeEncoderStep()),
        ("denoise", QwenImageCoreDenoiseStep()),
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

## 2.1 QwenImage-Edit/edit

#### QwenImage-Edit/edit vl encoder: take both image and text prompts
QwenImageEditVLEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImageEditResizeDynamicStep()),
        ("encode", QwenImageEditTextEncoderStep()),
    ]
)


class QwenImageEditVLEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditVLEncoderBlocks.values()
    block_names = QwenImageEditVLEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "QwenImage-Edit VL encoder step that encode the image an text prompts together."


#### QwenImage-Edit/edit vae encoder
QwenImageEditVaeEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImageEditResizeDynamicStep()),  # edit has a different resize step
        ("preprocess", QwenImageProcessImagesInputStep()),  # resized_image -> processed_image
        ("encode", QwenImageVaeEncoderDynamicStep()),  # processed_image -> image_latents
    ]
)


class QwenImageEditVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditVaeEncoderBlocks.values()
    block_names = QwenImageEditVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Vae encoder step that encode the image inputs into their latent representations."


#### QwenImage-Edit/edit input
QwenImageEditInputBlocks = InsertableDict(
    [
        ("text_inputs", QwenImageTextInputsStep()),  # default step to process text embeddings
        ("additional_inputs", QwenImageInputsDynamicStep(image_latent_inputs=["image_latents"])),
    ]
)


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


#### QwenImage/edit presets
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


## 2.2 QwenImage-Edit/edit inpaint

#### QwenImage-Edit/edit inpaint vae encoder: the difference from regular inpaint is the resize step
QwenImageEditInpaintVaeEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImageEditResizeDynamicStep()),  # image -> resized_image
        (
            "preprocess",
            QwenImageInpaintProcessImagesInputStep,
        ),  # resized_image, mask_image -> processed_image, processed_mask_image, mask_overlay_kwargs
        (
            "encode",
            QwenImageVaeEncoderDynamicStep(input_name="processed_image", output_name="image_latents"),
        ),  # processed_image -> image_latents
    ]
)


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


#### QwenImage-Edit/edit inpaint presets
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


## 2.3 QwenImage-Edit/auto encoders


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
            "Vae encoder step that encode the image inputs into their latent representations. \n"
            " This is an auto pipeline block that works for edit and edit_inpaint tasks.\n"
            + " - `QwenImageEditInpaintVaeEncoderStep` (edit_inpaint) is used when `mask_image` is provided.\n"
            + " - `QwenImageEditVaeEncoderStep` (edit) is used when `image` is provided.\n"
            + " - if `mask_image` or `image` is not provided, step will be skipped."
        )


## 2.4 QwenImage-Edit/auto inputs
class QwenImageEditAutoInputStep(AutoPipelineBlocks):
    block_classes = [QwenImageInpaintInputStep, QwenImageEditInputStep]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["processed_mask_image", "image_latents"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the edit denoising step.\n"
            + " It is an auto pipeline block that works for edit and edit_inpaint tasks.\n"
            + " - `QwenImageInpaintInputStep` (edit_inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageEditInputStep` (edit) is used when `image_latents` is provided.\n"
            + " - if `processed_mask_image` or `image_latents` is not provided, step will be skipped."
        )


## 2.5 QwenImage-Edit/auto before denoise
# compose the steps into a BeforeDenoiseStep for edit and edit_inpaint tasks before combine into an auto step

#### QwenImage-Edit/edit before denoise
QwenImageEditBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
    ]
)


class QwenImageEditBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditBeforeDenoiseBlocks.values()
    block_names = QwenImageEditBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for edit task."


#### QwenImage-Edit/edit inpaint before denoise
QwenImageEditInpaintBeforeDenoiseBlocks = InsertableDict(
    [
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
    ]
)


class QwenImageEditInpaintBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditInpaintBeforeDenoiseBlocks.values()
    block_names = QwenImageEditInpaintBeforeDenoiseBlocks.keys()

    @property
    def description(self):
        return "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for edit inpaint task."


# auto before_denoise step for edit and edit_inpaint tasks
class QwenImageEditAutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditInpaintBeforeDenoiseStep,
        QwenImageEditBeforeDenoiseStep,
    ]
    block_names = ["edit_inpaint", "edit"]
    block_trigger_inputs = ["processed_mask_image", "image_latents"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step.\n"
            + "This is an auto pipeline block that works for edit (img2img) and edit inpaint tasks.\n"
            + " - `QwenImageEditInpaintBeforeDenoiseStep` (edit_inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageEditBeforeDenoiseStep` (edit) is used when `image_latents` is provided and `processed_mask_image` is not provided.\n"
            + " - if `image_latents` or `processed_mask_image` is not provided, step will be skipped."
        )


## 2.6 QwenImage-Edit/auto denoise


class QwenImageEditAutoDenoiseStep(AutoPipelineBlocks):
    model_name = "qwenimage-edit"

    block_classes = [QwenImageEditInpaintDenoiseStep, QwenImageEditDenoiseStep]
    block_names = ["inpaint_denoise", "denoise"]
    block_trigger_inputs = ["processed_mask_image", "image_latents"]

    @property
    def description(self):
        return (
            "Denoise step that iteratively denoise the latents. \n"
            + "This block supports edit (img2img) and edit inpaint tasks for QwenImage Edit. \n"
            + " - `QwenImageEditInpaintDenoiseStep` (inpaint) is used when `processed_mask_image` is provided.\n"
            + " - `QwenImageEditDenoiseStep` (img2img) is used when `image_latents` is provided.\n"
            + " - if `processed_mask_image` or `image_latents` is not provided, step will be skipped."
        )


## 2.7 QwenImage-Edit/auto blocks & presets


class QwenImageEditCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditAutoInputStep,
        QwenImageEditAutoBeforeDenoiseStep,
        QwenImageEditAutoDenoiseStep,
    ]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `QwenImageEditAutoInputStep` (input) standardizes the inputs for the denoising step.\n"
            + " - `QwenImageEditAutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            + " - `QwenImageEditAutoDenoiseStep` (denoise) iteratively denoises the latents.\n\n"
            + "This step support edit (img2img) and edit inpainting workflow for QwenImage Edit:\n"
            + " - When `processed_mask_image` is provided, it will be used for edit inpainting task.\n"
            + " - When `image_latents` is provided, it will be used for edit (img2img) task.\n"
        )


EDIT_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditVLEncoderStep()),
        ("vae_encoder", QwenImageEditAutoVaeEncoderStep()),
        ("denoise", QwenImageEditCoreDenoiseStep()),
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


#################### QwenImage Edit Plus #####################

# 3. QwenImage-Edit Plus

## 3.1 QwenImage-Edit Plus / edit

#### QwenImage-Edit Plus vl encoder: take both image and text prompts
QwenImageEditPlusVLEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImageEditPlusResizeDynamicStep()),
        ("encode", QwenImageEditPlusTextEncoderStep()),
    ]
)


class QwenImageEditPlusVLEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditPlusVLEncoderBlocks.values()
    block_names = QwenImageEditPlusVLEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "QwenImage-Edit Plus VL encoder step that encode the image an text prompts together."


#### QwenImage-Edit Plus vae encoder
QwenImageEditPlusVaeEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImageEditPlusResizeDynamicStep()),  # edit plus has a different resize step
        ("preprocess", QwenImageEditPlusProcessImagesInputStep()),  # vae_image -> processed_image
        ("encode", QwenImageVaeEncoderDynamicStep()),  # processed_image -> image_latents
    ]
)


class QwenImageEditPlusVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    block_classes = QwenImageEditPlusVaeEncoderBlocks.values()
    block_names = QwenImageEditPlusVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Vae encoder step that encode the image inputs into their latent representations."


#### QwenImage Edit Plus presets
EDIT_PLUS_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditPlusVLEncoderStep()),
        ("vae_encoder", QwenImageEditPlusVaeEncoderStep()),
        ("input", QwenImageEditInputStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImageEditRoPEInputsStep()),
        ("denoise", QwenImageEditDenoiseStep()),
        ("decode", QwenImageDecodeStep()),
    ]
)


# auto before_denoise step for edit tasks
class QwenImageEditPlusAutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "qwenimage-edit-plus"
    block_classes = [QwenImageEditBeforeDenoiseStep]
    block_names = ["edit"]
    block_trigger_inputs = ["image_latents"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step.\n"
            + "This is an auto pipeline block that works for edit (img2img) task.\n"
            + " - `QwenImageEditBeforeDenoiseStep` (edit) is used when `image_latents` is provided and `processed_mask_image` is not provided.\n"
            + " - if `image_latents` is not provided, step will be skipped."
        )


## 3.2 QwenImage-Edit Plus/auto encoders


class QwenImageEditPlusAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [
        QwenImageEditPlusVaeEncoderStep,
    ]
    block_names = ["edit"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations. \n"
            " This is an auto pipeline block that works for edit task.\n"
            + " - `QwenImageEditPlusVaeEncoderStep` (edit) is used when `image` is provided.\n"
            + " - if `image` is not provided, step will be skipped."
        )


## 3.3 QwenImage-Edit/auto blocks & presets


class QwenImageEditPlusCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage-edit-plus"
    block_classes = [
        QwenImageEditAutoInputStep,
        QwenImageEditPlusAutoBeforeDenoiseStep,
        QwenImageEditAutoDenoiseStep,
    ]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `QwenImageEditAutoInputStep` (input) standardizes the inputs for the denoising step.\n"
            + " - `QwenImageEditPlusAutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            + " - `QwenImageEditAutoDenoiseStep` (denoise) iteratively denoises the latents.\n\n"
            + "This step support edit (img2img) workflow for QwenImage Edit Plus:\n"
            + " - When `image_latents` is provided, it will be used for edit (img2img) task.\n"
        )


EDIT_PLUS_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditPlusVLEncoderStep()),
        ("vae_encoder", QwenImageEditPlusAutoVaeEncoderStep()),
        ("denoise", QwenImageEditPlusCoreDenoiseStep()),
        ("decode", QwenImageAutoDecodeStep()),
    ]
)


class QwenImageEditPlusAutoBlocks(SequentialPipelineBlocks):
    model_name = "qwenimage-edit-plus"
    block_classes = EDIT_PLUS_AUTO_BLOCKS.values()
    block_names = EDIT_PLUS_AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Auto Modular pipeline for edit (img2img) and edit tasks using QwenImage-Edit Plus.\n"
            + "- for edit (img2img) generation, you need to provide `image`\n"
        )


# 3. all block presets supported in QwenImage, QwenImage-Edit, QwenImage-Edit Plus


ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "img2img": IMAGE2IMAGE_BLOCKS,
    "edit": EDIT_BLOCKS,
    "edit_inpaint": EDIT_INPAINT_BLOCKS,
    "edit_plus": EDIT_PLUS_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
    "auto": AUTO_BLOCKS,
    "edit_auto": EDIT_AUTO_BLOCKS,
    "edit_plus_auto": EDIT_PLUS_AUTO_BLOCKS,
}
