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


# auto_docstring
class QwenImageEditVLEncoderStep(SequentialPipelineBlocks):
    """
    class QwenImageEditVLEncoderStep

      QwenImage-Edit VL encoder step that encode the image and text prompts together.

      Components:

          image_resize_processor (`VaeImageProcessor`) [subfolder=]

          text_encoder (`Qwen2_5_VLForConditionalGeneration`) [subfolder=]

          processor (`Qwen2VLProcessor`) [subfolder=]

          guider (`ClassifierFreeGuidance`) [subfolder=]

      Configs:

          prompt_template_encode (default: <|im_start|>system
    Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how
    the user's text instruction should alter or modify the image. Generate a new image that meets the user's
    requirements while maintaining consistency with the original input where appropriate.<|im_end|> <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>{}<|im_end|> <|im_start|>assistant )

          prompt_template_encode_start_idx (default: 64)

      Inputs:

          image (`Image`):
              Input image for img2img, editing, or conditioning.

          prompt (`str`):
              The prompt or prompts to guide image generation.

          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.

      Outputs:

          resized_image (`List`):
              The resized images

          prompt_embeds (`Tensor`):
              The prompt embeddings

          prompt_embeds_mask (`Tensor`):
              The encoder attention mask

          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings

          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask
    """

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
# auto_docstring
class QwenImageEditVaeEncoderStep(SequentialPipelineBlocks):
    """
    class QwenImageEditVaeEncoderStep

      Vae encoder step that encode the image inputs into their latent representations.

      Components:

          image_resize_processor (`VaeImageProcessor`) [subfolder=]

          image_processor (`VaeImageProcessor`) [subfolder=]

          vae (`AutoencoderKLQwenImage`) [subfolder=]

      Inputs:

          image (`Image`):
              Input image for img2img, editing, or conditioning.

          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:

          resized_image (`List`):
              The resized images

          processed_image (`None`):

          image_latents (`Tensor`):
              The latents representing the reference image(s). Single tensor or list depending on input.
    """

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
# auto_docstring
class QwenImageEditInpaintVaeEncoderStep(SequentialPipelineBlocks):
    """
    class QwenImageEditInpaintVaeEncoderStep

      This step is used for processing image and mask inputs for QwenImage-Edit inpaint tasks. It:
       - resize the image for target area (1024 * 1024) while maintaining the aspect ratio.
       - process the resized image and mask image.
       - create image latents.

      Components:

          image_resize_processor (`VaeImageProcessor`) [subfolder=]

          image_mask_processor (`InpaintProcessor`) [subfolder=]

          vae (`AutoencoderKLQwenImage`) [subfolder=]

      Inputs:

          image (`Image`):
              Input image for img2img, editing, or conditioning.

          mask_image (`Image`):
              Mask image for inpainting.

          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.

          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:

          resized_image (`List`):
              The resized images

          processed_image (`None`):

          processed_mask_image (`None`):

          mask_overlay_kwargs (`Dict`):
              The kwargs for the postprocess step to apply the mask overlay

          image_latents (`Tensor`):
              The latents representing the reference image(s). Single tensor or list depending on input.
    """

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
# auto_docstring
class QwenImageEditInputStep(SequentialPipelineBlocks):
    """
    class QwenImageEditInputStep

      Input step that prepares the inputs for the edit denoising step. It:
       - make sure the text embeddings have consistent batch size as well as the additional inputs.
       - update height/width based `image_latents`, patchify `image_latents`.

      Components:

          pachifier (`QwenImagePachifier`) [subfolder=]

      Inputs:

          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.

          prompt_embeds (`None`):

          prompt_embeds_mask (`None`):

          negative_prompt_embeds (`None`, *optional*):

          negative_prompt_embeds_mask (`None`, *optional*):

          height (`int`, *optional*):
              The height in pixels of the generated image.

          width (`int`, *optional*):
              The width in pixels of the generated image.

          image_latents (`None`, *optional*):

      Outputs:

          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt

          dtype (`dtype`):
              Data type of model tensor inputs (determined by `prompt_embeds`)

          image_height (`int`):
              The image height calculated from the image latents dimension

          image_width (`int`):
              The image width calculated from the image latents dimension
    """

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


# auto_docstring
class QwenImageEditInpaintInputStep(SequentialPipelineBlocks):
    """
    class QwenImageEditInpaintInputStep

      Input step that prepares the inputs for the edit inpaint denoising step. It:
       - make sure the text embeddings have consistent batch size as well as the additional inputs.
       - update height/width based `image_latents`, patchify `image_latents`.

      Components:

          pachifier (`QwenImagePachifier`) [subfolder=]

      Inputs:

          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.

          prompt_embeds (`None`):

          prompt_embeds_mask (`None`):

          negative_prompt_embeds (`None`, *optional*):

          negative_prompt_embeds_mask (`None`, *optional*):

          height (`int`, *optional*):
              The height in pixels of the generated image.

          width (`int`, *optional*):
              The width in pixels of the generated image.

          image_latents (`None`, *optional*):

          processed_mask_image (`None`, *optional*):

      Outputs:

          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt

          dtype (`dtype`):
              Data type of model tensor inputs (determined by `prompt_embeds`)

          image_height (`int`):
              The image height calculated from the image latents dimension

          image_width (`int`):
              The image width calculated from the image latents dimension
    """

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
# auto_docstring
class QwenImageEditInpaintPrepareLatentsStep(SequentialPipelineBlocks):
    """
    class QwenImageEditInpaintPrepareLatentsStep

      This step prepares the latents/image_latents and mask inputs for the edit inpainting denoising step. It:
       - Add noise to the image latents to create the latents input for the denoiser.
       - Create the patchified latents `mask` based on the processed mask image.

      Components:

          scheduler (`FlowMatchEulerDiscreteScheduler`) [subfolder=]

          pachifier (`QwenImagePachifier`) [subfolder=]

      Inputs:

          latents (`Tensor`):
              The initial random noised, can be generated in prepare latent step.

          image_latents (`Tensor`):
              The image latents to use for the denoising process. Can be generated in vae encoder and packed in input
              step.

          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.

          processed_mask_image (`Tensor`):
              The processed mask to use for the inpainting process.

          height (`None`):

          width (`None`):

          dtype (`None`):

      Outputs:

          initial_noise (`Tensor`):
              The initial random noised used for inpainting denoising.

          mask (`Tensor`):
              The mask to use for the inpainting process.
    """

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
# auto_docstring
class QwenImageEditCoreDenoiseStep(SequentialPipelineBlocks):
    """
    class QwenImageEditCoreDenoiseStep

      Core denoising workflow for QwenImage-Edit edit (img2img) task.

      Components:

          pachifier (`QwenImagePachifier`) [subfolder=]

          scheduler (`FlowMatchEulerDiscreteScheduler`) [subfolder=]

          guider (`ClassifierFreeGuidance`) [subfolder=]

          transformer (`QwenImageTransformer2DModel`) [subfolder=]

      Inputs:

          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.

          prompt_embeds (`None`):

          prompt_embeds_mask (`None`):

          negative_prompt_embeds (`None`, *optional*):

          negative_prompt_embeds_mask (`None`, *optional*):

          height (`int`, *optional*):
              The height in pixels of the generated image.

          width (`int`, *optional*):
              The width in pixels of the generated image.

          image_latents (`None`, *optional*):

          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.

          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.

          sigmas (`List`, *optional*):
              Custom sigmas for the denoising process.

          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.

          **denoiser_input_fields (`Tensor`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:

          latents (`Tensor`):
              Denoised latents.
    """

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

    @property
    def outputs(self):
        return [
            OutputParam.latents(),
        ]


# Qwen Image Edit (inpainting) core denoise step
# auto_docstring
class QwenImageEditInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    """
    class QwenImageEditInpaintCoreDenoiseStep

      Core denoising workflow for QwenImage-Edit edit inpaint task.

      Components:

          pachifier (`QwenImagePachifier`) [subfolder=]

          scheduler (`FlowMatchEulerDiscreteScheduler`) [subfolder=]

          guider (`ClassifierFreeGuidance`) [subfolder=]

          transformer (`QwenImageTransformer2DModel`) [subfolder=]

      Inputs:

          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.

          prompt_embeds (`None`):

          prompt_embeds_mask (`None`):

          negative_prompt_embeds (`None`, *optional*):

          negative_prompt_embeds_mask (`None`, *optional*):

          height (`int`, *optional*):
              The height in pixels of the generated image.

          width (`int`, *optional*):
              The width in pixels of the generated image.

          image_latents (`None`, *optional*):

          processed_mask_image (`None`, *optional*):

          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.

          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.

          sigmas (`List`, *optional*):
              Custom sigmas for the denoising process.

          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.

          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.

          **denoiser_input_fields (`Tensor`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:

          latents (`Tensor`):
              Denoised latents.
    """

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

    @property
    def outputs(self):
        return [
            OutputParam.latents(),
        ]


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

    @property
    def outputs(self):
        return [
            OutputParam.latents(),
        ]


# ====================
# 4. DECODE
# ====================


# Decode step (standard)
# auto_docstring
class QwenImageEditDecodeStep(SequentialPipelineBlocks):
    """
    class QwenImageEditDecodeStep

      Decode step that decodes the latents to images and postprocess the generated image.

      Components:

          vae (`AutoencoderKLQwenImage`) [subfolder=]

          image_processor (`VaeImageProcessor`) [subfolder=]

      Inputs:

          latents (`Tensor`):
              The latents to decode, can be generated in the denoise step

          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt''.

      Outputs:

          images (`List`):
              Generated images.
    """

    model_name = "qwenimage-edit"
    block_classes = [QwenImageDecoderStep(), QwenImageProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image."


# Inpaint decode step
# auto_docstring
class QwenImageEditInpaintDecodeStep(SequentialPipelineBlocks):
    """
    class QwenImageEditInpaintDecodeStep

      Decode step that decodes the latents to images and postprocess the generated image, optionally apply the mask
      overlay to the original image.

      Components:

          vae (`AutoencoderKLQwenImage`) [subfolder=]

          image_mask_processor (`InpaintProcessor`) [subfolder=]

      Inputs:

          latents (`Tensor`):
              The latents to decode, can be generated in the denoise step

          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt''.

          mask_overlay_kwargs (`None`, *optional*):

      Outputs:

          images (`List`):
              Generated images.
    """

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
            OutputParam.latents(),
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


# auto_docstring
class QwenImageEditAutoBlocks(SequentialPipelineBlocks):
    """
    class QwenImageEditAutoBlocks

      Auto Modular pipeline for edit (img2img) and edit inpaint tasks using QwenImage-Edit.
      - for edit (img2img) generation, you need to provide `image`
      - for edit inpainting, you need to provide `mask_image` and `image`, optionally you can provide
        `padding_mask_crop`

      Components:

          image_resize_processor (`VaeImageProcessor`) [subfolder=]

          text_encoder (`Qwen2_5_VLForConditionalGeneration`) [subfolder=]

          processor (`Qwen2VLProcessor`) [subfolder=]

          guider (`ClassifierFreeGuidance`) [subfolder=]

          image_mask_processor (`InpaintProcessor`) [subfolder=]

          vae (`AutoencoderKLQwenImage`) [subfolder=]

          image_processor (`VaeImageProcessor`) [subfolder=]

          pachifier (`QwenImagePachifier`) [subfolder=]

          scheduler (`FlowMatchEulerDiscreteScheduler`) [subfolder=]

          transformer (`QwenImageTransformer2DModel`) [subfolder=]

      Configs:

          prompt_template_encode (default: <|im_start|>system
    Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how
    the user's text instruction should alter or modify the image. Generate a new image that meets the user's
    requirements while maintaining consistency with the original input where appropriate.<|im_end|> <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>{}<|im_end|> <|im_start|>assistant )

          prompt_template_encode_start_idx (default: 64)

      Inputs:

          image (`Image`):
              Input image for img2img, editing, or conditioning.

          prompt (`str`):
              The prompt or prompts to guide image generation.

          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.

          mask_image (`Image`, *optional*):
              Mask image for inpainting.

          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.

          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.

          height (`int`):
              The height in pixels of the generated image.

          width (`int`):
              The width in pixels of the generated image.

          image_latents (`None`):

          processed_mask_image (`None`, *optional*):

          latents (`Tensor`):
              Pre-generated noisy latents for image generation.

          num_inference_steps (`int`):
              The number of denoising steps.

          sigmas (`List`, *optional*):
              Custom sigmas for the denoising process.

          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.

          attention_kwargs (`Dict`, *optional*):
              Additional kwargs for attention processors.

          **denoiser_input_fields (`Tensor`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt''.

          mask_overlay_kwargs (`None`, *optional*):

      Outputs:

          images (`List`):
              Generated images.
    """

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
            OutputParam.images(),
        ]
