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


import torch

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InputParam, InsertableDict, OutputParam
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
    QwenImage-Edit VL encoder step that encode the image and text prompts together.

      Components:
          image_resize_processor (`VaeImageProcessor`) text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor
          (`Qwen2VLProcessor`) guider (`ClassifierFreeGuidance`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.

      Outputs:
          resized_image (`list`):
              The resized images
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
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
    Vae encoder step that encode the image inputs into their latent representations.

      Components:
          image_resize_processor (`VaeImageProcessor`) image_processor (`VaeImageProcessor`) vae
          (`AutoencoderKLQwenImage`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          resized_image (`list`):
              The resized images
          processed_image (`Tensor`):
              The processed image
          image_latents (`Tensor`):
              The latent representation of the input image.
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
    This step is used for processing image and mask inputs for QwenImage-Edit inpaint tasks. It:
       - resize the image for target area (1024 * 1024) while maintaining the aspect ratio.
       - process the resized image and mask image.
       - create image latents.

      Components:
          image_resize_processor (`VaeImageProcessor`) image_mask_processor (`InpaintProcessor`) vae
          (`AutoencoderKLQwenImage`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          mask_image (`Image`):
              Mask image for inpainting.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          resized_image (`list`):
              The resized images
          processed_image (`Tensor`):
              The processed image
          processed_mask_image (`Tensor`):
              The processed mask image
          mask_overlay_kwargs (`dict`):
              The kwargs for the postprocess step to apply the mask overlay
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageEditResizeStep(),
        QwenImageEditInpaintProcessImagesInputStep(),
        QwenImageVaeEncoderStep(),
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
    Input step that prepares the inputs for the edit denoising step. It:
       - make sure the text embeddings have consistent batch size as well as the additional inputs.
       - update height/width based `image_latents`, patchify `image_latents`.

      Components:
          pachifier (`QwenImagePachifier`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.

      Outputs:
          batch_size (`int`):
              The batch size of the prompt embeddings
          dtype (`dtype`):
              The data type of the prompt embeddings
          prompt_embeds (`Tensor`):
              The prompt embeddings. (batch-expanded)
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask. (batch-expanded)
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings. (batch-expanded)
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask. (batch-expanded)
          image_height (`int`):
              The image height calculated from the image latents dimension
          image_width (`int`):
              The image width calculated from the image latents dimension
          height (`int`):
              if not provided, updated to image height
          width (`int`):
              if not provided, updated to image width
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (patchified and
              batch-expanded)
    """

    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageAdditionalInputsStep(),
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
    Input step that prepares the inputs for the edit inpaint denoising step. It:
       - make sure the text embeddings have consistent batch size as well as the additional inputs.
       - update height/width based `image_latents`, patchify `image_latents`.

      Components:
          pachifier (`QwenImagePachifier`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The processed mask image

      Outputs:
          batch_size (`int`):
              The batch size of the prompt embeddings
          dtype (`dtype`):
              The data type of the prompt embeddings
          prompt_embeds (`Tensor`):
              The prompt embeddings. (batch-expanded)
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask. (batch-expanded)
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings. (batch-expanded)
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask. (batch-expanded)
          image_height (`int`):
              The image height calculated from the image latents dimension
          image_width (`int`):
              The image width calculated from the image latents dimension
          height (`int`):
              if not provided, updated to image height
          width (`int`):
              if not provided, updated to image width
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (patchified and
              batch-expanded)
          processed_mask_image (`Tensor`):
              The processed mask image (batch-expanded)
    """

    model_name = "qwenimage-edit"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageAdditionalInputsStep(
            additional_batch_inputs=[
                InputParam(name="processed_mask_image", type_hint=torch.Tensor, description="The processed mask image")
            ]
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
    This step prepares the latents/image_latents and mask inputs for the edit inpainting denoising step. It:
       - Add noise to the image latents to create the latents input for the denoiser.
       - Create the patchified latents `mask` based on the processed mask image.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) pachifier (`QwenImagePachifier`)

      Inputs:
          latents (`Tensor`):
              The initial random noised, can be generated in prepare latent step.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (Can be
              generated from vae encoder and updated in input step.)
          timesteps (`Tensor`):
              The timesteps to use for the denoising process. Can be generated in set_timesteps step.
          processed_mask_image (`Tensor`):
              The processed mask to use for the inpainting process.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          initial_noise (`Tensor`):
              The initial random noised used for inpainting denoising.
          latents (`Tensor`):
              The scaled noisy latents to use for inpainting/image-to-image denoising.
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
    Core denoising workflow for QwenImage-Edit edit (img2img) task.

      Components:
          pachifier (`QwenImagePachifier`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
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
            OutputParam.template("latents"),
        ]


# Qwen Image Edit (inpainting) core denoise step
# auto_docstring
class QwenImageEditInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for QwenImage-Edit edit inpaint task.

      Components:
          pachifier (`QwenImagePachifier`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The processed mask image
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
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
            OutputParam.template("latents"),
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

    def select_block(self, processed_mask_image=None, image_latents=None) -> str | None:
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
            OutputParam.template("latents"),
        ]


# ====================
# 4. DECODE
# ====================


# Decode step (standard)
# auto_docstring
class QwenImageEditDecodeStep(SequentialPipelineBlocks):
    """
    Decode step that decodes the latents to images and postprocess the generated image.

      Components:
          vae (`AutoencoderKLQwenImage`) image_processor (`VaeImageProcessor`)

      Inputs:
          latents (`Tensor`):
              The denoised latents to decode, can be generated in the denoise step and unpacked in the after denoise
              step.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images. (tensor output of the vae decoder.)
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
    Decode step that decodes the latents to images and postprocess the generated image, optionally apply the mask
    overlay to the original image.

      Components:
          vae (`AutoencoderKLQwenImage`) image_mask_processor (`InpaintProcessor`)

      Inputs:
          latents (`Tensor`):
              The denoised latents to decode, can be generated in the denoise step and unpacked in the after denoise
              step.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          mask_overlay_kwargs (`dict`, *optional*):
              The kwargs for the postprocess step to apply the mask overlay. generated in
              InpaintProcessImagesInputStep.

      Outputs:
          images (`list`):
              Generated images. (tensor output of the vae decoder.)
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
            OutputParam.template("latents"),
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
    Auto Modular pipeline for edit (img2img) and edit inpaint tasks using QwenImage-Edit.
      - for edit (img2img) generation, you need to provide `image`
      - for edit inpainting, you need to provide `mask_image` and `image`, optionally you can provide
        `padding_mask_crop`


      Supported workflows:
        - `image_conditioned`: requires `prompt`, `image`
        - `image_conditioned_inpainting`: requires `prompt`, `mask_image`, `image`

      Components:
          image_resize_processor (`VaeImageProcessor`) text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor
          (`Qwen2VLProcessor`) guider (`ClassifierFreeGuidance`) image_mask_processor (`InpaintProcessor`) vae
          (`AutoencoderKLQwenImage`) image_processor (`VaeImageProcessor`) pachifier (`QwenImagePachifier`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) transformer (`QwenImageTransformer2DModel`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
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
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The processed mask image
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          mask_overlay_kwargs (`dict`, *optional*):
              The kwargs for the postprocess step to apply the mask overlay. generated in
              InpaintProcessImagesInputStep.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage-edit"
    block_classes = EDIT_AUTO_BLOCKS.values()
    block_names = EDIT_AUTO_BLOCKS.keys()
    _workflow_map = {
        "image_conditioned": {"prompt": True, "image": True},
        "image_conditioned_inpainting": {"prompt": True, "mask_image": True, "image": True},
    }

    @property
    def description(self):
        return (
            "Auto Modular pipeline for edit (img2img) and edit inpaint tasks using QwenImage-Edit.\n"
            "- for edit (img2img) generation, you need to provide `image`\n"
            "- for edit inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop`\n"
        )

    @property
    def outputs(self):
        return [OutputParam.template("images")]
