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
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    QwenImageEditPlusRoPEInputsStep,
    QwenImagePrepareLatentsStep,
    QwenImageSetTimestepsStep,
)
from .decoders import (
    QwenImageAfterDenoiseStep,
    QwenImageDecoderStep,
    QwenImageProcessImagesOutputStep,
)
from .denoise import (
    QwenImageEditDenoiseStep,
)
from .encoders import (
    QwenImageEditPlusProcessImagesInputStep,
    QwenImageEditPlusResizeStep,
    QwenImageEditPlusTextEncoderStep,
    QwenImageVaeEncoderStep,
)
from .inputs import (
    QwenImageEditPlusAdditionalInputsStep,
    QwenImageTextInputsStep,
)


logger = logging.get_logger(__name__)


# ====================
# 1. TEXT ENCODER
# ====================


# auto_docstring
class QwenImageEditPlusVLEncoderStep(SequentialPipelineBlocks):
    """
    QwenImage-Edit Plus VL encoder step that encodes the image and text prompts together.

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
              Images resized to 1024x1024 target area for VAE encoding
          resized_cond_image (`list`):
              Images resized to 384x384 target area for VL text encoding
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "qwenimage-edit-plus"
    block_classes = [
        QwenImageEditPlusResizeStep(),
        QwenImageEditPlusTextEncoderStep(),
    ]
    block_names = ["resize", "encode"]

    @property
    def description(self) -> str:
        return "QwenImage-Edit Plus VL encoder step that encodes the image and text prompts together."


# ====================
# 2. VAE ENCODER
# ====================


# auto_docstring
class QwenImageEditPlusVaeEncoderStep(SequentialPipelineBlocks):
    """
    VAE encoder step that encodes image inputs into latent representations.
      Each image is resized independently based on its own aspect ratio to 1024x1024 target area.

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
              Images resized to 1024x1024 target area for VAE encoding
          resized_cond_image (`list`):
              Images resized to 384x384 target area for VL text encoding
          processed_image (`Tensor`):
              The processed image
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "qwenimage-edit-plus"
    block_classes = [
        QwenImageEditPlusResizeStep(),
        QwenImageEditPlusProcessImagesInputStep(),
        QwenImageVaeEncoderStep(),
    ]
    block_names = ["resize", "preprocess", "encode"]

    @property
    def description(self) -> str:
        return (
            "VAE encoder step that encodes image inputs into latent representations.\n"
            "Each image is resized independently based on its own aspect ratio to 1024x1024 target area."
        )


# ====================
# 3. DENOISE (input -> prepare_latents -> set_timesteps -> prepare_rope_inputs -> denoise -> after_denoise)
# ====================


# assemble input steps
# auto_docstring
class QwenImageEditPlusInputStep(SequentialPipelineBlocks):
    """
    Input step that prepares the inputs for the Edit Plus denoising step. It:
       - Standardizes text embeddings batch size.
       - Processes list of image latents: patchifies, concatenates along dim=1, expands batch.
       - Outputs lists of image_height/image_width for RoPE calculation.
       - Defaults height/width from last image in the list.

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
          image_height (`list`):
              The image heights calculated from the image latents dimension
          image_width (`list`):
              The image widths calculated from the image latents dimension
          height (`int`):
              if not provided, updated to image height
          width (`int`):
              if not provided, updated to image width
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step. (patchified,
              concatenated, and batch-expanded)
    """

    model_name = "qwenimage-edit-plus"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageEditPlusAdditionalInputsStep(),
    ]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the Edit Plus denoising step. It:\n"
            " - Standardizes text embeddings batch size.\n"
            " - Processes list of image latents: patchifies, concatenates along dim=1, expands batch.\n"
            " - Outputs lists of image_height/image_width for RoPE calculation.\n"
            " - Defaults height/width from last image in the list."
        )


# Qwen Image Edit Plus (image2image) core denoise step
# auto_docstring
class QwenImageEditPlusCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for QwenImage-Edit Plus edit (img2img) task.

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

    model_name = "qwenimage-edit-plus"
    block_classes = [
        QwenImageEditPlusInputStep(),
        QwenImagePrepareLatentsStep(),
        QwenImageSetTimestepsStep(),
        QwenImageEditPlusRoPEInputsStep(),
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
        return "Core denoising workflow for QwenImage-Edit Plus edit (img2img) task."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# ====================
# 4. DECODE
# ====================


# auto_docstring
class QwenImageEditPlusDecodeStep(SequentialPipelineBlocks):
    """
    Decode step that decodes the latents to images and postprocesses the generated image.

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

    model_name = "qwenimage-edit-plus"
    block_classes = [QwenImageDecoderStep(), QwenImageProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocesses the generated image."


# ====================
# 5. AUTO BLOCKS & PRESETS
# ====================

EDIT_PLUS_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageEditPlusVLEncoderStep()),
        ("vae_encoder", QwenImageEditPlusVaeEncoderStep()),
        ("denoise", QwenImageEditPlusCoreDenoiseStep()),
        ("decode", QwenImageEditPlusDecodeStep()),
    ]
)


# auto_docstring
class QwenImageEditPlusAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for edit (img2img) tasks using QwenImage-Edit Plus.
      - `image` is required input (can be single image or list of images).
      - Each image is resized independently based on its own aspect ratio.
      - VL encoder uses 384x384 target area, VAE encoder uses 1024x1024 target area.

      Components:
          image_resize_processor (`VaeImageProcessor`) text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor
          (`Qwen2VLProcessor`) guider (`ClassifierFreeGuidance`) image_processor (`VaeImageProcessor`) vae
          (`AutoencoderKLQwenImage`) pachifier (`QwenImagePachifier`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          transformer (`QwenImageTransformer2DModel`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage-edit-plus"
    block_classes = EDIT_PLUS_AUTO_BLOCKS.values()
    block_names = EDIT_PLUS_AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Auto Modular pipeline for edit (img2img) tasks using QwenImage-Edit Plus.\n"
            "- `image` is required input (can be single image or list of images).\n"
            "- Each image is resized independently based on its own aspect ratio.\n"
            "- VL encoder uses 384x384 target area, VAE encoder uses 1024x1024 target area."
        )

    @property
    def outputs(self):
        return [OutputParam.template("images")]
