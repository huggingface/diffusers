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
# 1. TEXT ENCODER
# ====================


# auto_docstring
class QwenImageAutoTextEncoderStep(AutoPipelineBlocks):
    """
    Text encoder step that encodes the text prompt into a text embedding. This is an auto pipeline block.

      Components:
          text_encoder (`Qwen2_5_VLForConditionalGeneration`): The text encoder to use tokenizer (`Qwen2Tokenizer`):
          The tokenizer to use guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "qwenimage"
    block_classes = [QwenImageTextEncoderStep()]
    block_names = ["text_encoder"]
    block_trigger_inputs = ["prompt"]

    @property
    def description(self) -> str:
        return "Text encoder step that encodes the text prompt into a text embedding. This is an auto pipeline block."
        " - `QwenImageTextEncoderStep` (text_encoder) is used when `prompt` is provided."
        " - if `prompt` is not provided, step will be skipped."


# ====================
# 2. VAE ENCODER
# ====================


# auto_docstring
class QwenImageInpaintVaeEncoderStep(SequentialPipelineBlocks):
    """
    This step is used for processing image and mask inputs for inpainting tasks. It:
       - Resizes the image to the target size, based on `height` and `width`.
       - Processes and updates `image` and `mask_image`.
       - Creates `image_latents`.

      Components:
          image_mask_processor (`InpaintProcessor`) vae (`AutoencoderKLQwenImage`)

      Inputs:
          mask_image (`Image`):
              Mask image for inpainting.
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          processed_image (`Tensor`):
              The processed image
          processed_mask_image (`Tensor`):
              The processed mask image
          mask_overlay_kwargs (`dict`):
              The kwargs for the postprocess step to apply the mask overlay
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

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


# auto_docstring
class QwenImageImg2ImgVaeEncoderStep(SequentialPipelineBlocks):
    """
    Vae encoder step that preprocess andencode the image inputs into their latent representations.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          processed_image (`Tensor`):
              The processed image
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "qwenimage"

    block_classes = [QwenImageProcessImagesInputStep(), QwenImageVaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


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
# auto_docstring
class QwenImageOptionalControlNetVaeEncoderStep(AutoPipelineBlocks):
    """
    Vae encoder step that encode the image inputs into their latent representations.
      This is an auto pipeline block.
       - `QwenImageControlNetVaeEncoderStep` (controlnet) is used when `control_image` is provided.
       - if `control_image` is not provided, step will be skipped.

      Components:
          vae (`AutoencoderKLQwenImage`) controlnet (`QwenImageControlNetModel`) control_image_processor
          (`VaeImageProcessor`)

      Inputs:
          control_image (`Image`, *optional*):
              Control image for ControlNet conditioning.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          control_image_latents (`Tensor`):
              The latents representing the control image
    """

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
# 3. DENOISE (input -> prepare_latents -> set_timesteps -> prepare_rope_inputs -> denoise -> after_denoise)
# ====================


# assemble input steps
# auto_docstring
class QwenImageImg2ImgInputStep(SequentialPipelineBlocks):
    """
    Input step that prepares the inputs for the img2img denoising step. It:

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

    model_name = "qwenimage"
    block_classes = [QwenImageTextInputsStep(), QwenImageAdditionalInputsStep()]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return "Input step that prepares the inputs for the img2img denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


# auto_docstring
class QwenImageInpaintInputStep(SequentialPipelineBlocks):
    """
    Input step that prepares the inputs for the inpainting denoising step. It:

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
          image_latents (`Tensor`, *optional*):
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

    model_name = "qwenimage"
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
        return "Input step that prepares the inputs for the inpainting denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents` and `processed_mask_image`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


# assemble prepare latents steps
# auto_docstring
class QwenImageInpaintPrepareLatentsStep(SequentialPipelineBlocks):
    """
    This step prepares the latents/image_latents and mask inputs for the inpainting denoising step. It:
       - Add noise to the image latents to create the latents input for the denoiser.
       - Create the pachified latents `mask` based on the processedmask image.

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
# auto_docstring
class QwenImageCoreDenoiseStep(SequentialPipelineBlocks):
    """
    step that denoise noise into image for text2image task. It includes the denoise loop, as well as prepare the inputs
    (timesteps, latents, rope inputs etc.).

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
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
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

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# Qwen Image (inpainting)
# auto_docstring
class QwenImageInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for inpaint
    task.

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
          image_latents (`Tensor`, *optional*):
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

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# Qwen Image (image2image)
# auto_docstring
class QwenImageImg2ImgCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for img2img
    task.

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

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# Qwen Image (text2image) with controlnet
# auto_docstring
class QwenImageControlNetCoreDenoiseStep(SequentialPipelineBlocks):
    """
    step that denoise noise into image for text2image task. It includes the denoise loop, as well as prepare the inputs
    (timesteps, latents, rope inputs etc.).

      Components:
          pachifier (`QwenImagePachifier`) scheduler (`FlowMatchEulerDiscreteScheduler`) controlnet
          (`QwenImageControlNetModel`) guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`)

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
          control_image_latents (`Tensor`):
              The control image latents to use for the denoising process. Can be generated in controlnet vae encoder
              step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          control_guidance_start (`float`, *optional*, defaults to 0.0):
              When to start applying ControlNet.
          control_guidance_end (`float`, *optional*, defaults to 1.0):
              When to stop applying ControlNet.
          controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
              Scale for ControlNet conditioning.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

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

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# Qwen Image (inpainting) with controlnet
# auto_docstring
class QwenImageControlNetInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for inpaint
    task.

      Components:
          pachifier (`QwenImagePachifier`) scheduler (`FlowMatchEulerDiscreteScheduler`) controlnet
          (`QwenImageControlNetModel`) guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`)

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
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The processed mask image
          control_image_latents (`Tensor`):
              The control image latents to use for the denoising process. Can be generated in controlnet vae encoder
              step.
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
          control_guidance_start (`float`, *optional*, defaults to 0.0):
              When to start applying ControlNet.
          control_guidance_end (`float`, *optional*, defaults to 1.0):
              When to stop applying ControlNet.
          controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
              Scale for ControlNet conditioning.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

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

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# Qwen Image (image2image) with controlnet
# auto_docstring
class QwenImageControlNetImg2ImgCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepare the inputs (timesteps, latents, rope inputs etc.) for the denoise step for img2img
    task.

      Components:
          pachifier (`QwenImagePachifier`) scheduler (`FlowMatchEulerDiscreteScheduler`) controlnet
          (`QwenImageControlNetModel`) guider (`ClassifierFreeGuidance`) transformer (`QwenImageTransformer2DModel`)

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
          control_image_latents (`Tensor`):
              The control image latents to use for the denoising process. Can be generated in controlnet vae encoder
              step.
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
          control_guidance_start (`float`, *optional*, defaults to 0.0):
              When to start applying ControlNet.
          control_guidance_end (`float`, *optional*, defaults to 1.0):
              When to stop applying ControlNet.
          controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
              Scale for ControlNet conditioning.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

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

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


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
            OutputParam.template("latents"),
        ]


# ====================
# 4. DECODE
# ====================


# standard decode step works for most tasks except for inpaint
# auto_docstring
class QwenImageDecodeStep(SequentialPipelineBlocks):
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

    model_name = "qwenimage"
    block_classes = [QwenImageDecoderStep(), QwenImageProcessImagesOutputStep()]
    block_names = ["decode", "postprocess"]

    @property
    def description(self):
        return "Decode step that decodes the latents to images and postprocess the generated image."


# Inpaint decode step
# auto_docstring
class QwenImageInpaintDecodeStep(SequentialPipelineBlocks):
    """
    Decode step that decodes the latents to images and postprocess the generated image, optional apply the mask
    overally to the original image.

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
# 5. AUTO BLOCKS & PRESETS
# ====================
AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageAutoTextEncoderStep()),
        ("vae_encoder", QwenImageAutoVaeEncoderStep()),
        ("controlnet_vae_encoder", QwenImageOptionalControlNetVaeEncoderStep()),
        ("denoise", QwenImageAutoCoreDenoiseStep()),
        ("decode", QwenImageAutoDecodeStep()),
    ]
)


# auto_docstring
class QwenImageAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using QwenImage.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `prompt`, `image`
        - `inpainting`: requires `prompt`, `mask_image`, `image`
        - `controlnet_text2image`: requires `prompt`, `control_image`
        - `controlnet_image2image`: requires `prompt`, `image`, `control_image`
        - `controlnet_inpainting`: requires `prompt`, `mask_image`, `image`, `control_image`

      Components:
          text_encoder (`Qwen2_5_VLForConditionalGeneration`): The text encoder to use tokenizer (`Qwen2Tokenizer`):
          The tokenizer to use guider (`ClassifierFreeGuidance`) image_mask_processor (`InpaintProcessor`) vae
          (`AutoencoderKLQwenImage`) image_processor (`VaeImageProcessor`) controlnet (`QwenImageControlNetModel`)
          control_image_processor (`VaeImageProcessor`) pachifier (`QwenImagePachifier`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) transformer (`QwenImageTransformer2DModel`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          mask_image (`Image`, *optional*):
              Mask image for inpainting.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          control_image (`Image`, *optional*):
              Control image for ControlNet conditioning.
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
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The processed mask image
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          control_image_latents (`Tensor`, *optional*):
              The control image latents to use for the denoising process. Can be generated in controlnet vae encoder
              step.
          control_guidance_start (`float`, *optional*, defaults to 0.0):
              When to start applying ControlNet.
          control_guidance_end (`float`, *optional*, defaults to 1.0):
              When to stop applying ControlNet.
          controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
              Scale for ControlNet conditioning.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          mask_overlay_kwargs (`dict`, *optional*):
              The kwargs for the postprocess step to apply the mask overlay. generated in
              InpaintProcessImagesInputStep.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage"

    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    # Workflow map defines the trigger conditions for each workflow.
    # How to define:
    #   - Only include required inputs and trigger inputs (inputs that determine which blocks run)
    #   - currently, only supports `True` means the workflow triggers when the input is not None

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"prompt": True, "image": True},
        "inpainting": {"prompt": True, "mask_image": True, "image": True},
        "controlnet_text2image": {"prompt": True, "control_image": True},
        "controlnet_image2image": {"prompt": True, "image": True, "control_image": True},
        "controlnet_inpainting": {"prompt": True, "mask_image": True, "image": True, "control_image": True},
    }

    @property
    def description(self):
        return "Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using QwenImage."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
