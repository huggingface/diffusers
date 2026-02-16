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
    QwenImageLayeredPrepareLatentsStep,
    QwenImageLayeredRoPEInputsStep,
    QwenImageLayeredSetTimestepsStep,
)
from .decoders import (
    QwenImageLayeredAfterDenoiseStep,
    QwenImageLayeredDecoderStep,
)
from .denoise import (
    QwenImageLayeredDenoiseStep,
)
from .encoders import (
    QwenImageEditProcessImagesInputStep,
    QwenImageLayeredGetImagePromptStep,
    QwenImageLayeredPermuteLatentsStep,
    QwenImageLayeredResizeStep,
    QwenImageTextEncoderStep,
    QwenImageVaeEncoderStep,
)
from .inputs import (
    QwenImageLayeredAdditionalInputsStep,
    QwenImageTextInputsStep,
)


logger = logging.get_logger(__name__)


# ====================
# 1. TEXT ENCODER
# ====================


# auto_docstring
class QwenImageLayeredTextEncoderStep(SequentialPipelineBlocks):
    """
    QwenImage-Layered Text encoder step that encode the text prompt, will generate a prompt based on image if not
    provided.

      Components:
          image_resize_processor (`VaeImageProcessor`) text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor
          (`Qwen2VLProcessor`) tokenizer (`Qwen2Tokenizer`): The tokenizer to use guider (`ClassifierFreeGuidance`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          resolution (`int`, *optional*, defaults to 640):
              The target area to resize the image to, can be 1024 or 640
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          use_en_prompt (`bool`, *optional*, defaults to False):
              Whether to use English prompt template
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.

      Outputs:
          resized_image (`list`):
              The resized images
          prompt (`str`):
              The prompt or prompts to guide image generation. If not provided, updated using image caption
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "qwenimage-layered"
    block_classes = [
        QwenImageLayeredResizeStep(),
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
# auto_docstring
class QwenImageLayeredVaeEncoderStep(SequentialPipelineBlocks):
    """
    Vae encoder step that encode the image inputs into their latent representations.

      Components:
          image_resize_processor (`VaeImageProcessor`) image_processor (`VaeImageProcessor`) vae
          (`AutoencoderKLQwenImage`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          resolution (`int`, *optional*, defaults to 640):
              The target area to resize the image to, can be 1024 or 640
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

    model_name = "qwenimage-layered"
    block_classes = [
        QwenImageLayeredResizeStep(),
        QwenImageEditProcessImagesInputStep(),
        QwenImageVaeEncoderStep(),
        QwenImageLayeredPermuteLatentsStep(),
    ]
    block_names = ["resize", "preprocess", "encode", "permute"]

    @property
    def description(self) -> str:
        return "Vae encoder step that encode the image inputs into their latent representations."


# ====================
# 3. DENOISE (input -> prepare_latents -> set_timesteps -> prepare_rope_inputs -> denoise -> after_denoise)
# ====================


# assemble input steps
# auto_docstring
class QwenImageLayeredInputStep(SequentialPipelineBlocks):
    """
    Input step that prepares the inputs for the layered denoising step. It:
       - make sure the text embeddings have consistent batch size as well as the additional inputs.
       - update height/width based `image_latents`, patchify `image_latents`.

      Components:
          pachifier (`QwenImageLayeredPachifier`)

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
              image latents used to guide the image generation. Can be generated from vae_encoder step. (patchified
              with layered pachifier and batch-expanded)
    """

    model_name = "qwenimage-layered"
    block_classes = [
        QwenImageTextInputsStep(),
        QwenImageLayeredAdditionalInputsStep(),
    ]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return (
            "Input step that prepares the inputs for the layered denoising step. It:\n"
            " - make sure the text embeddings have consistent batch size as well as the additional inputs.\n"
            " - update height/width based `image_latents`, patchify `image_latents`."
        )


# Qwen Image Layered (image2image) core denoise step
# auto_docstring
class QwenImageLayeredCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for QwenImage-Layered img2img task.

      Components:
          pachifier (`QwenImageLayeredPachifier`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
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
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          layers (`int`, *optional*, defaults to 4):
              Number of layers to extract from the image
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

    model_name = "qwenimage-layered"
    block_classes = [
        QwenImageLayeredInputStep(),
        QwenImageLayeredPrepareLatentsStep(),
        QwenImageLayeredSetTimestepsStep(),
        QwenImageLayeredRoPEInputsStep(),
        QwenImageLayeredDenoiseStep(),
        QwenImageLayeredAfterDenoiseStep(),
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
        return "Core denoising workflow for QwenImage-Layered img2img task."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# ====================
# 4. AUTO BLOCKS & PRESETS
# ====================

LAYERED_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageLayeredTextEncoderStep()),
        ("vae_encoder", QwenImageLayeredVaeEncoderStep()),
        ("denoise", QwenImageLayeredCoreDenoiseStep()),
        ("decode", QwenImageLayeredDecoderStep()),
    ]
)


# auto_docstring
class QwenImageLayeredAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for layered denoising tasks using QwenImage-Layered.

      Components:
          image_resize_processor (`VaeImageProcessor`) text_encoder (`Qwen2_5_VLForConditionalGeneration`) processor
          (`Qwen2VLProcessor`) tokenizer (`Qwen2Tokenizer`): The tokenizer to use guider (`ClassifierFreeGuidance`)
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage`) pachifier (`QwenImageLayeredPachifier`)
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`QwenImageTransformer2DModel`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          resolution (`int`, *optional*, defaults to 640):
              The target area to resize the image to, can be 1024 or 640
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          use_en_prompt (`bool`, *optional*, defaults to False):
              Whether to use English prompt template
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          layers (`int`, *optional*, defaults to 4):
              Number of layers to extract from the image
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

    model_name = "qwenimage-layered"
    block_classes = LAYERED_AUTO_BLOCKS.values()
    block_names = LAYERED_AUTO_BLOCKS.keys()

    @property
    def description(self):
        return "Auto Modular pipeline for layered denoising tasks using QwenImage-Layered."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
