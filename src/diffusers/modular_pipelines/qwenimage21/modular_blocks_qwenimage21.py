# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    QwenImage21ImageConditionedRoPEInputsStep,
    QwenImage21PrepareLatentsStep,
    QwenImage21RoPEInputsStep,
    QwenImage21SetTimestepsStep,
)
from .decoders import QwenImage21DecodeStep, QwenImage21UnpackLatentsStep
from .denoise import QwenImage21DenoiseStep, QwenImage21ImageConditionedDenoiseStep
from .encoders import (
    QwenImage21ProcessImagesInputStep,
    QwenImage21ResizeStep,
    QwenImage21TextEncoderStep,
    QwenImage21VaeEncoderStep,
    QwenImage21VLTextEncoderStep,
)
from .inputs import QwenImage21AdditionalInputsStep, QwenImage21TextInputsStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. TEXT ENCODER
# ====================

QwenImage21VLEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImage21ResizeStep()),
        ("encode", QwenImage21VLTextEncoderStep()),
    ]
)


# auto_docstring
class QwenImage21VLEncoderStep(SequentialPipelineBlocks):
    """
    Vision-language encoder step that resizes the condition images and encodes them together with the prompt.

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) processor
          (`Qwen3VLProcessor`) guider (`ClassifierFreeGuidance`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.

      Outputs:
          resized_image (`list`):
              RGBA condition images resized to the `output_resolution` target area
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings
          negative_image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the negative prompt embeddings
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21VLEncoderBlocks.values()
    block_names = QwenImage21VLEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return (
            "Vision-language encoder step that resizes the condition images and encodes them together with the prompt."
        )


# auto_docstring
class QwenImage21AutoTextEncoderStep(AutoPipelineBlocks):
    """
    Text encoder step that encodes the prompt, together with the condition images when there are any.
      This is an auto pipeline block that works for text-to-image and image-conditioned generation.
       - `QwenImage21VLEncoderStep` is used when `image` is provided.
       - `QwenImage21TextEncoderStep` is used otherwise.

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) processor
          (`Qwen3VLProcessor`) guider (`ClassifierFreeGuidance`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.

      Outputs:
          resized_image (`list`):
              RGBA condition images resized to the `output_resolution` target area
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings
          negative_image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the negative prompt embeddings
    """

    model_name = "qwenimage21"
    block_classes = [QwenImage21VLEncoderStep, QwenImage21TextEncoderStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self) -> str:
        return (
            "Text encoder step that encodes the prompt, together with the condition images when there are any.\n"
            "This is an auto pipeline block that works for text-to-image and image-conditioned generation.\n"
            " - `QwenImage21VLEncoderStep` is used when `image` is provided.\n"
            " - `QwenImage21TextEncoderStep` is used otherwise."
        )


# ====================
# 2. VAE ENCODER
# ====================

QwenImage21VaeEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImage21ResizeStep()),
        ("preprocess", QwenImage21ProcessImagesInputStep()),
        ("encode", QwenImage21VaeEncoderStep()),
    ]
)


# auto_docstring
class QwenImage21VaeEncoderSequentialStep(SequentialPipelineBlocks):
    """
    VAE encoder step that resizes, preprocesses and encodes the condition images into latents.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage21`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          resized_image (`list`):
              RGBA condition images resized to the `output_resolution` target area
          processed_image (`list`):
              Normalized RGBA image tensors, one per condition image
          image_latents (`list`):
              Normalized latents of each condition image, each of shape (1, C, 1, H, W)
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21VaeEncoderBlocks.values()
    block_names = QwenImage21VaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "VAE encoder step that resizes, preprocesses and encodes the condition images into latents."


# auto_docstring
class QwenImage21AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the condition images into their latent representations.
      This is an auto pipeline block that works for image-conditioned generation.
       - `QwenImage21VaeEncoderSequentialStep` is used when `image` is provided.
       - If `image` is not provided, step will be skipped.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage21`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          resized_image (`list`):
              RGBA condition images resized to the `output_resolution` target area
          processed_image (`list`):
              Normalized RGBA image tensors, one per condition image
          image_latents (`list`):
              Normalized latents of each condition image, each of shape (1, C, 1, H, W)
    """

    model_name = "qwenimage21"
    block_classes = [QwenImage21VaeEncoderSequentialStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self) -> str:
        return (
            "VAE encoder step that encodes the condition images into their latent representations.\n"
            "This is an auto pipeline block that works for image-conditioned generation.\n"
            " - `QwenImage21VaeEncoderSequentialStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


# ====================
# 3. DENOISE
# ====================

QwenImage21CoreDenoiseBlocks = InsertableDict(
    [
        ("input", QwenImage21TextInputsStep()),
        ("prepare_latents", QwenImage21PrepareLatentsStep()),
        ("set_timesteps", QwenImage21SetTimestepsStep()),
        ("prepare_rope_inputs", QwenImage21RoPEInputsStep()),
        ("denoise", QwenImage21DenoiseStep()),
        ("unpack_latents", QwenImage21UnpackLatentsStep()),
    ]
)


# auto_docstring
class QwenImage21CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for text-to-image generation.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Default sampling grid of the checkpoint, used when `sigmas` is not passed.

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
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.
          negative_image_pad_mask (`Tensor`, *optional*):
              Bool mask marking the vision positions of the negative prompt embeddings. Can be generated from
              text_encoder step.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21CoreDenoiseBlocks.values()
    block_names = QwenImage21CoreDenoiseBlocks.keys()

    @property
    def description(self) -> str:
        return "Core denoise step that performs the denoising process for text-to-image generation."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


QwenImage21ImageConditionedCoreDenoiseBlocks = InsertableDict(
    [
        ("input", QwenImage21TextInputsStep()),
        ("additional_inputs", QwenImage21AdditionalInputsStep()),
        ("prepare_latents", QwenImage21PrepareLatentsStep()),
        ("set_timesteps", QwenImage21SetTimestepsStep()),
        ("prepare_rope_inputs", QwenImage21ImageConditionedRoPEInputsStep()),
        ("denoise", QwenImage21ImageConditionedDenoiseStep()),
        ("unpack_latents", QwenImage21UnpackLatentsStep()),
    ]
)


# auto_docstring
class QwenImage21ImageConditionedCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for image-conditioned generation.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Default sampling grid of the checkpoint, used when `sigmas` is not passed.

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
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.
          negative_image_pad_mask (`Tensor`, *optional*):
              Bool mask marking the vision positions of the negative prompt embeddings. Can be generated from
              text_encoder step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_latents (`list`):
              Normalized latents of each condition image. Can be generated from vae_encoder step.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21ImageConditionedCoreDenoiseBlocks.values()
    block_names = QwenImage21ImageConditionedCoreDenoiseBlocks.keys()

    @property
    def description(self) -> str:
        return "Core denoise step that performs the denoising process for image-conditioned generation."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# auto_docstring
class QwenImage21AutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Auto core denoise step that performs the denoising process.
      This is an auto pipeline block that works for text-to-image and image-conditioned generation.
       - `QwenImage21ImageConditionedCoreDenoiseStep` is used when `image_latents` is provided.
       - `QwenImage21CoreDenoiseStep` is used otherwise.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Default sampling grid of the checkpoint, used when `sigmas` is not passed.

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
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.
          negative_image_pad_mask (`Tensor`, *optional*):
              Bool mask marking the vision positions of the negative prompt embeddings. Can be generated from
              text_encoder step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          image_latents (`list`, *optional*):
              Normalized latents of each condition image. Can be generated from vae_encoder step.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21"
    block_classes = [QwenImage21ImageConditionedCoreDenoiseStep, QwenImage21CoreDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self) -> str:
        return (
            "Auto core denoise step that performs the denoising process.\n"
            "This is an auto pipeline block that works for text-to-image and image-conditioned generation.\n"
            " - `QwenImage21ImageConditionedCoreDenoiseStep` is used when `image_latents` is provided.\n"
            " - `QwenImage21CoreDenoiseStep` is used otherwise."
        )


# ====================
# 4. AUTO BLOCKS
# ====================

AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImage21AutoTextEncoderStep()),
        ("vae_encoder", QwenImage21AutoVaeEncoderStep()),
        ("denoise", QwenImage21AutoCoreDenoiseStep()),
        ("decode", QwenImage21DecodeStep()),
    ]
)


# auto_docstring
class QwenImage21AutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image and image-conditioned generation using Qwen-Image 2.1.
      - for text-to-image generation, all you need to provide is `prompt`
      - for image-conditioned generation, you need to provide `prompt` and `image` (one image or a list)

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image_conditioned`: requires `prompt`, `image`

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) processor
          (`Qwen3VLProcessor`) guider (`ClassifierFreeGuidance`) vae (`AutoencoderKLQwenImage21`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) transformer (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Default sampling grid of the checkpoint, used when `sigmas` is not passed.

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
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
          image_latents (`list`, *optional*):
              Normalized latents of each condition image. Can be generated from vae_encoder step.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache the text and condition-image keys and values after the first step. Valid because `causal_condition`
              modulates those tokens from `t = 0`, making their activations step-independent. Toggling it does not
              reproduce the same image bit-for-bit in reduced precision.
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

    model_name = "qwenimage21"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
    }

    @property
    def description(self) -> str:
        return (
            "Auto Modular pipeline for text-to-image and image-conditioned generation using Qwen-Image 2.1.\n"
            "- for text-to-image generation, all you need to provide is `prompt`\n"
            "- for image-conditioned generation, you need to provide `prompt` and `image` (one image or a list)"
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("images"),
        ]
