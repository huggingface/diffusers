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
    QwenImage21TurboSetTimestepsStep,
)
from .decoders import QwenImage21DecodeStep, QwenImage21UnpackLatentsStep
from .denoise import QwenImage21TurboDenoiseStep, QwenImage21TurboImageConditionedDenoiseStep
from .encoders import (
    QwenImage21ProcessImagesInputStep,
    QwenImage21ResizeStep,
    QwenImage21TurboTextEncoderStep,
    QwenImage21TurboVLTextEncoderStep,
    QwenImage21VaeEncoderStep,
)
from .inputs import QwenImage21AdditionalInputsStep, QwenImage21TurboTextInputsStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. TEXT ENCODER
# ====================

QwenImage21TurboVLEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImage21ResizeStep()),
        ("encode", QwenImage21TurboVLTextEncoderStep()),
    ]
)


# auto_docstring
class QwenImage21TurboVLEncoderStep(SequentialPipelineBlocks):
    """
    Vision-language encoder step that resizes the condition images and encodes them together with the prompt.

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) processor
          (`Qwen3VLProcessor`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          prompt (`str`):
              The prompt or prompts to guide image generation.

      Outputs:
          resized_image (`list`):
              RGBA condition images resized to the `output_resolution` target area
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings
    """

    model_name = "qwenimage21-turbo"
    block_classes = QwenImage21TurboVLEncoderBlocks.values()
    block_names = QwenImage21TurboVLEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return (
            "Vision-language encoder step that resizes the condition images and encodes them together with the prompt."
        )


# auto_docstring
class QwenImage21TurboAutoTextEncoderStep(AutoPipelineBlocks):
    """
    Text encoder step that encodes the prompt, together with the condition images when there are any.
      This is an auto pipeline block that works for text-to-image and image-conditioned generation.
       - `QwenImage21TurboVLEncoderStep` is used when `image` is provided.
       - `QwenImage21TurboTextEncoderStep` is used otherwise.

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) processor
          (`Qwen3VLProcessor`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          prompt (`str`):
              The prompt or prompts to guide image generation.

      Outputs:
          resized_image (`list`):
              RGBA condition images resized to the `output_resolution` target area
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings
    """

    model_name = "qwenimage21-turbo"
    block_classes = [QwenImage21TurboVLEncoderStep, QwenImage21TurboTextEncoderStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self) -> str:
        return (
            "Text encoder step that encodes the prompt, together with the condition images when there are any.\n"
            "This is an auto pipeline block that works for text-to-image and image-conditioned generation.\n"
            " - `QwenImage21TurboVLEncoderStep` is used when `image` is provided.\n"
            " - `QwenImage21TurboTextEncoderStep` is used otherwise."
        )


# ====================
# 2. VAE ENCODER
# ====================

QwenImage21TurboVaeEncoderBlocks = InsertableDict(
    [
        ("resize", QwenImage21ResizeStep()),
        ("preprocess", QwenImage21ProcessImagesInputStep()),
        ("encode", QwenImage21VaeEncoderStep()),
    ]
)


# auto_docstring
class QwenImage21TurboVaeEncoderSequentialStep(SequentialPipelineBlocks):
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

    model_name = "qwenimage21-turbo"
    block_classes = QwenImage21TurboVaeEncoderBlocks.values()
    block_names = QwenImage21TurboVaeEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "VAE encoder step that resizes, preprocesses and encodes the condition images into latents."


# auto_docstring
class QwenImage21TurboAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the condition images into their latent representations.
      This is an auto pipeline block that works for image-conditioned generation.
       - `QwenImage21TurboVaeEncoderSequentialStep` is used when `image` is provided.
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

    model_name = "qwenimage21-turbo"
    block_classes = [QwenImage21TurboVaeEncoderSequentialStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self) -> str:
        return (
            "VAE encoder step that encodes the condition images into their latent representations.\n"
            "This is an auto pipeline block that works for image-conditioned generation.\n"
            " - `QwenImage21TurboVaeEncoderSequentialStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


# ====================
# 3. DENOISE
# ====================

QwenImage21TurboCoreDenoiseBlocks = InsertableDict(
    [
        ("input", QwenImage21TurboTextInputsStep()),
        ("prepare_latents", QwenImage21PrepareLatentsStep()),
        ("set_timesteps", QwenImage21TurboSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImage21RoPEInputsStep()),
        ("denoise", QwenImage21TurboDenoiseStep()),
        ("unpack_latents", QwenImage21UnpackLatentsStep()),
    ]
)


# auto_docstring
class QwenImage21TurboCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for text-to-image generation with the Turbo checkpoint.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Sampling grid the checkpoint ships with, used when `sigmas` is not passed.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.
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
          num_inference_steps (`int`, *optional*, defaults to 8):
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

    model_name = "qwenimage21-turbo"
    block_classes = QwenImage21TurboCoreDenoiseBlocks.values()
    block_names = QwenImage21TurboCoreDenoiseBlocks.keys()

    @property
    def description(self) -> str:
        return "Core denoise step that performs the denoising process for text-to-image generation with the Turbo checkpoint."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


QwenImage21TurboImageConditionedCoreDenoiseBlocks = InsertableDict(
    [
        ("input", QwenImage21TurboTextInputsStep()),
        ("additional_inputs", QwenImage21AdditionalInputsStep()),
        ("prepare_latents", QwenImage21PrepareLatentsStep()),
        ("set_timesteps", QwenImage21TurboSetTimestepsStep()),
        ("prepare_rope_inputs", QwenImage21ImageConditionedRoPEInputsStep()),
        ("denoise", QwenImage21TurboImageConditionedDenoiseStep()),
        ("unpack_latents", QwenImage21UnpackLatentsStep()),
    ]
)


# auto_docstring
class QwenImage21TurboImageConditionedCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for image-conditioned generation with the Turbo checkpoint.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Sampling grid the checkpoint ships with, used when `sigmas` is not passed.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.
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
          num_inference_steps (`int`, *optional*, defaults to 8):
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

    model_name = "qwenimage21-turbo"
    block_classes = QwenImage21TurboImageConditionedCoreDenoiseBlocks.values()
    block_names = QwenImage21TurboImageConditionedCoreDenoiseBlocks.keys()

    @property
    def description(self) -> str:
        return "Core denoise step that performs the denoising process for image-conditioned generation with the Turbo checkpoint."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# auto_docstring
class QwenImage21TurboAutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Auto core denoise step that performs the denoising process with the Turbo checkpoint.
      This is an auto pipeline block that works for text-to-image and image-conditioned generation.
       - `QwenImage21TurboImageConditionedCoreDenoiseStep` is used when `image_latents` is provided.
       - `QwenImage21TurboCoreDenoiseStep` is used otherwise.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Sampling grid the checkpoint ships with, used when `sigmas` is not passed.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          image_pad_mask (`Tensor`):
              Bool mask marking the vision positions of the prompt embeddings. Can be generated from text_encoder step.
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

    model_name = "qwenimage21-turbo"
    block_classes = [QwenImage21TurboImageConditionedCoreDenoiseStep, QwenImage21TurboCoreDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self) -> str:
        return (
            "Auto core denoise step that performs the denoising process with the Turbo checkpoint.\n"
            "This is an auto pipeline block that works for text-to-image and image-conditioned generation.\n"
            " - `QwenImage21TurboImageConditionedCoreDenoiseStep` is used when `image_latents` is provided.\n"
            " - `QwenImage21TurboCoreDenoiseStep` is used otherwise."
        )


# ====================
# 4. AUTO BLOCKS
# ====================

TURBO_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImage21TurboAutoTextEncoderStep()),
        ("vae_encoder", QwenImage21TurboAutoVaeEncoderStep()),
        ("denoise", QwenImage21TurboAutoCoreDenoiseStep()),
        ("decode", QwenImage21DecodeStep()),
    ]
)


# auto_docstring
class QwenImage21TurboAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image and image-conditioned generation using Qwen-Image 2.1 Turbo, the
    accelerated checkpoint that samples its shipped sigma grid in a few steps without guidance.
      - for text-to-image generation, all you need to provide is `prompt`
      - for image-conditioned generation, you need to provide `prompt` and `image` (one image or a list)

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image_conditioned`: requires `prompt`, `image`

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) processor
          (`Qwen3VLProcessor`) vae (`AutoencoderKLQwenImage21`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          transformer (`QwenImage21Transformer2DModel`)

      Configs:
          sample_sigmas (default: None): Sampling grid the checkpoint ships with, used when `sigmas` is not passed.

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to derive the output size and to resize condition images.
          prompt (`str`):
              The prompt or prompts to guide image generation.
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

    model_name = "qwenimage21-turbo"
    block_classes = TURBO_AUTO_BLOCKS.values()
    block_names = TURBO_AUTO_BLOCKS.keys()

    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
    }

    @property
    def description(self) -> str:
        return (
            "Auto Modular pipeline for text-to-image and image-conditioned generation using Qwen-Image 2.1 Turbo, "
            "the accelerated checkpoint that samples its shipped sigma grid in a few steps without guidance.\n"
            "- for text-to-image generation, all you need to provide is `prompt`\n"
            "- for image-conditioned generation, you need to provide `prompt` and `image` (one image or a list)"
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("images"),
        ]
