# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    ZImageAdditionalInputsStep,
    ZImagePrepareLatentsStep,
    ZImagePrepareLatentswithImageStep,
    ZImageSetTimestepsStep,
    ZImageSetTimestepsWithStrengthStep,
    ZImageTextInputStep,
)
from .decoders import ZImageVaeDecoderStep
from .denoise import (
    ZImageDenoiseStep,
)
from .encoders import (
    ZImageTextEncoderStep,
    ZImageVaeImageEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. DENOISE
# ====================


# text2image: inputs(text) -> set_timesteps -> prepare_latents -> denoise
# auto_docstring
class ZImageCoreDenoiseStep(SequentialPipelineBlocks):
    """
    denoise block that takes encoded conditions and runs the denoising process.

      Components:
          transformer (`ZImageTransformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`list`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`list`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 9):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          **denoiser_input_fields (`None`, *optional*):
              The conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    block_classes = [
        ZImageTextInputStep,
        ZImagePrepareLatentsStep,
        ZImageSetTimestepsStep,
        ZImageDenoiseStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps", "denoise"]

    @property
    def description(self):
        return "denoise block that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# image2image: inputs(text + image_latents) -> prepare_latents -> set_timesteps -> set_timesteps_with_strength -> prepare_latents_with_image -> denoise
# auto_docstring
class ZImageImage2ImageCoreDenoiseStep(SequentialPipelineBlocks):
    """
    denoise block that takes encoded text and image latent conditions and runs the denoising process.

      Components:
          transformer (`ZImageTransformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`list`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`list`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          image_latents (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 9):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          strength (`None`, *optional*, defaults to 0.6):
              TODO: Add description.
          **denoiser_input_fields (`None`, *optional*):
              The conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    block_classes = [
        ZImageTextInputStep,
        ZImageAdditionalInputsStep(image_latent_inputs=["image_latents"]),
        ZImagePrepareLatentsStep,
        ZImageSetTimestepsStep,
        ZImageSetTimestepsWithStrengthStep,
        ZImagePrepareLatentswithImageStep,
        ZImageDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "prepare_latents",
        "set_timesteps",
        "set_timesteps_with_strength",
        "prepare_latents_with_image",
        "denoise",
    ]

    @property
    def description(self):
        return "denoise block that takes encoded text and image latent conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class ZImageAutoDenoiseStep(AutoPipelineBlocks):
    """
    Denoise step that iteratively denoise the latents. This is a auto pipeline block that works for text2image and
    image2image tasks. - `ZImageCoreDenoiseStep` (text2image) for text2image tasks. -
    `ZImageImage2ImageCoreDenoiseStep` (image2image) for image2image tasks. - if `image_latents` is provided,
    `ZImageImage2ImageCoreDenoiseStep` will be used.
       - if `image_latents` is not provided, `ZImageCoreDenoiseStep` will be used.

      Components:
          transformer (`ZImageTransformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`list`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`list`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          image_latents (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          strength (`None`, *optional*, defaults to 0.6):
              TODO: Add description.
          **denoiser_input_fields (`None`, *optional*):
              The conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    block_classes = [
        ZImageImage2ImageCoreDenoiseStep,
        ZImageCoreDenoiseStep,
    ]
    block_names = ["image2image", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2image and image2image tasks."
            " - `ZImageCoreDenoiseStep` (text2image) for text2image tasks."
            " - `ZImageImage2ImageCoreDenoiseStep` (image2image) for image2image tasks."
            + " - if `image_latents` is provided, `ZImageImage2ImageCoreDenoiseStep` will be used.\n"
            + " - if `image_latents` is not provided, `ZImageCoreDenoiseStep` will be used.\n"
        )


# auto_docstring
class ZImageAutoVaeImageEncoderStep(AutoPipelineBlocks):
    """
    Vae Image Encoder step that encode the image to generate the image latents

      Components:
          vae (`AutoencoderKL`) image_processor (`VaeImageProcessor`)

      Inputs:
          image (`Image`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          image_latents (`Tensor`):
              video latent representation with the first frame image condition
    """

    block_classes = [ZImageVaeImageEncoderStep]
    block_names = ["vae_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self) -> str:
        return "Vae Image Encoder step that encode the image to generate the image latents"
        +"This is an auto pipeline block that works for image2image tasks."
        +" - `ZImageVaeImageEncoderStep` is used when `image` is provided."
        +" - if `image` is not provided, step will be skipped."


# auto_docstring
class ZImageAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image and image-to-image using ZImage.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `image`, `prompt`

      Components:
          text_encoder (`Qwen3Model`) tokenizer (`Qwen2Tokenizer`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKL`) image_processor (`VaeImageProcessor`) transformer (`ZImageTransformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          prompt (`None`, *optional*):
              TODO: Add description.
          negative_prompt (`None`, *optional*):
              TODO: Add description.
          max_sequence_length (`None`, *optional*, defaults to 512):
              TODO: Add description.
          image (`Image`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          image_latents (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`):
              TODO: Add description.
          num_inference_steps (`None`):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          strength (`None`, *optional*, defaults to 0.6):
              TODO: Add description.
          **denoiser_input_fields (`None`, *optional*):
              The conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          output_type (`str`, *optional*, defaults to pil):
              The type of the output images, can be 'pil', 'np', 'pt'

      Outputs:
          images (`list`):
              Generated images.
    """

    block_classes = [
        ZImageTextEncoderStep,
        ZImageAutoVaeImageEncoderStep,
        ZImageAutoDenoiseStep,
        ZImageVaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"image": True, "prompt": True},
    }

    @property
    def description(self) -> str:
        return "Auto Modular pipeline for text-to-image and image-to-image using ZImage."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
