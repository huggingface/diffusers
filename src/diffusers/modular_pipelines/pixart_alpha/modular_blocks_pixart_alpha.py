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
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    PixArtAlphaPrepareLatentsStep,
    PixArtAlphaPrepareMicroConditionsStep,
    PixArtAlphaSetTimestepsStep,
    PixArtAlphaTextInputStep,
)
from .decoders import PixArtAlphaDecodeStep, PixArtAlphaProcessImagesOutputStep
from .denoise import PixArtAlphaDenoiseStep
from .encoders import PixArtAlphaTextEncoderStep


logger = logging.get_logger(__name__)


# auto_docstring
class PixArtAlphaCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Components:
          scheduler (`DPMSolverMultistepScheduler`) transformer (`PixArtTransformer2DModel`) guider
          (`ClassifierFreeGuidance`)

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
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "pixart-alpha"
    block_classes = [
        PixArtAlphaTextInputStep(),
        PixArtAlphaSetTimestepsStep(),
        PixArtAlphaPrepareLatentsStep(),
        PixArtAlphaPrepareMicroConditionsStep(),
        PixArtAlphaDenoiseStep(),
    ]
    block_names = ["text_inputs", "set_timesteps", "prepare_latents", "prepare_micro_conditions", "denoise"]

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", PixArtAlphaTextEncoderStep()),
        ("denoise", PixArtAlphaCoreDenoiseStep()),
        ("decode", PixArtAlphaDecodeStep()),
        ("postprocess", PixArtAlphaProcessImagesOutputStep()),
    ]
)


# auto_docstring
class PixArtAlphaAutoBlocks(SequentialPipelineBlocks):
    """
    Supported workflows:
        - `text2image`: requires `prompt`

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`T5Tokenizer`) guider (`ClassifierFreeGuidance`) scheduler
          (`DPMSolverMultistepScheduler`) transformer (`PixArtTransformer2DModel`) vae (`AutoencoderKL`)
          image_processor (`VaeImageProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 120):
              Maximum sequence length for prompt encoding.
          clean_caption (`bool`, *optional*, defaults to True):
              Whether to clean the caption before encoding (requires the `bs4` and `ftfy` packages).
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "pixart-alpha"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    _workflow_map = {
        "text2image": {"prompt": True},
    }

    @property
    def outputs(self):
        return [OutputParam.template("images")]
