# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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
    ErnieImagePrepareLatentsStep,
    ErnieImageSetTimestepsStep,
    ErnieImageTextInputStep,
)
from .decoders import ErnieImageVaeDecoderStep
from .denoise import ErnieImageDenoiseStep
from .encoders import ErnieImagePromptEnhancerStep, ErnieImageTextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class ErnieImageAutoPromptEnhancerStep(AutoPipelineBlocks):
    """
    Auto block that runs the optional prompt enhancer when `use_pe` is provided.
       - `ErnieImagePromptEnhancerStep` is used when `use_pe` is set.
       - If `use_pe` is not provided, the step is skipped.

      Components:
          pe (`AutoModelForCausalLM`) pe_tokenizer (`AutoTokenizer`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          pe_system_prompt (`str`, *optional*):
              Optional system prompt passed to the prompt enhancer.
          pe_temperature (`float`, *optional*, defaults to 0.6):
              Sampling temperature used when generating with the prompt enhancer.
          pe_top_p (`float`, *optional*, defaults to 0.95):
              Nucleus sampling `top_p` used when generating with the prompt enhancer.

      Outputs:
          prompt (`list`):
              The prompt list after prompt-enhancer rewriting.
          height (`int`):
              The resolved image height in pixels.
          width (`int`):
              The resolved image width in pixels.
    """

    model_name = "ernie-image"
    block_classes = [ErnieImagePromptEnhancerStep]
    block_names = ["prompt_enhancer"]
    block_trigger_inputs = ["use_pe"]

    @property
    def description(self):
        return (
            "Auto block that runs the optional prompt enhancer when `use_pe` is provided.\n"
            " - `ErnieImagePromptEnhancerStep` is used when `use_pe` is set.\n"
            " - If `use_pe` is not provided, the step is skipped."
        )


# auto_docstring
class ErnieImageCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block that takes encoded conditions and runs the denoising process for ErnieImage.

      Components:
          transformer (`ErnieImageTransformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          prompt_embeds (`list`):
              List of per-prompt text embeddings from the text encoder step.
          negative_prompt_embeds (`list`, *optional*):
              List of per-prompt negative text embeddings from the text encoder step.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              Number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 50):
              Number of denoising steps.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents. If provided, skips noise sampling.
          generator (`Generator`, *optional*):
              Torch generator for deterministic noise sampling.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "ernie-image"
    block_classes = [
        ErnieImageTextInputStep,
        ErnieImageSetTimestepsStep,
        ErnieImagePrepareLatentsStep,
        ErnieImageDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block that takes encoded conditions and runs the denoising process for ErnieImage."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class ErnieImageAutoBlocks(SequentialPipelineBlocks):
    """
    Auto modular pipeline for ErnieImage text-to-image generation. Supports an optional prompt enhancer when the `pe`
    components are loaded and `use_pe=True`.

      Supported workflows:
        - `text2image`: requires `prompt`

      Components:
          pe (`AutoModelForCausalLM`) pe_tokenizer (`AutoTokenizer`) text_encoder (`AutoModel`) tokenizer
          (`AutoTokenizer`) guider (`ClassifierFreeGuidance`) transformer (`ErnieImageTransformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) vae (`AutoencoderKLFlux2`) pachifier (`ErnieImagePachifier`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          pe_system_prompt (`str`, *optional*):
              Optional system prompt passed to the prompt enhancer.
          pe_temperature (`float`, *optional*, defaults to 0.6):
              Sampling temperature used when generating with the prompt enhancer.
          pe_top_p (`float`, *optional*, defaults to 0.95):
              Nucleus sampling `top_p` used when generating with the prompt enhancer.
          negative_prompt (`str`, *optional*):
              The prompt or prompts to avoid during image generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              Number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 50):
              Number of denoising steps.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents. If provided, skips noise sampling.
          generator (`Generator`, *optional*):
              Torch generator for deterministic noise sampling.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', or 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "ernie-image"
    block_classes = [
        ErnieImageAutoPromptEnhancerStep,
        ErnieImageTextEncoderStep,
        ErnieImageCoreDenoiseStep,
        ErnieImageVaeDecoderStep,
    ]
    block_names = ["prompt_enhancer", "text_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2image": {"prompt": True},
    }

    @property
    def description(self):
        return (
            "Auto modular pipeline for ErnieImage text-to-image generation. Supports an optional prompt enhancer "
            "when the `pe` components are loaded and `use_pe=True`."
        )

    @property
    def outputs(self):
        return [OutputParam.template("images")]
