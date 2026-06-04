# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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
    Ideogram4PrepareAdditionalInputsStep,
    Ideogram4PrepareLatentsStep,
    Ideogram4SetTimestepsStep,
    Ideogram4TextInputsStep,
)
from .decoders import Ideogram4DecodeStep
from .denoise import Ideogram4AfterDenoiseStep, Ideogram4DenoiseStep
from .encoders import Ideogram4TextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Core denoise: consumes the per-prompt text features and produces the unpatchified latents
# (batch/latents/timesteps/ids inputs -> denoising loop -> unpatchify).
CORE_DENOISE_BLOCKS = InsertableDict(
    [
        ("input", Ideogram4TextInputsStep()),
        ("prepare_latents", Ideogram4PrepareLatentsStep()),
        ("set_timesteps", Ideogram4SetTimestepsStep()),
        ("prepare_additional_inputs", Ideogram4PrepareAdditionalInputsStep()),
        ("denoise", Ideogram4DenoiseStep()),
        ("after_denoise", Ideogram4AfterDenoiseStep()),
    ]
)


# auto_docstring
class Ideogram4CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for Ideogram4 text-to-image: prepares the batch/latents/timesteps and the packed denoiser
    inputs, runs the asymmetric-CFG denoising loop over the conditional and unconditional transformers, and
    unpatchifies the result for the decoder.

      Components:
          transformer (`Ideogram4Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          unconditional_transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          text_features (`Tensor`):
              Per-prompt text features from the encoder.
          text_lengths (`list`):
              Per-prompt text-token counts from the encoder.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).

      Outputs:
          latents (`Tensor`):
              Unpatchified (B, ae_channels, H, W) latents.
    """

    model_name = "ideogram4"
    block_classes = list(CORE_DENOISE_BLOCKS.values())
    block_names = list(CORE_DENOISE_BLOCKS.keys())

    @property
    def description(self) -> str:
        return (
            "Core denoising workflow for Ideogram4 text-to-image: prepares the batch/latents/timesteps and the packed "
            "denoiser inputs, runs the asymmetric-CFG denoising loop over the conditional and unconditional "
            "transformers, and unpatchifies the result for the decoder."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        # The only meaningful product of the core step is the unpatchified latents; the batch/timesteps/packed-sequence
        # inputs prepared along the way are consumed within the loop and are not updated by it.
        return [OutputParam.template("latents", description="Unpatchified (B, ae_channels, H, W) latents.")]


# auto_docstring
class Ideogram4AutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image generation using Ideogram4: encode text -> core denoise (asymmetric CFG
    over two transformers) -> decode.

      Supported workflows:
        - `text2image`: requires `prompt`

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`Qwen2Tokenizer`): The tokenizer paired
          with the text encoder. transformer (`Ideogram4Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) unconditional_transformer (`Ideogram4Transformer2DModel`) vae
          (`AutoencoderKLFlux2`) image_processor (`VaeImageProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 2048):
              Maximum sequence length for prompt encoding.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "ideogram4"
    block_classes = [Ideogram4TextEncoderStep(), Ideogram4CoreDenoiseStep(), Ideogram4DecodeStep()]
    block_names = ["text_encoder", "denoise", "decode"]

    # Workflow map declaring the trigger conditions for each supported workflow.
    # `True` means the workflow triggers when the input is not None.
    _workflow_map = {
        "text2image": {"prompt": True},
    }

    @property
    def description(self) -> str:
        return (
            "Auto Modular pipeline for text-to-image generation using Ideogram4: encode text -> core denoise "
            "(asymmetric CFG over two transformers) -> decode."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]
