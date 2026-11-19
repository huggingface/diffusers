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
    ChromaPrepareAttentionMaskStep,
    ChromaPrepareLatentsStep,
    ChromaRoPEInputsStep,
    ChromaSetTimestepsStep,
)
from .decoders import ChromaDecodeStep
from .denoise import ChromaDenoiseStep
from .encoders import ChromaTextEncoderStep
from .inputs import ChromaTextInputStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class ChromaCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core step that performs the denoising process for Chroma.
      This step takes the encoded conditions (prompt embeddings and attention masks) and runs the text-to-image
      denoising process.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`ChromaTransformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_attention_mask (`Tensor`, *optional*):
              Attention mask for the prompt embeddings. Can be generated from text_encoder step.
          negative_prompt_attention_mask (`Tensor`, *optional*):
              Attention mask for the negative prompt embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 35):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          joint_attention_kwargs (`dict`, *optional*):
              Additional kwargs passed along to the attention processors.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "chroma"
    block_classes = [
        ChromaTextInputStep,
        ChromaPrepareLatentsStep,
        ChromaSetTimestepsStep,
        ChromaPrepareAttentionMaskStep,
        ChromaRoPEInputsStep,
        ChromaDenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_attention_mask",
        "prepare_rope_inputs",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process for Chroma.\n"
            + "This step takes the encoded conditions (prompt embeddings and attention masks) and runs the "
            + "text-to-image denoising process."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", ChromaTextEncoderStep()),
        ("denoise", ChromaCoreDenoiseStep()),
        ("decode", ChromaDecodeStep()),
    ]
)


# auto_docstring
class ChromaAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image using Chroma.

      Supported workflows:
        - `text2image`: requires `prompt`

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`T5Tokenizer`) guider (`ClassifierFreeGuidance`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) transformer (`ChromaTransformer2DModel`) vae (`AutoencoderKL`)
          image_processor (`VaeImageProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.
          joint_attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors; `scale` is used as the text encoder LoRA scale.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 35):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "chroma"

    block_classes = TEXT2IMAGE_BLOCKS.values()
    block_names = TEXT2IMAGE_BLOCKS.keys()

    _workflow_map = {
        "text2image": {"prompt": True},
    }

    @property
    def description(self):
        return "Auto Modular pipeline for text-to-image using Chroma."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
