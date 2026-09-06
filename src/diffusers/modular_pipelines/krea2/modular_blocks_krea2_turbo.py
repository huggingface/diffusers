# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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
    Krea2PrepareLatentsStep,
    Krea2PreparePositionIdsStep,
    Krea2TurboSetTimestepsStep,
    Krea2TurboTextInputsStep,
)
from .decoders import Krea2DecodeStep
from .denoise import Krea2TurboDenoiseStep
from .encoders import Krea2TurboTextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


CORE_DENOISE_BLOCKS = InsertableDict(
    [
        ("input", Krea2TurboTextInputsStep()),
        ("prepare_latents", Krea2PrepareLatentsStep()),
        ("set_timesteps", Krea2TurboSetTimestepsStep()),
        ("prepare_position_ids", Krea2PreparePositionIdsStep()),
        ("denoise", Krea2TurboDenoiseStep()),
    ]
)


# auto_docstring
class Krea2TurboCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for the distilled Krea 2 turbo text-to-image checkpoint: prepares the
    batch/latents/timesteps and the shared position ids, then runs the guidance-free denoising loop, producing the
    denoised packed latents for the decoder.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The denoised packed latents (B, image_seq_len, in_channels).
    """

    model_name = "krea2"
    block_classes = list(CORE_DENOISE_BLOCKS.values())
    block_names = list(CORE_DENOISE_BLOCKS.keys())

    @property
    def description(self) -> str:
        return (
            "Core denoising workflow for the distilled Krea 2 turbo text-to-image checkpoint: prepares the "
            "batch/latents/timesteps and the shared position ids, then runs the guidance-free denoising loop, "
            "producing the denoised packed latents for the decoder."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents", description="The denoised packed latents (B, image_seq_len, in_channels).")
        ]


# auto_docstring
class Krea2TurboAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image generation using the distilled Krea 2 turbo checkpoint: encode text -> core
    denoise (guidance-free) -> decode.

      Supported workflows:
        - `text2image`: requires `prompt`

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`AutoTokenizer`): The tokenizer paired
          with the text encoder. transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          vae (`AutoencoderKLQwenImage`) image_processor (`VaeImageProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "krea2"
    block_classes = [
        Krea2TurboTextEncoderStep,
        Krea2TurboCoreDenoiseStep,
        Krea2DecodeStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    _workflow_map = {
        "text2image": {"prompt": True},
    }

    @property
    def description(self) -> str:
        return (
            "Auto Modular pipeline for text-to-image generation using the distilled Krea 2 turbo checkpoint: encode "
            "text -> core denoise (guidance-free) -> decode."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]
