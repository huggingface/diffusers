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
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanTextInputStep,
    WanVaceAdditionalInputsStep,
)
from .decoders import WanVaceTrimReferenceLatentsStep, WanVaeDecoderStep
from .denoise import (
    Wan22VaceDenoiseStep,
)
from .encoders import (
    WanTextEncoderStep,
    WanVaceEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. DENOISE
# ====================

# inputs (text + vace_conditioning_latents) -> additional_inputs -> set_timesteps -> prepare_latents -> denoise


# auto_docstring
class Wan22VaceCoreDenoiseStep(SequentialPipelineBlocks):
    """
    denoise block that takes encoded text and vace conditioning latents and runs the denoising process.

      Components:
          transformer (`WanTransformer3DModel`) scheduler (`UniPCMultistepScheduler`) guider (`ClassifierFreeGuidance`)
          guider_2 (`ClassifierFreeGuidance`) transformer_2 (`WanVACETransformer3DModel`)

      Configs:
          boundary_ratio (default: 0.875): The boundary ratio to divide the denoising loop into high noise and low
          noise stages.

      Inputs:
          num_videos_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          num_frames (`int`, *optional*):
              TODO: Add description.
          num_reference_images (`int`, *optional*, defaults to 0):
              Number of reference images prepended on the frame dimension of the conditioning latents. Can be generated
              in vace_encoder step.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          attention_kwargs (`None`, *optional*):
              TODO: Add description.
          vace_conditioning_latents (`Tensor`):
              The conditioning latents fed into the VACE control branch of the transformer. Can be generated in
              vace_encoder step.
          conditioning_scale (`Tensor`):
              The per-layer conditioning scale tensor applied to the VACE control branch. Can be generated in
              vace_encoder step.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "wan-vace"
    block_classes = [
        WanTextInputStep,
        WanVaceAdditionalInputsStep,
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        Wan22VaceDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "set_timesteps",
        "prepare_latents",
        "denoise",
    ]

    @property
    def description(self):
        return "denoise block that takes encoded text and vace conditioning latents and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# ====================
# 2. BLOCKS (Wan2.2 VACE)
# ====================


# auto_docstring
class Wan22VaceBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline for controllable video generation using Wan2.2 VACE.

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) guider (`ClassifierFreeGuidance`) transformer
          (`WanVACETransformer3DModel`) vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`) scheduler
          (`UniPCMultistepScheduler`) guider_2 (`ClassifierFreeGuidance`) transformer_2 (`WanVACETransformer3DModel`)

      Configs:
          boundary_ratio (default: 0.875): The boundary ratio to divide the denoising loop into high noise and low
          noise stages.

      Inputs:
          prompt (`None`, *optional*):
              TODO: Add description.
          negative_prompt (`None`, *optional*):
              TODO: Add description.
          max_sequence_length (`None`, *optional*, defaults to 512):
              TODO: Add description.
          video (`list`, *optional*):
              The control video to condition the generation on. If not provided, an empty video is used.
          mask (`list`, *optional*):
              The mask that defines which video regions to condition on (black) and which to generate (white). Can only
              be passed if `video` is passed as well.
          reference_images (`Image | list`, *optional*):
              One or more reference images as extra conditioning for the generation.
          conditioning_scale (`float | list | Tensor`, *optional*, defaults to 1.0):
              The conditioning scale applied in each control layer of the model. If a float, it is applied uniformly to
              all layers; a list or tensor must have the same length as the number of control layers.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          num_frames (`int`, *optional*, defaults to 81):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_videos_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          attention_kwargs (`None`, *optional*):
              TODO: Add description.
          output_type (`str`, *optional*, defaults to np):
              The output type of the decoded videos

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "wan-vace"
    block_classes = [
        WanTextEncoderStep,
        WanVaceEncoderStep,
        Wan22VaceCoreDenoiseStep,
        WanVaceTrimReferenceLatentsStep,
        WanVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "vace_encoder",
        "denoise",
        "trim_latents",
        "decode",
    ]

    @property
    def description(self):
        return "Modular pipeline for controllable video generation using Wan2.2 VACE."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
