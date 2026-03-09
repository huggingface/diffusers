# Copyright 2025 The HuggingFace Team. All rights reserved.
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
    WanAdditionalInputsStep,
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanTextInputStep,
)
from .decoders import WanVaeDecoderStep
from .denoise import (
    Wan22Image2VideoDenoiseStep,
)
from .encoders import (
    WanImageResizeStep,
    WanPrepareFirstFrameLatentsStep,
    WanTextEncoderStep,
    WanVaeEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. VAE ENCODER
# ====================


# auto_docstring
class WanImage2VideoVaeEncoderStep(SequentialPipelineBlocks):
    """
    Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent
    representation

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.
          num_frames (`int`, *optional*, defaults to 81):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          resized_image (`Image`):
              TODO: Add description.
          first_frame_latents (`Tensor`):
              video latent representation with the first frame image condition
          image_condition_latents (`Tensor | NoneType`):
              TODO: Add description.
    """

    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanVaeEncoderStep, WanPrepareFirstFrameLatentsStep]
    block_names = ["image_resize", "vae_encoder", "prepare_first_frame_latents"]

    @property
    def description(self):
        return "Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent representation"


# ====================
# 2. DENOISE
# ====================


# inputs (text + image_condition_latents) -> set_timesteps -> prepare_latents -> denoise (latents)
# auto_docstring
class Wan22Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    denoise block that takes encoded text and image latent conditions and runs the denoising process.

      Components:
          transformer (`WanTransformer3DModel`) scheduler (`UniPCMultistepScheduler`) guider (`ClassifierFreeGuidance`)
          guider_2 (`ClassifierFreeGuidance`) transformer_2 (`WanTransformer3DModel`)

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
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          num_frames (`None`, *optional*):
              TODO: Add description.
          image_condition_latents (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          attention_kwargs (`None`, *optional*):
              TODO: Add description.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "wan-i2v"
    block_classes = [
        WanTextInputStep,
        WanAdditionalInputsStep(image_latent_inputs=["image_condition_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        Wan22Image2VideoDenoiseStep,
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
        return "denoise block that takes encoded text and image latent conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# ====================
# 3. BLOCKS (Wan2.2 Image2Video)
# ====================


# auto_docstring
class Wan22Image2VideoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline for image-to-video using Wan2.2.

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKLWan`) video_processor (`VideoProcessor`) transformer (`WanTransformer3DModel`) scheduler
          (`UniPCMultistepScheduler`) guider_2 (`ClassifierFreeGuidance`) transformer_2 (`WanTransformer3DModel`)

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
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
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

    model_name = "wan-i2v"
    block_classes = [
        WanTextEncoderStep,
        WanImage2VideoVaeEncoderStep,
        Wan22Image2VideoCoreDenoiseStep,
        WanVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "vae_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return "Modular pipeline for image-to-video using Wan2.2."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
