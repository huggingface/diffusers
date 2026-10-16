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

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    WanTextInputStep,
    WanVideoToVideoPrepareLatentsStep,
    WanVideoToVideoSetTimestepsStep,
)
from .decoders import WanVaeDecoderStep
from .denoise import WanDenoiseStep
from .encoders import WanTextEncoderStep, WanVideoVaeEncoderStep


# auto_docstring
class WanVideoToVideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise the noisy input video latents for video-to-video generation.

      Components:
          transformer (`WanTransformer3DModel`) scheduler (`UniPCMultistepScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`None`, *optional*, defaults to 1):
              The number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          num_inference_steps (`None`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`None`, *optional*):
              Custom timesteps for the denoising process.
          sigmas (`None`, *optional*):
              Custom sigmas for the denoising process.
          strength (`float`, *optional*, defaults to 0.8):
              The amount of noise added to the input video latents.
          video_latents (`Tensor`):
              Normalized VAE latents of the input video.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy video latents to use instead of adding noise to the input video.
          generator (`None`, *optional*):
              Torch generator for deterministic noise generation.
          attention_kwargs (`None`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "wan-v2v"
    block_classes = [
        WanTextInputStep,
        WanVideoToVideoSetTimestepsStep,
        WanVideoToVideoPrepareLatentsStep,
        WanDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise the noisy input video latents for video-to-video generation."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class WanVideoToVideoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for Wan video-to-video generation.

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKLWan`) video_processor (`VideoProcessor`) transformer (`WanTransformer3DModel`) scheduler
          (`UniPCMultistepScheduler`)

      Inputs:
          prompt (`None`, *optional*):
              The prompt or prompts to guide video generation.
          negative_prompt (`None`, *optional*):
              The prompt or prompts not to guide video generation.
          max_sequence_length (`None`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.
          video (`None`):
              The input video to transform.
          height (`int`, *optional*):
              The height in pixels of the generated video.
          width (`int`, *optional*):
              The width in pixels of the generated video.
          generator (`None`, *optional*):
              Torch generator for deterministic latent generation.
          num_videos_per_prompt (`None`, *optional*, defaults to 1):
              The number of videos to generate per prompt.
          num_inference_steps (`None`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`None`, *optional*):
              Custom timesteps for the denoising process.
          sigmas (`None`, *optional*):
              Custom sigmas for the denoising process.
          strength (`float`, *optional*, defaults to 0.8):
              The amount of noise added to the input video latents.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy video latents to use instead of adding noise to the input video.
          attention_kwargs (`None`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to np):
              The output type of the decoded videos

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "wan-v2v"
    block_classes = [
        WanTextEncoderStep,
        WanVideoVaeEncoderStep,
        WanVideoToVideoCoreDenoiseStep,
        WanVaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for Wan video-to-video generation."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
