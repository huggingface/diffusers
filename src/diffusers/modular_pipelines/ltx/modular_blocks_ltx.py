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
    LTXImage2VideoPrepareLatentsStep,
    LTXPrepareLatentsStep,
    LTXSetTimestepsStep,
    LTXTextInputStep,
)
from .decoders import LTXVaeDecoderStep
from .denoise import LTXDenoiseStep, LTXImage2VideoDenoiseStep
from .encoders import LTXTextEncoderStep


logger = logging.get_logger(__name__)


# auto_docstring
class LTXCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block that takes encoded conditions and runs the denoising process.

      Components:
          transformer (`LTXVideoTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`)
          guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          guidance_scale (`float`, *optional*, defaults to 3.0):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_attention_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_attention_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 161):
              TODO: Add description.
          frame_rate (`int`, *optional*, defaults to 25):
              TODO: Add description.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "ltx"
    block_classes = [
        LTXTextInputStep,
        LTXSetTimestepsStep,
        LTXPrepareLatentsStep,
        LTXDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class LTXImage2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block for image-to-video that takes encoded conditions and an image, and runs the denoising process.

      Components:
          transformer (`LTXVideoTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`)
          vae (`AutoencoderKLLTXVideo`)
          guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          guidance_scale (`float`, *optional*, defaults to 3.0):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_attention_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_attention_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 161):
              TODO: Add description.
          frame_rate (`int`, *optional*, defaults to 25):
              TODO: Add description.
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "ltx"
    block_classes = [
        LTXTextInputStep,
        LTXSetTimestepsStep,
        LTXImage2VideoPrepareLatentsStep,
        LTXImage2VideoDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block for image-to-video that takes encoded conditions and an image, and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class LTXBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX Video text-to-video.

      Components:
          text_encoder (`T5EncoderModel`)
          tokenizer (`T5TokenizerFast`)
          guider (`ClassifierFreeGuidance`)
          transformer (`LTXVideoTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`)
          vae (`AutoencoderKLLTXVideo`)
          video_processor (`VideoProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_attention_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_attention_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          max_sequence_length (`int`, *optional*, defaults to 128):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          guidance_scale (`float`, *optional*, defaults to 3.0):
              TODO: Add description.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 161):
              TODO: Add description.
          frame_rate (`int`, *optional*, defaults to 25):
              TODO: Add description.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              TODO: Add description.
          decode_noise_scale (`None`, *optional*):
              TODO: Add description.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "ltx"
    block_classes = [
        LTXTextEncoderStep,
        LTXCoreDenoiseStep,
        LTXVaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX Video text-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]


# auto_docstring
class LTXImage2VideoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX Video image-to-video.

      Components:
          text_encoder (`T5EncoderModel`)
          tokenizer (`T5TokenizerFast`)
          guider (`ClassifierFreeGuidance`)
          transformer (`LTXVideoTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`)
          vae (`AutoencoderKLLTXVideo`)
          video_processor (`VideoProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_attention_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_attention_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          max_sequence_length (`int`, *optional*, defaults to 128):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          guidance_scale (`float`, *optional*, defaults to 3.0):
              TODO: Add description.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 161):
              TODO: Add description.
          frame_rate (`int`, *optional*, defaults to 25):
              TODO: Add description.
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              TODO: Add description.
          decode_noise_scale (`None`, *optional*):
              TODO: Add description.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "ltx"
    block_classes = [
        LTXTextEncoderStep,
        LTXImage2VideoCoreDenoiseStep,
        LTXVaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX Video image-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
