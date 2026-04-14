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
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    LTXImage2VideoPrepareLatentsStep,
    LTXPrepareLatentsStep,
    LTXSetTimestepsStep,
    LTXTextInputStep,
)
from .decoders import LTXVaeDecoderStep
from .denoise import LTXDenoiseStep, LTXImage2VideoDenoiseStep
from .encoders import LTXTextEncoderStep, LTXVaeEncoderStep


logger = logging.get_logger(__name__)


# auto_docstring
class LTXCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block that takes encoded conditions and runs the denoising process.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) pachifier (`LTXVideoPachifier`) guider
          (`ClassifierFreeGuidance`) transformer (`LTXVideoTransformer3DModel`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
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
    Denoise block for image-to-video that takes encoded conditions and image latents, and runs the denoising process.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) pachifier (`LTXVideoPachifier`) guider
          (`ClassifierFreeGuidance`) transformer (`LTXVideoTransformer3DModel`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
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
          image_latents (`Tensor`):
              TODO: Add description.
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
        LTXImage2VideoPrepareLatentsStep,
        LTXImage2VideoDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "prepare_i2v_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block for image-to-video that takes encoded conditions and image latents, and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class LTXBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX Video text-to-video.

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`T5TokenizerFast`) guider (`ClassifierFreeGuidance`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) pachifier (`LTXVideoPachifier`) transformer
          (`LTXVideoTransformer3DModel`) vae (`AutoencoderKLLTXVideo`) video_processor (`VideoProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 128):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
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
class LTXAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the image input into its latent representation.
      This is an auto pipeline block that works for image-to-video tasks.
       - `LTXVaeEncoderStep` is used when `image` is provided.
       - If `image` is not provided, step will be skipped.

      Components:
          vae (`AutoencoderKLLTXVideo`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          image_latents (`Tensor`):
              Encoded image latents from the VAE encoder
    """

    model_name = "ltx"
    block_classes = [LTXVaeEncoderStep]
    block_names = ["vae_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the image input into its latent representation.\n"
            "This is an auto pipeline block that works for image-to-video tasks.\n"
            " - `LTXVaeEncoderStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


# auto_docstring
class LTXAutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Auto denoise block that selects the appropriate denoise pipeline based on inputs.
       - `LTXImage2VideoCoreDenoiseStep` is used when `image_latents` is provided.
       - `LTXCoreDenoiseStep` is used otherwise (text-to-video).

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) pachifier (`LTXVideoPachifier`) guider
          (`ClassifierFreeGuidance`) transformer (`LTXVideoTransformer3DModel`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_attention_mask (`Tensor`):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_attention_mask (`Tensor`):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
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
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image_latents (`Tensor`, *optional*):
              TODO: Add description.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "ltx"
    block_classes = [LTXImage2VideoCoreDenoiseStep, LTXCoreDenoiseStep]
    block_names = ["image2video", "text2video"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Auto denoise block that selects the appropriate denoise pipeline based on inputs.\n"
            " - `LTXImage2VideoCoreDenoiseStep` is used when `image_latents` is provided.\n"
            " - `LTXCoreDenoiseStep` is used otherwise (text-to-video)."
        )


# auto_docstring
class LTXAutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks for LTX Video that support both text-to-video and image-to-video workflows.

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `image`, `prompt`

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`T5TokenizerFast`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKLLTXVideo`) video_processor (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          pachifier (`LTXVideoPachifier`) transformer (`LTXVideoTransformer3DModel`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 128):
              Maximum sequence length for prompt encoding.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 161):
              TODO: Add description.
          frame_rate (`int`, *optional*, defaults to 25):
              TODO: Add description.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          image_latents (`Tensor`, *optional*):
              TODO: Add description.
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
        LTXAutoVaeEncoderStep,
        LTXAutoCoreDenoiseStep,
        LTXVaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
    }

    @property
    def description(self):
        return "Auto blocks for LTX Video that support both text-to-video and image-to-video workflows."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]


# auto_docstring
class LTXImage2VideoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX Video image-to-video.

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`T5TokenizerFast`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKLLTXVideo`) video_processor (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          pachifier (`LTXVideoPachifier`) transformer (`LTXVideoTransformer3DModel`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 128):
              Maximum sequence length for prompt encoding.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 161):
              TODO: Add description.
          frame_rate (`int`, *optional*, defaults to 25):
              TODO: Add description.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          image_latents (`Tensor`):
              TODO: Add description.
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
        LTXAutoVaeEncoderStep,
        LTXImage2VideoCoreDenoiseStep,
        LTXVaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX Video image-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
