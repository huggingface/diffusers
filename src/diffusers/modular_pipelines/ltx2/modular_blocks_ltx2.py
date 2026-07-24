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

import torch

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    LTX2Image2VideoPrepareLatentsStep,
    LTX2PrepareAudioLatentsStep,
    LTX2PrepareCoordsStep,
    LTX2PrepareLatentsStep,
    LTX2SetTimestepsStep,
    LTX2TextInputStep,
)
from .decoders import LTX2AudioDecoderStep, LTX2VaeDecoderStep
from .denoise import LTX2DenoiseStep, LTX2Image2VideoDenoiseStep
from .encoders import (
    LTX2ImageToVideoPromptEnhancerStep,
    LTX2PromptEnhancerStep,
    LTX2TextConnectorStep,
    LTX2TextEncoderStep,
    LTX2VaeEncoderStep,
)


logger = logging.get_logger(__name__)


# auto_docstring
class LTX2AutoPromptEnhancerStep(ConditionalPipelineBlocks):
    """
    Conditional prompt-enhancer step, run only when `enable_prompt_enhancement` is truthy.
       - `LTX2ImageToVideoPromptEnhancerStep` when an `image` is provided (image-to-video).
       - `LTX2PromptEnhancerStep` otherwise (text-to-video).
       - Skipped when `enable_prompt_enhancement` is falsy / not provided.

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          system_prompt (`str`, *optional*):
              System prompt for enhancement. Defaults to `LTX2_4_I2V_DEFAULT_SYSTEM_PROMPT`.
          prompt_max_new_tokens (`int`, *optional*, defaults to 512):
              Maximum number of new tokens to generate during prompt enhancement.
          prompt_enhancement_kwargs (`dict`, *optional*):
              Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.
          prompt_enhancement_seed (`int`, *optional*, defaults to 10):
              Random seed for prompt enhancement (inert under LTX-2.4's greedy decoding).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          prompt (`list`):
              The prompt(s) after prompt-enhancer rewriting.
    """

    model_name = "ltx2"
    block_classes = [LTX2ImageToVideoPromptEnhancerStep, LTX2PromptEnhancerStep]
    block_names = ["image2video", "text2video"]
    block_trigger_inputs = ["image", "enable_prompt_enhancement"]

    def select_block(self, image=None, enable_prompt_enhancement=None) -> str | None:
        if not enable_prompt_enhancement:
            return None
        return "image2video" if image is not None else "text2video"

    @property
    def description(self):
        return (
            "Conditional prompt-enhancer step, run only when `enable_prompt_enhancement` is truthy.\n"
            " - `LTX2ImageToVideoPromptEnhancerStep` when an `image` is provided (image-to-video).\n"
            " - `LTX2PromptEnhancerStep` otherwise (text-to-video).\n"
            " - Skipped when `enable_prompt_enhancement` is falsy / not provided."
        )


# auto_docstring
class LTX2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (text-to-video) that prepares video/audio latents and runs the joint denoising loop.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              TODO: Add description.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          noise_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              TODO: Add description.
          frame_rate (`float`, *optional*, defaults to 24.0):
              TODO: Add description.
          audio_latents (`Tensor`, *optional*):
              TODO: Add description.
          dtype (`dtype`):
              TODO: Add description.
          connector_prompt_embeds (`Tensor`):
              TODO: Add description.
          connector_audio_prompt_embeds (`Tensor`):
              TODO: Add description.
          connector_attention_mask (`Tensor`):
              TODO: Add description.
          negative_connector_prompt_embeds (`Tensor`):
              TODO: Add description.
          negative_connector_audio_prompt_embeds (`Tensor`):
              TODO: Add description.
          negative_connector_attention_mask (`Tensor`):
              TODO: Add description.
          guidance_scale (`float`, *optional*, defaults to 4.0):
              TODO: Add description.
          audio_guidance_scale (`float`, *optional*):
              TODO: Add description.
          stg_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_stg_scale (`float`, *optional*):
              TODO: Add description.
          modality_scale (`float`, *optional*, defaults to 1.0):
              TODO: Add description.
          audio_modality_scale (`float`, *optional*):
              TODO: Add description.
          guidance_rescale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_guidance_rescale (`float`, *optional*):
              TODO: Add description.
          spatio_temporal_guidance_blocks (`list`, *optional*):
              TODO: Add description.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              TODO: Add description.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2SetTimestepsStep,
        LTX2PrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordsStep,
        LTX2DenoiseStep,
    ]
    block_names = ["set_timesteps", "prepare_latents", "prepare_audio_latents", "prepare_coords", "denoise"]

    @property
    def description(self):
        return "Denoise block (text-to-video) that prepares video/audio latents and runs the joint denoising loop."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX2Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (image-to-video) that adds image conditioning and runs the joint denoising loop.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`)

      Inputs:
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              TODO: Add description.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          noise_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              TODO: Add description.
          image_latents (`Tensor`):
              TODO: Add description.
          frame_rate (`float`, *optional*, defaults to 24.0):
              TODO: Add description.
          audio_latents (`Tensor`, *optional*):
              TODO: Add description.
          dtype (`dtype`):
              TODO: Add description.
          connector_prompt_embeds (`Tensor`):
              TODO: Add description.
          connector_audio_prompt_embeds (`Tensor`):
              TODO: Add description.
          connector_attention_mask (`Tensor`):
              TODO: Add description.
          negative_connector_prompt_embeds (`Tensor`):
              TODO: Add description.
          negative_connector_audio_prompt_embeds (`Tensor`):
              TODO: Add description.
          negative_connector_attention_mask (`Tensor`):
              TODO: Add description.
          guidance_scale (`float`, *optional*, defaults to 4.0):
              TODO: Add description.
          audio_guidance_scale (`float`, *optional*):
              TODO: Add description.
          stg_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_stg_scale (`float`, *optional*):
              TODO: Add description.
          modality_scale (`float`, *optional*, defaults to 1.0):
              TODO: Add description.
          audio_modality_scale (`float`, *optional*):
              TODO: Add description.
          guidance_rescale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_guidance_rescale (`float`, *optional*):
              TODO: Add description.
          spatio_temporal_guidance_blocks (`list`, *optional*):
              TODO: Add description.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              TODO: Add description.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2SetTimestepsStep,
        LTX2PrepareLatentsStep,
        LTX2Image2VideoPrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordsStep,
        LTX2Image2VideoDenoiseStep,
    ]
    block_names = [
        "set_timesteps",
        "prepare_latents",
        "prepare_i2v_latents",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
    ]

    @property
    def description(self):
        return "Denoise block (image-to-video) that adds image conditioning and runs the joint denoising loop."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX2AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the reference `image` into latents for image-to-video.
       - `LTX2VaeEncoderStep` runs when `image` is provided.
       - Skipped otherwise.

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`)

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
              Normalized image latents (a single latent frame) for image-to-video conditioning.
    """

    model_name = "ltx2"
    block_classes = [LTX2VaeEncoderStep]
    block_names = ["vae_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the reference `image` into latents for image-to-video.\n"
            " - `LTX2VaeEncoderStep` runs when `image` is provided.\n"
            " - Skipped otherwise."
        )


# auto_docstring
class LTX2AutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Auto denoise block that selects the workflow based on inputs.
       - `LTX2Image2VideoCoreDenoiseStep` when `image_latents` is provided.
       - `LTX2CoreDenoiseStep` otherwise (text-to-video).

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`)

      Inputs:
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
          num_frames (`int`, *optional*, defaults to 121):
              TODO: Add description.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          noise_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              TODO: Add description.
          image_latents (`Tensor`, *optional*):
              TODO: Add description.
          frame_rate (`float`, *optional*, defaults to 24.0):
              TODO: Add description.
          audio_latents (`Tensor`):
              TODO: Add description.
          dtype (`dtype`):
              TODO: Add description.
          connector_prompt_embeds (`Tensor`):
              TODO: Add description.
          connector_audio_prompt_embeds (`Tensor`):
              TODO: Add description.
          connector_attention_mask (`Tensor`):
              TODO: Add description.
          negative_connector_prompt_embeds (`Tensor`):
              TODO: Add description.
          negative_connector_audio_prompt_embeds (`Tensor`):
              TODO: Add description.
          negative_connector_attention_mask (`Tensor`):
              TODO: Add description.
          guidance_scale (`float`, *optional*, defaults to 4.0):
              TODO: Add description.
          audio_guidance_scale (`float`, *optional*):
              TODO: Add description.
          stg_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_stg_scale (`float`, *optional*):
              TODO: Add description.
          modality_scale (`float`, *optional*, defaults to 1.0):
              TODO: Add description.
          audio_modality_scale (`float`, *optional*):
              TODO: Add description.
          guidance_rescale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_guidance_rescale (`float`, *optional*):
              TODO: Add description.
          spatio_temporal_guidance_blocks (`list`, *optional*):
              TODO: Add description.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              TODO: Add description.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    block_classes = [LTX2Image2VideoCoreDenoiseStep, LTX2CoreDenoiseStep]
    block_names = ["image2video", "text2video"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Auto denoise block that selects the workflow based on inputs.\n"
            " - `LTX2Image2VideoCoreDenoiseStep` when `image_latents` is provided.\n"
            " - `LTX2CoreDenoiseStep` otherwise (text-to-video)."
        )


# auto_docstring
class LTX2Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX-2 text-to-video (joint video + audio).

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`)
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) vocoder
          (`LTX2Vocoder`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              TODO: Add description.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          frame_rate (`float`, *optional*, defaults to 24.0):
              TODO: Add description.
          audio_latents (`Tensor`, *optional*):
              TODO: Add description.
          guidance_scale (`float`, *optional*, defaults to 4.0):
              TODO: Add description.
          audio_guidance_scale (`float`, *optional*):
              TODO: Add description.
          stg_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_stg_scale (`float`, *optional*):
              TODO: Add description.
          modality_scale (`float`, *optional*, defaults to 1.0):
              TODO: Add description.
          audio_modality_scale (`float`, *optional*):
              TODO: Add description.
          guidance_rescale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_guidance_rescale (`float`, *optional*):
              TODO: Add description.
          spatio_temporal_guidance_blocks (`list`, *optional*):
              TODO: Add description.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              TODO: Add description.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              TODO: Add description.
          decode_noise_scale (`None`, *optional*):
              TODO: Add description.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextEncoderStep,
        LTX2TextInputStep,
        LTX2TextConnectorStep,
        LTX2CoreDenoiseStep,
        LTX2VaeDecoderStep,
        LTX2AudioDecoderStep,
    ]
    block_names = ["text_encoder", "text_input", "connectors", "denoise", "video_decode", "audio_decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX-2 text-to-video (joint video + audio)."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2ImageToVideoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX-2 image-to-video (joint video + audio).

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) vae
          (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          transformer (`LTX2VideoTransformer3DModel`) audio_vae (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 121):
              TODO: Add description.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          image_latents (`Tensor`):
              TODO: Add description.
          frame_rate (`float`, *optional*, defaults to 24.0):
              TODO: Add description.
          audio_latents (`Tensor`, *optional*):
              TODO: Add description.
          guidance_scale (`float`, *optional*, defaults to 4.0):
              TODO: Add description.
          audio_guidance_scale (`float`, *optional*):
              TODO: Add description.
          stg_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_stg_scale (`float`, *optional*):
              TODO: Add description.
          modality_scale (`float`, *optional*, defaults to 1.0):
              TODO: Add description.
          audio_modality_scale (`float`, *optional*):
              TODO: Add description.
          guidance_rescale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_guidance_rescale (`float`, *optional*):
              TODO: Add description.
          spatio_temporal_guidance_blocks (`list`, *optional*):
              TODO: Add description.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              TODO: Add description.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              TODO: Add description.
          decode_noise_scale (`None`, *optional*):
              TODO: Add description.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextEncoderStep,
        LTX2TextInputStep,
        LTX2TextConnectorStep,
        LTX2AutoVaeEncoderStep,
        LTX2Image2VideoCoreDenoiseStep,
        LTX2VaeDecoderStep,
        LTX2AudioDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "text_input",
        "connectors",
        "vae_encoder",
        "denoise",
        "video_decode",
        "audio_decode",
    ]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX-2 image-to-video (joint video + audio)."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2AutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks for LTX-2 supporting both text-to-video and image-to-video (joint video + audio).

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `image`, `prompt`

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) vae (`AutoencoderKLLTX2Video`) video_processor
          (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`)
          audio_vae (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          system_prompt (`str`, *optional*):
              System prompt for enhancement. Defaults to `LTX2_4_I2V_DEFAULT_SYSTEM_PROMPT`.
          prompt_max_new_tokens (`int`, *optional*, defaults to 512):
              Maximum number of new tokens to generate during prompt enhancement.
          prompt_enhancement_kwargs (`dict`, *optional*):
              Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.
          prompt_enhancement_seed (`int`, *optional*, defaults to 10):
              Random seed for prompt enhancement (inert under LTX-2.4's greedy decoding).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 121):
              TODO: Add description.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          image_latents (`Tensor`, *optional*):
              TODO: Add description.
          frame_rate (`float`, *optional*, defaults to 24.0):
              TODO: Add description.
          audio_latents (`Tensor`):
              TODO: Add description.
          guidance_scale (`float`, *optional*, defaults to 4.0):
              TODO: Add description.
          audio_guidance_scale (`float`, *optional*):
              TODO: Add description.
          stg_scale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_stg_scale (`float`, *optional*):
              TODO: Add description.
          modality_scale (`float`, *optional*, defaults to 1.0):
              TODO: Add description.
          audio_modality_scale (`float`, *optional*):
              TODO: Add description.
          guidance_rescale (`float`, *optional*, defaults to 0.0):
              TODO: Add description.
          audio_guidance_rescale (`float`, *optional*):
              TODO: Add description.
          spatio_temporal_guidance_blocks (`list`, *optional*):
              TODO: Add description.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              TODO: Add description.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              TODO: Add description.
          decode_noise_scale (`None`, *optional*):
              TODO: Add description.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextEncoderStep,
        LTX2TextInputStep,
        LTX2TextConnectorStep,
        LTX2AutoVaeEncoderStep,
        LTX2AutoCoreDenoiseStep,
        LTX2VaeDecoderStep,
        LTX2AudioDecoderStep,
    ]
    block_names = [
        "prompt_enhancer",
        "text_encoder",
        "text_input",
        "connectors",
        "vae_encoder",
        "denoise",
        "video_decode",
        "audio_decode",
    ]
    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
    }

    @property
    def description(self):
        return "Auto blocks for LTX-2 supporting both text-to-video and image-to-video (joint video + audio)."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]
