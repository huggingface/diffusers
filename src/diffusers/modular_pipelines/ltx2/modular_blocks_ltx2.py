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

# NOTE (modular.md gotcha #1): `LTX2AutoDuration` is imported from `diffusers.pipelines.ltx2.*` only for the
# `select_block` isinstance check; it should move to a neutral location with the other relocated LTX-2 classes.
from ...pipelines.ltx2.duration_head import LTX2AutoDuration
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
    LTX2DurationStep,
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
          enable_prompt_enhancement (`bool`, *optional*, defaults to False):
              Trigger for `LTX2AutoPromptEnhancerStep`; the enhancer runs only when this is truthy.
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
class LTX2TextConditioningStep(SequentialPipelineBlocks):
    """
    Text-conditioning stage for LTX-2.X: encodes the prompt(s), expands them per prompt (`num_videos_per_prompt`), then
    runs the text connectors to produce the video/audio-branch connector embeddings the denoiser consumes.

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.

      Outputs:
          prompt_embeds (`Tensor`):
              Packed per-layer Gemma hidden states for the prompt.
          prompt_attention_mask (`Tensor`):
              Binary attention mask for `prompt_embeds`.
          negative_prompt_embeds (`Tensor`):
              Packed per-layer Gemma hidden states for the negative prompt.
          negative_prompt_attention_mask (`Tensor`):
              Binary attention mask for `negative_prompt_embeds`.
          batch_size (`int`):
              TODO: Add description.
          dtype (`dtype`):
              TODO: Add description.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond).
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond).
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond).
          negative_connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (uncond).
          negative_connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (uncond).
          negative_connector_attention_mask (`Tensor`):
              Binary text attention mask (uncond).
    """

    model_name = "ltx2"
    block_classes = [LTX2TextEncoderStep, LTX2TextInputStep, LTX2TextConnectorStep]
    block_names = ["text_encoder", "text_input", "connectors"]

    @property
    def description(self):
        return (
            "Text-conditioning stage for LTX-2.X: encodes the prompt(s), expands them per prompt "
            "(`num_videos_per_prompt`), then runs the text connectors to produce the video/audio-branch "
            "connector embeddings the denoiser consumes."
        )


# auto_docstring
class LTX2AutoDurationStep(ConditionalPipelineBlocks):
    """
    Conditional duration-prediction step, run only when `num_frames` is an `LTX2AutoDuration` request.
       - `LTX2DurationStep` predicts `num_frames` from the connector text conditioning via the `duration_head`.
       - Skipped when `num_frames` is an integer (or not an `LTX2AutoDuration`).

      Components:
          duration_head (`LTX2DurationHead`)

      Inputs:
          num_frames (`LTX2AutoDuration`, *optional*):
              An `LTX2AutoDuration` request carrying the `[min_seconds, max_seconds]` bounds.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning from the connector (positive prompt).
          connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning from the connector (positive prompt).
          batch_size (`int`, *optional*):
              The number of prompts being denoised, used to expand conditioning per prompt.

      Outputs:
          num_frames (`int`):
              The predicted number of frames to generate.
    """

    model_name = "ltx2"
    block_classes = [LTX2DurationStep]
    block_names = ["duration"]
    block_trigger_inputs = ["num_frames"]

    def select_block(self, num_frames=None) -> str | None:
        return "duration" if isinstance(num_frames, LTX2AutoDuration) else None

    @property
    def description(self):
        return (
            "Conditional duration-prediction step, run only when `num_frames` is an `LTX2AutoDuration` request.\n"
            " - `LTX2DurationStep` predicts `num_frames` from the connector text conditioning via the `duration_head`.\n"
            " - Skipped when `num_frames` is an integer (or not an `LTX2AutoDuration`)."
        )


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
class LTX2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (text-to-video) that prepares video/audio latents and runs the joint denoising loop.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

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
              The number of frames in the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          noise_scale (`float`, *optional*, defaults to 0.0):
              Interpolation factor between random noise and any provided latents (0.0 keeps the provided latents).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          audio_latents (`Tensor`, *optional*):
              Optional pre-encoded audio latents; random noise is used when not provided.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          connector_audio_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_audio_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          connector_attention_mask (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_attention_mask (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.

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
          (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

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
              The number of frames in the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          noise_scale (`float`, *optional*, defaults to 0.0):
              Interpolation factor between random noise and any provided latents (0.0 keeps the provided latents).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          image_latents (`Tensor`):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          audio_latents (`Tensor`, *optional*):
              Optional pre-encoded audio latents; random noise is used when not provided.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          connector_audio_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_audio_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          connector_attention_mask (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_attention_mask (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.

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
class LTX2AutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Auto denoise block that selects the workflow based on inputs.
       - `LTX2Image2VideoCoreDenoiseStep` when `image_latents` is provided.
       - `LTX2CoreDenoiseStep` otherwise (text-to-video).

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

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
              The number of frames in the generated video.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          noise_scale (`float`, *optional*, defaults to 0.0):
              Interpolation factor between random noise and any provided latents (0.0 keeps the provided latents).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          image_latents (`Tensor`, *optional*):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          audio_latents (`Tensor`):
              Optional pre-encoded audio latents; random noise is used when not provided.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          connector_audio_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_audio_prompt_embeds (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          connector_attention_mask (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.
          negative_connector_attention_mask (`Tensor`):
              Per-pass text conditioning read by the guiders via `guider_input_fields`.

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
class LTX2DecoderStep(SequentialPipelineBlocks):
    """
    Decode stage: VAE-decodes the video latents and vocodes the audio latents (or returns latents).

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) audio_vae (`AutoencoderKLLTX2Audio`)
          vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              The number of frames in the generated video.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          dtype (`dtype`):
              The dtype of the model inputs, can be generated in input step.
          audio_latents (`Tensor`):
              Denoised audio latents.
          audio_num_frames (`int`):
              Number of audio latent frames, used to unpack the audio latent sequence.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [LTX2VaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["video_decode", "audio_decode"]

    @property
    def description(self):
        return "Decode stage: VAE-decodes the video latents and vocodes the audio latents (or returns latents)."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX-2 text-to-video (joint video + audio).

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`)
          duration_head (`LTX2DurationHead`) scheduler (`FlowMatchEulerDiscreteScheduler`) transformer
          (`LTX2VideoTransformer3DModel`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider
          (`LTX2Guidance`) vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_frames (`LTX2AutoDuration`, *optional*):
              An `LTX2AutoDuration` request carrying the `[min_seconds, max_seconds]` bounds.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
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
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*, defaults to 0.0):
              Interpolation factor between random noise and any provided latents (0.0 keeps the provided latents).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          audio_latents (`Tensor`, *optional*):
              Optional pre-encoded audio latents; random noise is used when not provided.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2CoreDenoiseStep,
        LTX2DecoderStep,
    ]
    block_names = ["text_encoder", "duration", "denoise", "decode"]

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
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`)
          duration_head (`LTX2DurationHead`) vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`)
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_frames (`LTX2AutoDuration`, *optional*):
              An `LTX2AutoDuration` request carrying the `[min_seconds, max_seconds]` bounds.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
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
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*, defaults to 0.0):
              Interpolation factor between random noise and any provided latents (0.0 keeps the provided latents).
          image_latents (`Tensor`):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          audio_latents (`Tensor`, *optional*):
              Optional pre-encoded audio latents; random noise is used when not provided.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2AutoVaeEncoderStep,
        LTX2Image2VideoCoreDenoiseStep,
        LTX2DecoderStep,
    ]
    block_names = ["text_encoder", "duration", "vae_encoder", "denoise", "decode"]

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
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
          (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          transformer (`LTX2VideoTransformer3DModel`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`)
          audio_guider (`LTX2Guidance`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          enable_prompt_enhancement (`bool`, *optional*, defaults to False):
              Trigger for `LTX2AutoPromptEnhancerStep`; the enhancer runs only when this is truthy.
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
          num_frames (`LTX2AutoDuration`, *optional*):
              An `LTX2AutoDuration` request carrying the `[min_seconds, max_seconds]` bounds.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
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
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*, defaults to 0.0):
              Interpolation factor between random noise and any provided latents (0.0 keeps the provided latents).
          image_latents (`Tensor`, *optional*):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          audio_latents (`Tensor`):
              Optional pre-encoded audio latents; random noise is used when not provided.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to False):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2AutoVaeEncoderStep,
        LTX2AutoCoreDenoiseStep,
        LTX2DecoderStep,
    ]
    block_names = ["prompt_enhancer", "text_encoder", "duration", "vae_encoder", "denoise", "decode"]
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
