# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math

import torch

from ...configuration_utils import FrozenDict
from ...guiders import LTX2Guidance
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ..ltx2.before_denoise import LTX2SetTimestepsStep, LTX2TextInputStep
from ..ltx2.decoders import LTX2AudioDecoderStep
from ..ltx2.modular_blocks_ltx2 import LTX2Image2VideoCoreDenoiseStep, LTX2TextConditioningStep
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, OutputParam
from .action import EchoWMCameraConditionStep, EchoWMFlashCameraConditionStep
from .before_denoise import (
    EchoWMImage2VideoPrepareLatentsStep,
    EchoWMPrepareAudioLatentsStep,
    EchoWMPrepareCoordsStep,
    EchoWMPrepareLatentsStep,
)
from .decoders import EchoWMVaeDecoderStep
from .denoise import EchoWMBaseDenoiseStep, EchoWMFlashDenoiseStep
from .encoders import EchoWMFlashTextConditioningStep, EchoWMVaeEncoderStep


class EchoWMImage2VideoCoreDenoiseStep(LTX2Image2VideoCoreDenoiseStep):
    """Echo-WM's packed-noise and model-dtype sampler, using the shared LTX-2 guider interface."""

    model_name = "echo-wm"
    block_classes = [
        LTX2TextInputStep,
        LTX2SetTimestepsStep,
        EchoWMPrepareLatentsStep,
        EchoWMImage2VideoPrepareLatentsStep,
        EchoWMPrepareAudioLatentsStep,
        EchoWMPrepareCoordsStep,
        EchoWMBaseDenoiseStep,
    ]


# auto_docstring
class EchoWMDecoderStep(SequentialPipelineBlocks):
    """
    Components:
          vae (`AutoencoderKLLTX2Video`)
          video_processor (`VideoProcessor`)
          audio_vae (`AutoencoderKLLTX2Audio`)
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
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be
              generated in input step.
          dtype (`dtype`):
              The dtype of the model inputs, can be generated in input step.
          vae_tiling (`bool`, *optional*, defaults to True):
              Enable spatial and temporal VAE decoding tiles to reduce peak memory usage.
          vae_tile_size (`int`, *optional*, defaults to 512):
              Spatial tile long-side size in pixels; the short side follows the video aspect ratio.
          vae_tile_overlap (`int`, *optional*, defaults to 64):
              Spatial tile overlap in pixels.
          vae_temporal_tile_size (`int`, *optional*, defaults to 64):
              Temporal tile size in sample frames, excluding the causal boundary frame.
          vae_temporal_tile_overlap (`int`, *optional*, defaults to 24):
              Temporal tile overlap in sample frames.
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

    model_name = "echo-wm"
    block_classes = [EchoWMVaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["video_decode", "audio_decode"]


# auto_docstring
class EchoWMBlocks(SequentialPipelineBlocks):
    """
    Components:
          text_encoder (`PreTrainedModel`)
          tokenizer (`PreTrainedTokenizerBase`)
          connectors (`LTX2TextConnectors`)
          transformer (`EchoWMTransformer3DModel`)
          vae (`AutoencoderKLLTX2Video`)
          video_processor (`VideoProcessor`)
          scheduler (`FlowMatchEulerDiscreteScheduler`)
          audio_vae (`AutoencoderKLLTX2Audio`)
          guider (`LTX2Guidance`)
          audio_guider (`LTX2Guidance`)
          vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          action (`str`):
              WASD/IJKL action program.
          height (`int`, *optional*, defaults to 704):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1280):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 241):
              Number of output video frames.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Output video frame rate.
          translation_speed (`float`, *optional*, defaults to 0.05):
              Per-frame camera translation speed for W/A/S/D actions.
          rotation_speed_deg (`float`, *optional*, defaults to 0.5):
              Per-frame camera yaw speed in degrees for J/L actions.
          pitch_speed_deg (`float`, *optional*, defaults to 0.2):
              Per-frame camera pitch speed in degrees for I/K actions.
          pitch_limit_deg (`float`, *optional*, defaults to 60.0):
              Maximum absolute camera pitch in degrees.
          fov_deg (`float`, *optional*, defaults to 70.0):
              Horizontal camera field of view in degrees.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          image_crf (`int`, *optional*):
              H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the model was
              trained against. `None` (default) resolves from the text-encoder generation (33 through LTX-2.3, 18 for LTX-2.5).
              Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when re-compression runs.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 30):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Interpolation factor between random noise and any provided latents. `None` (default) resolves to 0.0, which keeps
              the provided latents.
          audio_latents (`Tensor`, *optional*):
              Optional pre-encoded audio latents; random noise is used when not provided.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to True):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          vae_tiling (`bool`, *optional*, defaults to True):
              Enable spatial and temporal VAE decoding tiles to reduce peak memory usage.
          vae_tile_size (`int`, *optional*, defaults to 512):
              Spatial tile long-side size in pixels; the short side follows the video aspect ratio.
          vae_tile_overlap (`int`, *optional*, defaults to 64):
              Spatial tile overlap in pixels.
          vae_temporal_tile_size (`int`, *optional*, defaults to 64):
              Temporal tile size in sample frames, excluding the causal boundary frame.
          vae_temporal_tile_overlap (`int`, *optional*, defaults to 24):
              Temporal tile overlap in sample frames.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "echo-wm"
    block_classes = [
        LTX2TextConditioningStep,
        EchoWMCameraConditionStep,
        EchoWMVaeEncoderStep,
        EchoWMImage2VideoCoreDenoiseStep,
        EchoWMDecoderStep,
    ]
    block_names = ["text", "camera", "image_encoder", "denoise", "decode"]

    @property
    def expected_components(self):
        default_specs = {
            "scheduler": ComponentSpec(
                "scheduler",
                FlowMatchEulerDiscreteScheduler,
                config=FrozenDict(
                    {
                        # The reference calls LTX2Scheduler without a latent, using its fixed 4096-token anchor.
                        "shift": math.exp(2.05),
                        "shift_terminal": 0.1,
                    }
                ),
                default_creation_method="from_config",
            ),
            "guider": ComponentSpec(
                "guider",
                LTX2Guidance,
                config=FrozenDict(
                    {
                        "guidance_scale": 4.0,
                        "stg_scale": 1.0,
                        "modality_scale": 1.0,
                        "guidance_rescale": 0.0,
                        "spatio_temporal_guidance_blocks": [29],
                    }
                ),
                default_creation_method="from_config",
            ),
            "audio_guider": ComponentSpec(
                "audio_guider",
                LTX2Guidance,
                config=FrozenDict(
                    {
                        "guidance_scale": 2.0,
                        "stg_scale": 1.0,
                        "modality_scale": 1.0,
                        "guidance_rescale": 0.0,
                    }
                ),
                default_creation_method="from_config",
            ),
        }
        return [default_specs.get(spec.name, spec) for spec in super().expected_components]

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class EchoWMFlashBlocks(SequentialPipelineBlocks):
    """
    Components:
          text_encoder (`PreTrainedModel`)
          tokenizer (`PreTrainedTokenizerBase`)
          connectors (`LTX2TextConnectors`)
          transformer (`EchoWMTransformer3DModel`)
          vae (`AutoencoderKLLTX2Video`)
          video_processor (`VideoProcessor`)
          audio_vae (`AutoencoderKLLTX2Audio`)
          scheduler (`FlowMatchEulerDiscreteScheduler`)
          vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          action (`str`):
              WASD/IJKL action program.
          height (`int`, *optional*, defaults to 704):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1280):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 241):
              Number of output video frames.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Output video frame rate.
          translation_speed (`float`, *optional*, defaults to 0.05):
              Per-frame camera translation speed for W/A/S/D actions.
          rotation_speed_deg (`float`, *optional*, defaults to 0.4):
              Per-frame camera yaw speed in degrees for J/L actions.
          pitch_speed_deg (`float`, *optional*, defaults to 0.2):
              Per-frame camera pitch speed in degrees for I/K actions.
          pitch_limit_deg (`float`, *optional*, defaults to 40.0):
              Maximum absolute camera pitch in degrees.
          fov_deg (`float`, *optional*, defaults to 70.0):
              Horizontal camera field of view in degrees.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          image_crf (`int`, *optional*):
              H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the model was
              trained against. `None` (default) resolves from the text-encoder generation (33 through LTX-2.3, 18 for LTX-2.5).
              Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when re-compression runs.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          timesteps (`list`, *optional*, defaults to [1000, 750, 500, 250]):
              Distilled denoising timestep IDs for each autoregressive chunk.
          video_cache_size (`int`, *optional*, defaults to 19):
              Maximum number of latent video frames retained in the bounded KV cache.
          video_sink_size (`int`, *optional*, defaults to 7):
              Number of leading latent video frames permanently retained as the cache sink.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          vae_tiling (`bool`, *optional*, defaults to True):
              Enable spatial and temporal VAE decoding tiles to reduce peak memory usage.
          vae_tile_size (`int`, *optional*, defaults to 512):
              Spatial tile long-side size in pixels; the short side follows the video aspect ratio.
          vae_tile_overlap (`int`, *optional*, defaults to 64):
              Spatial tile overlap in pixels.
          vae_temporal_tile_size (`int`, *optional*, defaults to 64):
              Temporal tile size in sample frames, excluding the causal boundary frame.
          vae_temporal_tile_overlap (`int`, *optional*, defaults to 24):
              Temporal tile overlap in sample frames.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "echo-wm-flash"
    block_classes = [
        EchoWMFlashTextConditioningStep,
        EchoWMFlashCameraConditionStep,
        EchoWMVaeEncoderStep,
        EchoWMFlashDenoiseStep,
        EchoWMDecoderStep,
    ]
    block_names = ["text", "camera", "image_encoder", "denoise", "decode"]

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]
