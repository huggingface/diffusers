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

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import EchoInputsStep, EchoPrepareConditioningStep, EchoPrepareLatentsStep
from .decoders import EchoAudioDecoderStep, EchoVaeDecoderStep
from .denoise import EchoDenoiseLoopStep, EchoUnpackLatentsStep
from .encoders import EchoTextConnectorStep, EchoTextEncoderStep, EchoVaeEncoderStep


EchoTextEncoderBlocks = InsertableDict(
    [
        ("encode", EchoTextEncoderStep()),
        ("connect", EchoTextConnectorStep()),
    ]
)


# auto_docstring
class EchoTextConditioningStep(SequentialPipelineBlocks):
    """
    Positive-only text conditioning for the guidance-free Echo DMD checkpoint.

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              Packed per-layer Gemma hidden states for the prompt.
          prompt_attention_mask (`Tensor`):
              Binary attention mask for `prompt_embeds`.
          connector_prompt_embeds (`Tensor`):
              Video-branch positive text conditioning.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch positive text conditioning.
          connector_attention_mask (`Tensor`):
              Binary attention mask for the positive text conditioning.
    """

    model_name = "echo"
    block_classes = EchoTextEncoderBlocks.values()
    block_names = EchoTextEncoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Positive-only text conditioning for the guidance-free Echo DMD checkpoint."


EchoDenoiseBlocks = InsertableDict(
    [
        ("prepare_conditioning", EchoPrepareConditioningStep()),
        ("input", EchoInputsStep()),
        ("prepare_latents", EchoPrepareLatentsStep()),
        ("denoise_loop", EchoDenoiseLoopStep()),
        ("after_denoise", EchoUnpackLatentsStep()),
    ]
)


# auto_docstring
class EchoDenoiseStep(SequentialPipelineBlocks):
    """
    Prepare Echo conditioning and target latents, then run stochastic DMD denoising.

      Components:
          transformer (`LTX2VideoTransformer3DModel`)

      Inputs:
          first_frame_latents (`Tensor`, *optional*):
              Normalized VAE latents for the optional clean first frame.
          memory_video_latents (`list`, *optional*):
              Normalized VAE latents for each ordered memory image.
          memory_audio_latents (`list`, *optional*):
              Normalized audio VAE latents of shape (B, C, L, M) for each memory slot.
          model_frame_rate (`float`, *optional*, defaults to 24.0):
              Training-time frame rate used for Echo video RoPE coordinates.
          memory_position_offset (`float`, *optional*, defaults to 500.0):
              Temporal center assigned to the first memory slot.
          memory_position_slot_stride (`float`, *optional*, defaults to 50.0):
              Temporal distance between consecutive memory-slot centers.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              Number of videos per prompt.
          connector_prompt_embeds (`Tensor`):
              Per-prompt video-branch text conditioning.
          connector_audio_prompt_embeds (`Tensor`):
              Per-prompt audio-branch text conditioning.
          connector_attention_mask (`Tensor`):
              Per-prompt binary text attention mask.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 241):
              Number of generated pixel frames; must be `1 + k * vae_temporal_compression_ratio`.
          frame_rate (`float`, *optional*, defaults to 25.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Optional initial video noise in VAE form (B, C, F, H, W).
          audio_latents (`Tensor`, *optional*):
              Optional initial audio noise in VAE form (B, C, L, M).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          sigmas (`list | tuple`):
              DMD sigma schedule, including the terminal zero.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Normalized video VAE latents (B, C, F, H, W).
          audio_latents (`Tensor`):
              Normalized audio VAE latents (B, C, L, M).
    """

    model_name = "echo"
    block_classes = EchoDenoiseBlocks.values()
    block_names = EchoDenoiseBlocks.keys()

    @property
    def description(self) -> str:
        return "Prepare Echo conditioning and target latents, then run stochastic DMD denoising."

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents", type_hint=torch.Tensor, description="Normalized video VAE latents (B, C, F, H, W)."
            ),
            OutputParam(
                "audio_latents", type_hint=torch.Tensor, description="Normalized audio VAE latents (B, C, L, M)."
            ),
        ]


EchoDecoderBlocks = InsertableDict(
    [
        ("video_decode", EchoVaeDecoderStep()),
        ("audio_decode", EchoAudioDecoderStep()),
    ]
)


# auto_docstring
class EchoDecoderStep(SequentialPipelineBlocks):
    """
    Decode Echo video and audio outputs with mixed-precision-safe audio vocoding.

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) audio_vae (`AutoencoderKLLTX2Audio`)
          vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Normalized video VAE latents of shape (B, C, F, H, W).
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              Timestep used to decode the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          audio_latents (`Tensor`):
              Normalized audio VAE latents of shape (B, C, L, M).

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "echo"
    block_classes = EchoDecoderBlocks.values()
    block_names = EchoDecoderBlocks.keys()

    @property
    def description(self) -> str:
        return "Decode Echo video and audio outputs with mixed-precision-safe audio vocoding."

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


EchoPipelineBlocks = InsertableDict(
    [
        ("text_encoder", EchoTextConditioningStep()),
        ("vae_encoder", EchoVaeEncoderStep()),
        ("denoise", EchoDenoiseStep()),
        ("decode", EchoDecoderStep()),
    ]
)


# auto_docstring
class EchoBlocks(SequentialPipelineBlocks):
    """
    Echo reference-to-video generation with clean first-frame conditioning, ordered image/audio memory slots, and
    stochastic DMD denoising.

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) vae
          (`AutoencoderKLLTX2Video`) audio_vae (`AutoencoderKLLTX2Audio`) transformer (`LTX2VideoTransformer3DModel`)
          video_processor (`VideoProcessor`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          image (`Image | Tensor`, *optional*):
              Optional single first frame used as a clean reference condition.
          memory_images (`list`, *optional*):
              Ordered reference images, one per Echo memory slot.
          memory_audio_waveforms (`list`, *optional*):
              Ordered memory waveforms as `(channels, samples)` tensors. Inputs longer than 9.62 seconds are cropped to
              their highest-response window. Use `None` for a silent slot.
          memory_audio_sample_rates (`int | list`, *optional*):
              Sampling rate shared by all memory waveforms, or one rate per slot.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          model_frame_rate (`float`, *optional*, defaults to 24.0):
              Training-time frame rate used for Echo video RoPE coordinates.
          memory_position_offset (`float`, *optional*, defaults to 500.0):
              Temporal center assigned to the first memory slot.
          memory_position_slot_stride (`float`, *optional*, defaults to 50.0):
              Temporal distance between consecutive memory-slot centers.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              Number of videos per prompt.
          num_frames (`int`, *optional*, defaults to 241):
              Number of generated pixel frames; must be `1 + k * vae_temporal_compression_ratio`.
          frame_rate (`float`, *optional*, defaults to 25.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Optional initial video noise in VAE form (B, C, F, H, W).
          audio_latents (`Tensor`, *optional*):
              Optional initial audio noise in VAE form (B, C, L, M).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          sigmas (`list | tuple`):
              DMD sigma schedule, including the terminal zero.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              Timestep used to decode the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "echo"
    block_classes = EchoPipelineBlocks.values()
    block_names = EchoPipelineBlocks.keys()

    @property
    def description(self) -> str:
        return (
            "Echo reference-to-video generation with clean first-frame conditioning, ordered image/audio memory "
            "slots, and stochastic DMD denoising."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]
