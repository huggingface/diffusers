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

from ...models import LTX2VideoTransformer3DModel
from ..modular_pipeline import LoopSequentialPipelineBlocks, PipelineState, SequentialPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, InsertableDict, OutputParam
from .before_denoise import EchoPrepareConditioningStep, EchoPrepareLatentsStep
from .decoders import EchoAudioDecoderStep, EchoVaeDecoderStep
from .denoise import (
    DEFAULT_ECHO_SIGMAS,
    EchoLoopAfterDenoiser,
    EchoLoopBeforeDenoiser,
    EchoLoopDenoiser,
)
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


EchoDenoiseLoopBlocks = InsertableDict(
    [
        ("before_denoiser", EchoLoopBeforeDenoiser()),
        ("denoiser", EchoLoopDenoiser()),
        ("after_denoiser", EchoLoopAfterDenoiser()),
    ]
)


# auto_docstring
class EchoDenoiseLoopStep(LoopSequentialPipelineBlocks):
    """
    Iteratively predicts clean video/audio latents and re-noises them with fresh Gaussian noise according to Echo's DMD
    sigma schedule.

      Components:
          transformer (`LTX2VideoTransformer3DModel`)

      Inputs:
          sigmas (`list | tuple`):
              DMD sigma schedule, including the terminal zero.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          audio_latents (`Tensor`):
              Packed noisy target audio tokens.
          first_frame_token_count (`int`):
              Number of clean first-frame tokens.
          memory_video_tokens (`Tensor`, *optional*):
              Packed clean image-memory tokens.
          memory_video_coords (`Tensor`, *optional*):
              RoPE coordinates for image-memory tokens.
          memory_audio_tokens (`Tensor`, *optional*):
              Packed clean audio-memory tokens.
          memory_audio_coords (`Tensor`, *optional*):
              RoPE coordinates for audio-memory tokens.
          video_coords (`Tensor`):
              RoPE coordinates for target video tokens.
          audio_coords (`Tensor`):
              RoPE coordinates for target audio tokens.
          connector_prompt_embeds (`Tensor`):
              Positive video-branch text conditioning.
          connector_audio_prompt_embeds (`Tensor`):
              Positive audio-branch text conditioning.
          connector_attention_mask (`Tensor`):
              Binary attention mask for text conditioning.
          latent_num_frames (`int`):
              Number of target video latent frames.
          latent_height (`int`):
              Target video latent height.
          latent_width (`int`):
              Target video latent width.
          audio_num_frames (`int`):
              Number of target audio latent frames.
          memory_video_token_count (`int`):
              Number of prepended image-memory tokens.
          memory_audio_token_count (`int`):
              Number of prepended audio-memory tokens.
          frame_rate (`float`, *optional*, defaults to 25.0):
              Frame rate of the generated video.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          audio_latents (`Tensor`):
              Packed target audio tokens.
          sigmas (`list | tuple`):
              DMD sigma schedule including the terminal zero.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          first_frame_tokens (`Tensor`, *optional*):
              Packed clean first-frame tokens.
    """

    model_name = "echo"
    block_classes = EchoDenoiseLoopBlocks.values()
    block_names = EchoDenoiseLoopBlocks.keys()

    @property
    def description(self) -> str:
        return (
            "Iteratively predicts clean video/audio latents and re-noises them with fresh Gaussian noise according "
            "to Echo's DMD sigma schedule."
        )

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", LTX2VideoTransformer3DModel)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "sigmas",
                type_hint=list | tuple,
                default=DEFAULT_ECHO_SIGMAS,
                description="DMD sigma schedule, including the terminal zero.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        sigmas = [float(value) for value in block_state.sigmas]
        if len(sigmas) < 2 or sigmas[-1] != 0.0:
            raise ValueError("Echo `sigmas` must contain at least two values and end at 0.")
        if any(left < right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError("Echo `sigmas` must be monotonically non-increasing.")
        block_state.sigmas = sigmas

        with self.progress_bar(total=len(sigmas) - 1) as progress_bar:
            for i, sigma in enumerate(sigmas[:-1]):
                components, block_state = self.loop_step(components, block_state, i=i, sigma=sigma)
                progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


EchoDenoiseBlocks = InsertableDict(
    [
        ("prepare_conditioning", EchoPrepareConditioningStep()),
        ("prepare_latents", EchoPrepareLatentsStep()),
        ("denoise_loop", EchoDenoiseLoopStep()),
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
              Normalized packed audio VAE latents for each memory slot.
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
          num_frames (`int`, *optional*, defaults to 241):
              Number of generated pixel frames; must be `1 + k * vae_temporal_compression_ratio`.
          frame_rate (`float`, *optional*, defaults to 25.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          audio_latents (`Tensor`, *optional*):
              Optional packed initial audio noise latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          connector_prompt_embeds (`Tensor`):
              Positive video-branch text conditioning.
          sigmas (`list | tuple`):
              DMD sigma schedule, including the terminal zero.
          connector_audio_prompt_embeds (`Tensor`):
              Positive audio-branch text conditioning.
          connector_attention_mask (`Tensor`):
              Binary attention mask for text conditioning.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised packed audio latents.
          audio_num_frames (`int`):
              Number of audio latent frames required for decoding.
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
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised packed audio latents."),
            OutputParam(
                "audio_num_frames",
                type_hint=int,
                description="Number of audio latent frames required for decoding.",
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
              Pre-generated noisy latents for image generation.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              Number of generated video frames.
          decode_timestep (`None`, *optional*, defaults to 0.0):
              Timestep used to decode the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          audio_latents (`Tensor`):
              Denoised audio latents.
          audio_num_frames (`int`):
              Number of audio latent frames used to unpack the audio latent sequence.

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
          num_frames (`int`, *optional*, defaults to 241):
              Number of generated pixel frames; must be `1 + k * vae_temporal_compression_ratio`.
          frame_rate (`float`, *optional*, defaults to 25.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          audio_latents (`Tensor`, *optional*):
              Optional packed initial audio noise latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
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
