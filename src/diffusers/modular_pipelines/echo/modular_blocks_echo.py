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
from ..modular_pipeline_utils import OutputParam
from .echo_before_denoise import EchoPrepareLatentsStep
from .echo_decoders import EchoDecoderStep
from .echo_denoise import EchoDenoiseStep
from .echo_encoders import EchoConditionEncoderStep, EchoTextConditioningStep


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
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "echo"
    block_classes = [
        EchoTextConditioningStep,
        EchoConditionEncoderStep,
        EchoPrepareLatentsStep,
        EchoDenoiseStep,
        EchoDecoderStep,
    ]
    block_names = ["text_encoder", "condition_encoder", "prepare_latents", "denoise", "decode"]

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
