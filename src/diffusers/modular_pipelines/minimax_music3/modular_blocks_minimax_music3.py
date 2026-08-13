# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch

from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import MiniMaxMusic3PrepareChunksStep
from .decoders import MiniMaxMusic3VocoderDecodeStep
from .denoise import MiniMaxMusic3ChunkDenoiseStep
from .encoders import MiniMaxMusic3SemanticGenerationStep, MiniMaxMusic3TextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class MiniMaxMusic3Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline for lyrics- and caption-conditioned music generation using MiniMax Music 3. An autoregressive
    Qwen3 language model generates per-frame semantic codes and hidden states from the lyrics and the music
    description; a flow-matching transformer turns the hidden states into Flow-VAE latents chunk by chunk; and a
    DAC-style vocoder decodes them into a stereo waveform at 44.1 kHz.

      Components:
          tokenizer (`Qwen2Tokenizer`) language_model (`Qwen3ForCausalLM`) rvq_depth_decoder
          (`MiniMaxMusic3RVQDepthDecoder`) condition_encoder (`MiniMaxMusic3ConditionEncoder`) transformer
          (`MiniMaxMusic3Transformer1DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`) vocoder (`MiniMaxMusic3Vocoder`)

      Inputs:
          prompt (`str`):
              The music description (genre, mood, vocals, instrumentation, arrangement).
          lyrics (`str`):
              The lyrics to sing. Structure tags such as `[verse]` or `[chorus]` must each be on their own line; text
              on the same line as a leading tag is dropped by the checkpoint's input contract.
          audio_duration (`float`, *optional*, defaults to 60.0):
              Upper bound on the generated audio length in seconds. The language model may stop earlier. Capped at 9000
              frames (six minutes).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 30):
              Number of flow-matching Euler steps per chunk.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'np' or 'pt'.

      Outputs:
          audios (`Tensor | ndarray`):
              The generated stereo waveform of shape `(batch, channels, samples)` in `[-1, 1]`.
    """

    block_classes = [
        MiniMaxMusic3TextEncoderStep,
        MiniMaxMusic3SemanticGenerationStep,
        MiniMaxMusic3PrepareChunksStep,
        MiniMaxMusic3ChunkDenoiseStep,
        MiniMaxMusic3VocoderDecodeStep,
    ]
    block_names = ["text_encoder", "semantic_generator", "prepare_chunks", "denoise", "decode"]

    @property
    def description(self) -> str:
        return (
            "Modular pipeline for lyrics- and caption-conditioned music generation using MiniMax Music 3. "
            "An autoregressive Qwen3 language model generates per-frame semantic codes and hidden states from the "
            "lyrics and the music description; a flow-matching transformer turns the hidden states into Flow-VAE "
            "latents chunk by chunk; and a DAC-style vocoder decodes them into a stereo waveform at 44.1 kHz."
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                "audios",
                type_hint=torch.Tensor | np.ndarray,
                description="The generated stereo waveform of shape `(batch, channels, samples)` in `[-1, 1]`.",
            ),
        ]
