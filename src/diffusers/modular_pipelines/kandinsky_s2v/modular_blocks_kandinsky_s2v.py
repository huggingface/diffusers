# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import KandinskyS2VPrepareStep
from .decoders import KandinskyS2VDecodeStep
from .denoise import KandinskyS2VChunkDenoiseStep
from .encoders import KandinskyS2VInputStep, KandinskyS2VTextAudioEncoderStep, KandinskyS2VVaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


AUTO_BLOCKS = InsertableDict(
    [
        ("input", KandinskyS2VInputStep()),
        ("text_encoder", KandinskyS2VTextAudioEncoderStep()),
        ("vae_encoder", KandinskyS2VVaeEncoderStep()),
        ("prepare", KandinskyS2VPrepareStep()),
        ("denoise", KandinskyS2VChunkDenoiseStep()),
        ("decode", KandinskyS2VDecodeStep()),
    ]
)


# auto_docstring
class KandinskyS2VAutoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline for speech-to-video generation with Kandinsky S2V 1.0: an avatar video from a reference frame and
    a speech track, generated chunk by chunk.

      Components:
          tokenizer (`Qwen2_5OmniProcessor`) transformer (`KandinskyS2VTransformer3DModel`) text_encoder
          (`Qwen2_5OmniThinkerForConditionalGeneration`) text_encoder_2 (`CLIPTextModel`) tokenizer_2 (`CLIPTokenizer`)
          guider (`ClassifierFreeGuidance`) vae (`AutoencoderKLHunyuanVideo`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image | str | Path`):
              The reference frame to animate: a PIL image or a path to one.
          audio (`str | Path`):
              The speech track that drives the generation: a path to an audio file.
          height (`int`, *optional*):
              The height in pixels of the generated video. Defaults to the 512 bucket closest to the reference frame.
          width (`int`, *optional*):
              The width in pixels of the generated video, chosen together with `height`.
          num_latent_frames (`int`, *optional*, defaults to 32):
              Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).
          fps (`int`, *optional*, defaults to 25):
              Frame rate the chunk length is computed at.
          num_chunks (`int`, *optional*):
              Number of chunks to generate. Defaults to however many cover the speech track; the video is trimmed to
              the track either way.
          prompt (`str`, *optional*):
              The prompt to guide the generation. Defaults to the reference prompt, `DEFAULT_PROMPT`.
          negative_prompt (`str`, *optional*):
              The prompt for the unconditional pass of the guidance. Defaults to `DEFAULT_NEGATIVE_PROMPT`.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          scale_factor (`tuple`, *optional*, defaults to (1.0, 2.0, 2.0)):
              Per-axis RoPE scale for the resolution bucket.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt' or 'latent'. Latents are returned whole, untrimmed.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "kandinsky-s2v"

    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Modular pipeline for speech-to-video generation with Kandinsky S2V 1.0: an avatar video from a reference "
            "frame and a speech track, generated chunk by chunk."
        )

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
