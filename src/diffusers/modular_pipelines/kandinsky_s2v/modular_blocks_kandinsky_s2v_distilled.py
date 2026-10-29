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
from .before_denoise import KandinskyS2VDistilledPrepareStep
from .decoders import KandinskyS2VDecodeStep
from .denoise import KandinskyS2VDistilledChunkDenoiseStep
from .encoders import KandinskyS2VDistilledTextAudioEncoderStep, KandinskyS2VVaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


KandinskyS2VDistilledCoreDenoiseBlocks = InsertableDict(
    [
        ("prepare", KandinskyS2VDistilledPrepareStep()),
        ("chunk_loop", KandinskyS2VDistilledChunkDenoiseStep()),
    ]
)


# auto_docstring
class KandinskyS2VDistilledCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step of the distilled checkpoints that prepares the conditions and the FramePack memory, then denoises
    the video chunk by chunk with the pi-Flow scheduler.

      Components:
          transformer (`KandinskyS2VTransformer3DModel`) scheduler (`PiflowScheduler`)

      Inputs:
          reference_latents (`Tensor`):
              Scaled VAE latent of the reference frame, from the VAE encoder step.
          audio_embeds (`Tensor`):
              Audio-tower features of the speech track, from the text encoder step.
          num_chunks (`int`):
              Number of chunks to generate.
          num_latent_frames (`int`, *optional*, defaults to 32):
              Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          pooled_prompt_embeds (`Tensor`):
              CLIP pooled prompt embedding.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 16):
              The number of denoising steps.
          scale_factor (`tuple`, *optional*, defaults to (1.0, 2.0, 2.0)):
              Per-axis RoPE scale for the resolution bucket.

      Outputs:
          latent_chunks (`list`):
              Denoised latents of each chunk.
    """

    model_name = "kandinsky-s2v-distilled"
    block_classes = KandinskyS2VDistilledCoreDenoiseBlocks.values()
    block_names = KandinskyS2VDistilledCoreDenoiseBlocks.keys()

    @property
    def outputs(self):
        return [OutputParam("latent_chunks", type_hint=list, description="Denoised latents of each chunk.")]

    @property
    def description(self):
        return (
            "Core denoise step of the distilled checkpoints that prepares the conditions and the FramePack memory, "
            "then denoises the video chunk by chunk with the pi-Flow scheduler."
        )


DISTILLED_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", KandinskyS2VDistilledTextAudioEncoderStep()),
        ("vae_encoder", KandinskyS2VVaeEncoderStep()),
        ("denoise", KandinskyS2VDistilledCoreDenoiseStep()),
        ("decode", KandinskyS2VDecodeStep()),
    ]
)


# auto_docstring
class KandinskyS2VDistilledAutoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline for speech-to-video generation with the pi-Flow distilled Kandinsky S2V 1.0 checkpoints: 16
    segments per chunk and no guidance, which is distilled into the student.

      Components:
          text_encoder (`Qwen2_5OmniThinkerForConditionalGeneration`) tokenizer (`Qwen2_5OmniProcessor`) text_encoder_2
          (`CLIPTextModel`) tokenizer_2 (`CLIPTokenizer`) vae (`AutoencoderKLHunyuanVideo`) transformer
          (`KandinskyS2VTransformer3DModel`) scheduler (`PiflowScheduler`) video_processor (`VideoProcessor`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt to guide the generation. Defaults to the reference prompt, `DEFAULT_PROMPT`.
          audio (`Tensor | ndarray`):
              The speech track that drives the generation: a mono float waveform in [-1, 1] at 16000 Hz, shaped
              (samples,).
          num_latent_frames (`int`, *optional*, defaults to 32):
              Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).
          fps (`int`, *optional*, defaults to 25):
              Frame rate the chunk length is computed at.
          num_chunks (`int`, *optional*):
              Number of chunks to generate. Defaults to however many cover the speech track; the video is trimmed to
              the track either way.
          image (`Image`):
              The reference frame to animate.
          height (`int`, *optional*):
              The height in pixels of the generated video. Defaults to the 512 bucket closest to the reference frame.
          width (`int`, *optional*):
              The width in pixels of the generated video, chosen together with `height`.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 16):
              The number of denoising steps.
          scale_factor (`tuple`, *optional*, defaults to (1.0, 2.0, 2.0)):
              Per-axis RoPE scale for the resolution bucket.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt' or 'latent'. Latents are returned whole, untrimmed.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "kandinsky-s2v-distilled"

    block_classes = DISTILLED_AUTO_BLOCKS.values()
    block_names = DISTILLED_AUTO_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Modular pipeline for speech-to-video generation with the pi-Flow distilled Kandinsky S2V 1.0 "
            "checkpoints: 16 segments per chunk and no guidance, which is distilled into the student."
        )

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
