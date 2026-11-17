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
from .before_denoise import KandinskyS2VPrepareNegativeStep, KandinskyS2VPrepareStep
from .decoders import KandinskyS2VDecodeStep
from .denoise import KandinskyS2VChunkDenoiseStep
from .encoders import KandinskyS2VTextAudioEncoderStep, KandinskyS2VVaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


KandinskyS2VCoreDenoiseBlocks = InsertableDict(
    [
        ("prepare", KandinskyS2VPrepareStep()),
        ("prepare_negative", KandinskyS2VPrepareNegativeStep()),
        ("chunk_loop", KandinskyS2VChunkDenoiseStep()),
    ]
)


# auto_docstring
class KandinskyS2VCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that prepares the conditions and the FramePack memory, then denoises the video chunk by chunk.

      Components:
          transformer (`KandinskyS2VTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

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
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_pooled_prompt_embeds (`Tensor`, *optional*):
              CLIP pooled negative prompt embedding.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          scale_factor (`tuple`, *optional*, defaults to (1.0, 2.0, 2.0)):
              Per-axis RoPE scale for the resolution bucket.

      Outputs:
          latent_chunks (`list`):
              Denoised latents of each chunk, each shaped (num_latent_frames, latent_height, latent_width, channels).
    """

    model_name = "kandinsky-s2v"
    block_classes = KandinskyS2VCoreDenoiseBlocks.values()
    block_names = KandinskyS2VCoreDenoiseBlocks.keys()

    @property
    def outputs(self):
        return [
            OutputParam(
                "latent_chunks",
                type_hint=list,
                description="Denoised latents of each chunk, each shaped "
                "(num_latent_frames, latent_height, latent_width, channels).",
            )
        ]

    @property
    def description(self):
        return (
            "Core denoise step that prepares the conditions and the FramePack memory, then denoises the video "
            "chunk by chunk."
        )


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", KandinskyS2VTextAudioEncoderStep()),
        ("vae_encoder", KandinskyS2VVaeEncoderStep()),
        ("denoise", KandinskyS2VCoreDenoiseStep()),
        ("decode", KandinskyS2VDecodeStep()),
    ]
)


# auto_docstring
class KandinskyS2VAutoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline for speech-to-video generation with Kandinsky S2V 1.0: an avatar video from a reference frame and
    a speech track, generated chunk by chunk.

      Components:
          text_encoder (`Qwen2_5OmniThinkerForConditionalGeneration`) tokenizer (`Qwen2_5OmniProcessor`) text_encoder_2
          (`CLIPTextModel`) tokenizer_2 (`CLIPTokenizer`) vae (`AutoencoderKLHunyuanVideo`) video_processor
          (`VideoProcessor`) transformer (`KandinskyS2VTransformer3DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`, *optional*, defaults to The video presents a stable medium close-up of a subject facing the
          camera directly. The subject exhibits natural, lifelike body dynamics: the head and shoulders move fluidly
          with the rhythm of speech. The lighting is even, maintaining a balanced natural exposure to clearly define
          facial features without washout. Throughout the sequence, the subject engages in continuous speech marked by
          precise, high-fidelity articulation. The lips move in a rhythmic, natural pattern: alternating between brief
          firm closures and relaxed openings. During these openings, the tips of the upper teeth are momentarily and
          realistically visible, exhibiting natural enamel texture. The upper lip lifts subtly over the teeth during
          enunciation, while the jaw performs minimal vertical oscillations synchronized with the speech cadence.
          Facial micro-expressions are active yet restrained; eyebrows rise and lower slightly with minute asymmetry.
          The eyes maintain a steady, engaged gaze forward, punctuated by smooth, deliberate blinks. ):
              The prompt to guide the generation.
          negative_prompt (`str`, *optional*):
              The prompt for the unconditional pass of the guidance. Defaults to `DEFAULT_NEGATIVE_PROMPT` when the
              guidance needs one.
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
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          scale_factor (`tuple`, *optional*, defaults to (1.0, 2.0, 2.0)):
              Per-axis RoPE scale for the resolution bucket.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

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
