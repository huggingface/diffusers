# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

from ..modular_pipeline import ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    MiniMaxH3PrepareLatentsStep,
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from .before_encoder import MiniMaxH3Ref2VASetupStep, MiniMaxH3SetupStep
from .decoders import MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep
from .denoise import MiniMaxH3DenoiseStep, MiniMaxH3Ref2VADenoiseStep
from .encoders import (
    MiniMaxH3KeyframeVaeEncoderStep,
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    MiniMaxH3TextEncoderStep,
)


def _generation_outputs() -> list[OutputParam]:
    r"""Video and audio are generated jointly; muxing them into one file is up to the caller."""
    return [
        OutputParam.template("videos", description="The generated video."),
        OutputParam(
            "audio",
            type_hint=torch.Tensor,
            description="The generated soundtrack, of shape `(1, 2, num_samples)`.",
        ),
        OutputParam("sampling_rate", type_hint=int, description="Sample rate of the generated soundtrack in Hz."),
    ]


# auto_docstring
class MiniMaxH3AutoKeyframeVaeEncoderStep(ConditionalPipelineBlocks):
    """
    Keyframe conditioning block.
       - MiniMaxH3KeyframeVaeEncoderStep runs for the `fl2va` task, whichever of the two keyframes is given.
       - when neither `image` nor `last_image` is provided (`t2va`), this block is skipped.

      Components:
          vae (`AutoencoderKLMiniMaxH3`) scheduler (`MiniMaxH3Scheduler`)

      Inputs:
          keyframes (`list`, *optional*):
              The keyframes put onto the target canvas, in packed order.
          latent_height (`int`, *optional*):
              Height of the video latents.
          latent_width (`int`, *optional*):
              Width of the video latents.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it before the target noise of the
              prepare-latents step.

      Outputs:
          condition_latents (`Tensor`):
              The noise-augmented video conditioning rows, in packed order.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3KeyframeVaeEncoderStep]
    block_names = ["keyframes"]
    block_trigger_inputs = ["image", "last_image"]
    default_block_name = None

    def select_block(self, **kwargs) -> str | None:
        if kwargs.get("image") is not None or kwargs.get("last_image") is not None:
            return "keyframes"
        return None

    @property
    def description(self):
        return (
            "Keyframe conditioning block.\n"
            + " - MiniMaxH3KeyframeVaeEncoderStep runs for the `fl2va` task, whichever of the two keyframes is given.\n"
            + " - when neither `image` nor `last_image` is provided (`t2va`), this block is skipped."
        )


# auto_docstring
class MiniMaxH3DecodeStep(SequentialPipelineBlocks):
    """
    Decodes the denoised rows of the packed sequence into the generated video and its soundtrack.

      Components:
          vae (`AutoencoderKLMiniMaxH3`) video_processor (`VideoProcessor`) audio_vae (`AutoencoderKLMiniMaxH3Audio`)

      Inputs:
          latents (`Tensor`):
              The denoised video rows of the packed sequence, conditioning rows first.
          num_condition_video_rows (`int`, *optional*, defaults to 0):
              How many leading video rows are conditioning rows and are dropped here.
          num_latent_frames (`int`):
              Number of video latent frames.
          latent_height (`int`):
              Height of the video latents.
          latent_width (`int`):
              Width of the video latents.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt' or 'latent' for the raw latents.
          audio_latents (`Tensor`):
              The denoised audio rows of the packed sequence, reference rows first.
          num_condition_audio_rows (`int`, *optional*, defaults to 0):
              How many leading audio rows are reference rows and are dropped here.
          num_audio_latents (`int`):
              Number of audio latents per channel.

      Outputs:
          videos (`list`):
              The generated video.
          audio (`Tensor`):
              The generated soundtrack, of shape `(1, 2, num_samples)`.
          sampling_rate (`int`):
              Sample rate of the generated soundtrack in Hz.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep]
    block_names = ["video", "audio"]

    @property
    def description(self):
        return "Decodes the denoised rows of the packed sequence into the generated video and its soundtrack."

    @property
    def outputs(self):
        return _generation_outputs()


# auto_docstring
class MiniMaxH3Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for joint video + audio generation with MiniMax-H3, covering the `t2va` (text only) and
    `fl2va` (first and/or last keyframe) tasks of the FL2VA checkpoint.

      Supported workflows:
        - `t2va`: requires `prompt`
        - `fl2va`: requires `prompt`, `image`
        - `fl2va_last_frame`: requires `prompt`, `last_image`

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer (`Qwen2Tokenizer`) processor (`Qwen3VLProcessor`)
          vae (`AutoencoderKLMiniMaxH3`) scheduler (`MiniMaxH3Scheduler`) audio_scheduler (`MiniMaxH3Scheduler`)
          transformer (`MiniMaxH3Transformer3DModel`) video_processor (`VideoProcessor`) audio_vae
          (`AutoencoderKLMiniMaxH3Audio`)

      Inputs:
          image (`Image`, *optional*):
              Keyframe the video starts from. It is *stretched* onto the target canvas, which by default is derived
              from its own aspect ratio.
          last_image (`Image`, *optional*):
              Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. Combined with `image`
              it is the follower of the two and is cover-cropped onto the canvas.
          height (`int`, *optional*):
              Height of the generated video in pixels, a multiple of 32.
          width (`int`, *optional*):
              Width of the generated video in pixels, a multiple of 32.
          num_frames (`int`, *optional*, defaults to 124):
              Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE can
              decode; the resulting duration must stay between 5 and 15 seconds.
          prompt (`str`):
              The prompt to guide generation, a single string.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it before the target noise of the
              prepare-latents step.
          latents (`Tensor`, *optional*):
              Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used
              instead of the draw.
          audio_latents (`Tensor`, *optional*):
              Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
          condition_latents (`Tensor`, *optional*):
              The video conditioning rows to prepend, or None for a request that has none.
          audio_condition_latents (`Tensor`, *optional*):
              The audio conditioning rows to prepend, or None for a request that has none.
          num_inference_steps (`int`):
              The number of denoising steps.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt' or 'latent' for the raw latents.

      Outputs:
          videos (`list`):
              The generated video.
          audio (`Tensor`):
              The generated soundtrack, of shape `(1, 2, num_samples)`.
          sampling_rate (`int`):
              Sample rate of the generated soundtrack in Hz.
    """

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3SetupStep,
        MiniMaxH3TextEncoderStep,
        MiniMaxH3AutoKeyframeVaeEncoderStep,
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3SetTimestepsStep,
        MiniMaxH3DenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = [
        "setup",
        "text_encoder",
        "vae_encoder",
        "prepare_layout",
        "prepare_latents",
        "set_timesteps",
        "denoise",
        "decode",
    ]
    # One repository holds both checkpoint partitions, so the two blocksets are two workflows over one shared
    # `modular_model_index.json`, and each declares only the components its own half loads. The keys below are the
    # task names a `workflow=` argument to `ModularPipeline.from_pretrained` would take, so that the component subset
    # of a task can be resolved from the index without instantiating the blocks first.
    _workflow_map = {
        "t2va": {"prompt": True},
        "fl2va": {"prompt": True, "image": True},
        "fl2va_last_frame": {"prompt": True, "last_image": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline blocks for joint video + audio generation with MiniMax-H3, covering the `t2va` (text "
            "only) and `fl2va` (first and/or last keyframe) tasks of the FL2VA checkpoint."
        )

    @property
    def outputs(self):
        return _generation_outputs()


# auto_docstring
class MiniMaxH3Ref2VABlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for joint video + audio generation from an ordered list of image, video and audio
    references with MiniMax-H3, the `ref2va` task of the Ref2VA checkpoint.

      Supported workflows:
        - `ref2va`: requires `prompt`, `references`

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer (`Qwen2Tokenizer`) processor (`Qwen3VLProcessor`)
          vae (`AutoencoderKLMiniMaxH3`) audio_vae (`AutoencoderKLMiniMaxH3Audio`) scheduler (`MiniMaxH3Scheduler`)
          audio_scheduler (`MiniMaxH3Scheduler`) transformer_ref (`MiniMaxH3Transformer3DModel`) video_processor
          (`VideoProcessor`)

      Inputs:
          references (`list`):
              The references to condition on, **in the order the model should read them**: the order labels them in the
              prompt presentation and lays them out on the shared rotary clock, so a different order is a different
              request. Every [`MiniMaxH3Reference`] carries exactly one medium, a path or in-memory media — `image` (at
              most 9), `video` at its own `fps` (at most 3, whose `audio` soundtrack is conditioned on as well), or
              `audio` at its own `sample_rate` (at most 3) — for at most 12 references in total, and audio references
              cannot be the only ones. A path is decoded when the reference is built, so these blocks only ever see
              pixels and samples.
          height (`int`, *optional*):
              Height of the generated video in pixels, a multiple of 32.
          width (`int`, *optional*):
              Width of the generated video in pixels, a multiple of 32.
          num_frames (`int`, *optional*):
              Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE can
              decode. May be left out, but only when exactly one reference carries audio, in which case the duration is
              that soundtrack's.
          prompt (`str`):
              The prompt to guide generation, a single string.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it before the target noise of the
              prepare-latents step.
          latents (`Tensor`, *optional*):
              Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used
              instead of the draw.
          audio_latents (`Tensor`, *optional*):
              Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
          num_inference_steps (`int`):
              The number of denoising steps.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt' or 'latent' for the raw latents.

      Outputs:
          videos (`list`):
              The generated video.
          audio (`Tensor`):
              The generated soundtrack, of shape `(1, 2, num_samples)`.
          sampling_rate (`int`):
              Sample rate of the generated soundtrack in Hz.
    """

    # The `ref2va` task family has its own `MODULAR_PIPELINE_MAPPING` key, so that `init_pipeline` resolves
    # `MiniMaxH3Ref2VAModularPipeline` (and its `transformer_ref` partition) rather than the FL2VA one.
    model_name = "minimax-h3-ref2va"
    block_classes = [
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3Ref2VATextEncoderStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3SetTimestepsStep,
        MiniMaxH3Ref2VADenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = [
        "setup",
        "text_encoder",
        "reference_encoder",
        "prepare_layout",
        "prepare_latents",
        "set_timesteps",
        "denoise",
        "decode",
    ]
    # The `ref2va` task name, i.e. the value a `workflow=` argument to `ModularPipeline.from_pretrained` would take
    # to resolve this half — and its `transformer_ref` partition — out of the shared `modular_model_index.json`.
    _workflow_map = {
        "ref2va": {"prompt": True, "references": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline blocks for joint video + audio generation from an ordered list of image, video and "
            "audio references with MiniMax-H3, the `ref2va` task of the Ref2VA checkpoint."
        )

    @property
    def outputs(self):
        return _generation_outputs()
