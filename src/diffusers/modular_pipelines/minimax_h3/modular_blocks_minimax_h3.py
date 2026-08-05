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
    MiniMaxH3FL2VAPrepareLatentsStep,
    MiniMaxH3NoKeyframeAnchorsStep,
    MiniMaxH3PrepareConditionLatentsStep,
    MiniMaxH3PrepareLatentsStep,
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3Ref2VAPrepareLatentsStep,
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from .before_encoder import MiniMaxH3Ref2VASetupStep, MiniMaxH3ResizeStep
from .decoders import MiniMaxH3AfterDenoiseStep, MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep
from .denoise import MiniMaxH3DenoiseStep, MiniMaxH3Ref2VADenoiseStep
from .encoders import (
    MiniMaxH3FL2VATextEncoderStep,
    MiniMaxH3KeyframeVaeEncoderStep,
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    MiniMaxH3TextEncoderStep,
)


# auto_docstring
class MiniMaxH3AutoBeforeEncodeStep(ConditionalPipelineBlocks):
    """
    Media preparation block.
       - `MiniMaxH3Ref2VASetupStep` runs when `references` is provided (`ref2va`): it resolves the plan and normalizes every reference onto MiniMax-H3's own rates and resolutions.
       - `MiniMaxH3ResizeStep` runs when a keyframe is provided (`fl2va`), putting the keyframes onto the target canvas.
       - a text-only request (`t2va`) skips this block, and the layout step falls back to MiniMax-H3's own 16:9 canvas.

      Components:
          image_processor (`VaeImageProcessor`)

      Configs:
          canvas_short_edge (default: 768)
          canvas_max_pixels (default: 1032192)

      Inputs:
          references (`list`, *optional*):
              The references to condition on, **in the order the model should read them**: the order labels them in the prompt
              presentation and lays them out on the shared rotary clock, so a different order is a different request. One
              dataclass per modality, all holding in-memory media — a [`MiniMaxH3ImageReference`] (at most 9), a
              [`MiniMaxH3VideoReference`] at its own `fps` (at most 3, whose `audio` soundtrack is conditioned on as well), or a
              [`MiniMaxH3AudioReference`] at its own `sample_rate` (at most 3) — for at most 12 references in total, and audio
              references cannot be the only ones. These blocks never open a media file: decode with each class's `from_file`
              classmethod, which brings the rates along.
          height (`int`, *optional*):
              Height of the generated video in pixels, a multiple of 32.
          width (`int`, *optional*):
              Width of the generated video in pixels, a multiple of 32.
          num_frames (`int`, *optional*):
              Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE can decode;
              the resulting duration must stay between 5 and 15 seconds. To generate a video as long as a reference soundtrack,
              pass `round(samples / sample_rate * 24)`.
          image (`Image`, *optional*):
              Keyframe the video starts from. It is *stretched* onto the target canvas, which by default is derived from its own
              aspect ratio.
          last_image (`Image`, *optional*):
              Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. Combined with `image` it is the
              follower of the two and is cover-cropped onto the canvas.

      Outputs:
          height (`int`):
              Resolved height of the generated video in pixels.
          width (`int`):
              Resolved width of the generated video in pixels.
          num_frames (`int`):
              Resolved number of frames, of the form 17 * n + 5.
          normalized_references (`list`):
              The references normalized onto MiniMax-H3's own rates and resolutions, in packed order: the same public reference
              types the request passed in, with an image resized to its own 2048 pixel short edge, a video resampled onto 24 fps
              and onto the canvas its own aspect ratio resolves to, and a soundtrack put on the audio VAE's sample rate and
              truncated to the generated duration.
          keyframes (`list`):
              The keyframes put onto the target canvas, in packed order.
          keyframe_anchors (`tuple`):
              Which end of the video every keyframe is anchored to, in packed order. Positional with `keyframes`, so both are
              resolved here.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VASetupStep, MiniMaxH3ResizeStep]
    block_names = ["ref2va", "keyframes"]
    block_trigger_inputs = ["references", "image", "last_image"]
    default_block_name = None

    def select_block(self, **kwargs) -> str | None:
        if kwargs.get("references") is not None:
            return "ref2va"
        if kwargs.get("image") is not None or kwargs.get("last_image") is not None:
            return "keyframes"
        return None

    @property
    def description(self):
        return (
            "Media preparation block.\n"
            + " - `MiniMaxH3Ref2VASetupStep` runs when `references` is provided (`ref2va`): it resolves the plan and "
            "normalizes every reference onto MiniMax-H3's own rates and resolutions.\n"
            + " - `MiniMaxH3ResizeStep` runs when a keyframe is provided (`fl2va`), putting the keyframes onto the "
            "target canvas.\n"
            + " - a text-only request (`t2va`) skips this block, and the layout step falls back to MiniMax-H3's own "
            "16:9 canvas."
        )


# auto_docstring
class MiniMaxH3AutoTextEncoderStep(ConditionalPipelineBlocks):
    """
    Text encoder block. Every branch encodes MiniMax-H3's presentation of the request with the Qwen3-VL conditioner.
       - `MiniMaxH3Ref2VATextEncoderStep` runs when `references` is provided (`ref2va`), labelling every reference in the presentation.
       - `MiniMaxH3FL2VATextEncoderStep` runs when a keyframe is provided (`fl2va`), labelling every keyframe in the presentation.
       - `MiniMaxH3TextEncoderStep` runs otherwise (`t2va`), presenting the prompt alone.

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`)
          tokenizer (`Qwen2Tokenizer`)
          processor (`Qwen3VLProcessor`)

      Inputs:
          prompt (`str`):
              The prompt to guide generation, a single string.
          normalized_references (`list`, *optional*):
              The references normalized by the setup step, in packed order.
          keyframes (`list`, *optional*):
              The keyframes put onto the target canvas, in packed order.

      Outputs:
          prompt_embeds (`Tensor`):
              The hidden state MiniMax-H3 conditions on, of shape `(1, num_text_tokens, 5120)`, read after the 50th decoder layer
              of the Qwen3-VL conditioner.
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`; a vision block is tagged as video.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VATextEncoderStep, MiniMaxH3FL2VATextEncoderStep, MiniMaxH3TextEncoderStep]
    block_names = ["ref2va", "fl2va", "t2va"]
    block_trigger_inputs = ["references", "image", "last_image"]
    default_block_name = "t2va"

    def select_block(self, **kwargs) -> str | None:
        if kwargs.get("references") is not None:
            return "ref2va"
        if kwargs.get("image") is not None or kwargs.get("last_image") is not None:
            return "fl2va"
        return None

    @property
    def description(self):
        return (
            "Text encoder block. Every branch encodes MiniMax-H3's presentation of the request with the Qwen3-VL "
            "conditioner.\n"
            + " - `MiniMaxH3Ref2VATextEncoderStep` runs when `references` is provided (`ref2va`), labelling every "
            "reference in the presentation.\n"
            + " - `MiniMaxH3FL2VATextEncoderStep` runs when a keyframe is provided (`fl2va`), labelling every "
            "keyframe in the presentation.\n"
            + " - `MiniMaxH3TextEncoderStep` runs otherwise (`t2va`), presenting the prompt alone."
        )


# auto_docstring
class MiniMaxH3AutoVaeEncoderStep(ConditionalPipelineBlocks):
    """
    VAE encoder block.
       - `MiniMaxH3Ref2VAReferenceEncoderStep` runs when `references` is provided (`ref2va`).
       - `MiniMaxH3KeyframeVaeEncoderStep` runs when a keyframe is provided (`fl2va`).
       - a text-only request (`t2va`) skips this block.

      Components:
          vae (`AutoencoderKLMiniMaxH3`)
          audio_vae (`AutoencoderKLMiniMaxH3Audio`)

      Inputs:
          normalized_references (`list`, *optional*):
              The references normalized by the setup step, in packed order.
          keyframes (`list`, *optional*):
              The keyframes put onto the target canvas, in packed order.

      Outputs:
          condition_latents (`list`):
              The encoded video conditioning latents of the image and video references, one `(1, latent_channels,
              num_latent_frames, latent_height, latent_width)` tensor each in packed order, or None when the references carry
              none.
          audio_condition_latents (`list`):
              The clean audio conditioning rows of the reference soundtracks, one `(num_audio_latents * 2,
              audio_latent_channels)` tensor per audio-bearing reference in packed order. One entry per reference rather than one
              concatenated block, because the packed layout is built from the row count of each.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VAReferenceEncoderStep, MiniMaxH3KeyframeVaeEncoderStep]
    block_names = ["ref2va", "keyframes"]
    block_trigger_inputs = ["references", "image", "last_image"]
    default_block_name = None

    def select_block(self, **kwargs) -> str | None:
        if kwargs.get("references") is not None:
            return "ref2va"
        if kwargs.get("image") is not None or kwargs.get("last_image") is not None:
            return "keyframes"
        return None

    @property
    def description(self):
        return (
            "VAE encoder block.\n"
            + " - `MiniMaxH3Ref2VAReferenceEncoderStep` runs when `references` is provided (`ref2va`).\n"
            + " - `MiniMaxH3KeyframeVaeEncoderStep` runs when a keyframe is provided (`fl2va`).\n"
            + " - a text-only request (`t2va`) skips this block."
        )


# auto_docstring
class MiniMaxH3CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for `t2va`: builds the packed layout, draws and packs the noise, plans the per-row timesteps, runs the denoising loop against the `transformer` partition and unpacks the denoised rows back into latents.

      Components:
          scheduler (`MiniMaxH3Scheduler`)
          audio_scheduler (`MiniMaxH3Scheduler`)
          transformer (`MiniMaxH3Transformer3DModel`)

      Configs:
          canvas_short_edge (default: 768)
          canvas_max_pixels (default: 1032192)

      Inputs:
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`.
          height (`int`, *optional*):
              Height of the generated video in pixels, a multiple of 32.
          width (`int`, *optional*):
              Width of the generated video in pixels, a multiple of 32.
          num_frames (`int`, *optional*, defaults to 124):
              Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE can decode;
              the resulting duration must stay between 5 and 15 seconds.
          generator (`Generator`, *optional*):
              The generator of the request. The video noise is drawn from it first, then the audio noise.
          latents (`Tensor`, *optional*):
              Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used instead of the
              draw.
          audio_latents (`Tensor`, *optional*):
              Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
          num_inference_steps (`int`):
              The number of denoising steps.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`, `position_ids` and
              the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The generated video latents, of shape `(1, latent_channels, num_latent_frames, latent_height, latent_width)`.
          audio_latents (`Tensor`):
              The generated audio latents, one batch item per stereo channel.
    """

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3NoKeyframeAnchorsStep,
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3SetTimestepsStep,
        MiniMaxH3DenoiseStep,
        MiniMaxH3AfterDenoiseStep,
    ]
    block_names = [
        "no_keyframe_anchors",
        "prepare_layout",
        "prepare_latents",
        "set_timesteps",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return (
            "Core denoising workflow for `t2va`: builds the packed layout, draws and packs the noise, plans the "
            "per-row timesteps, runs the denoising loop against the `transformer` partition and unpacks the "
            "denoised rows back into latents."
        )

    @property
    def outputs(self):
        # What the decode steps consume: the generated latents of either modality.
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated video latents, of shape `(1, latent_channels, num_latent_frames, "
                "latent_height, latent_width)`.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The generated audio latents, one batch item per stereo channel.",
            ),
        ]


# auto_docstring
class MiniMaxH3FL2VACoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for `fl2va`: builds the packed layout with the keyframes anchored, draws and packs the noise around the keyframe conditioning, plans the per-row timesteps, runs the denoising loop against the `transformer` partition and unpacks the denoised rows back into latents.

      Components:
          scheduler (`MiniMaxH3Scheduler`)
          audio_scheduler (`MiniMaxH3Scheduler`)
          transformer (`MiniMaxH3Transformer3DModel`)

      Configs:
          canvas_short_edge (default: 768)
          canvas_max_pixels (default: 1032192)

      Inputs:
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`.
          height (`int`, *optional*):
              Height of the generated video in pixels, a multiple of 32.
          width (`int`, *optional*):
              Width of the generated video in pixels, a multiple of 32.
          num_frames (`int`, *optional*, defaults to 124):
              Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE can decode;
              the resulting duration must stay between 5 and 15 seconds.
          keyframe_anchors (`tuple`, *optional*, defaults to ()):
              Which end of the video every keyframe is anchored to, in packed order.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it first, one draw per condition, before the
              noise of the generated rows.
          condition_latents (`list`):
              The encoded video conditioning latents, one `(1, latent_channels, num_latent_frames, latent_height, latent_width)`
              tensor per condition in packed order.
          latents (`Tensor`, *optional*):
              Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used instead of the
              draw.
          audio_latents (`Tensor`, *optional*):
              Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
          num_inference_steps (`int`):
              The number of denoising steps.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`, `position_ids` and
              the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The generated video latents, of shape `(1, latent_channels, num_latent_frames, latent_height, latent_width)`.
          audio_latents (`Tensor`):
              The generated audio latents, one batch item per stereo channel.
    """

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3PrepareConditionLatentsStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3FL2VAPrepareLatentsStep,
        MiniMaxH3SetTimestepsStep,
        MiniMaxH3DenoiseStep,
        MiniMaxH3AfterDenoiseStep,
    ]
    block_names = [
        "prepare_layout",
        "prepare_condition_latents",
        "prepare_latents",
        "prepare_latents_fl2va",
        "set_timesteps",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return (
            "Core denoising workflow for `fl2va`: builds the packed layout with the keyframes anchored, draws and "
            "packs the noise around the keyframe conditioning, plans the per-row timesteps, runs the denoising loop "
            "against the `transformer` partition and unpacks the denoised rows back into latents."
        )

    @property
    def outputs(self):
        # What the decode steps consume: the generated latents of either modality.
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated video latents, of shape `(1, latent_channels, num_latent_frames, "
                "latent_height, latent_width)`.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The generated audio latents, one batch item per stereo channel.",
            ),
        ]


# auto_docstring
class MiniMaxH3Ref2VACoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for `ref2va`: builds the packed layout with one block per reference, draws and packs the noise, plans the per-row timesteps, runs the denoising loop against the `transformer_ref` partition and unpacks the denoised rows back into latents.

      Components:
          scheduler (`MiniMaxH3Scheduler`)
          audio_scheduler (`MiniMaxH3Scheduler`)
          transformer_ref (`MiniMaxH3Transformer3DModel`)

      Inputs:
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`.
          normalized_references (`list`):
              The references normalized by the setup step, in packed order.
          condition_latents (`list`):
              The encoded video conditioning latents, one per image and video reference in packed order. Their shape is where
              every reference block's geometry comes from.
          audio_condition_latents (`list`):
              The encoded audio conditioning rows, one per audio-bearing reference in packed order.
          height (`int`):
              Height of the generated video in pixels.
          width (`int`):
              Width of the generated video in pixels.
          num_frames (`int`):
              Resolved number of frames, of the form 17 * n + 5.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it first, one draw per condition, before the
              noise of the generated rows.
          latents (`Tensor`, *optional*):
              Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used instead of the
              draw.
          audio_latents (`Tensor`, *optional*):
              Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
          num_inference_steps (`int`):
              The number of denoising steps.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`, `position_ids` and
              the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The generated video latents, of shape `(1, latent_channels, num_latent_frames, latent_height, latent_width)`.
          audio_latents (`Tensor`):
              The generated audio latents, one batch item per stereo channel.
    """

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3PrepareConditionLatentsStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3Ref2VAPrepareLatentsStep,
        MiniMaxH3SetTimestepsStep,
        MiniMaxH3Ref2VADenoiseStep,
        MiniMaxH3AfterDenoiseStep,
    ]
    block_names = [
        "prepare_layout",
        "prepare_condition_latents",
        "prepare_latents",
        "prepare_latents_ref2va",
        "set_timesteps",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return (
            "Core denoising workflow for `ref2va`: builds the packed layout with one block per reference, draws and "
            "packs the noise, plans the per-row timesteps, runs the denoising loop against the `transformer_ref` "
            "partition and unpacks the denoised rows back into latents."
        )

    @property
    def outputs(self):
        # What the decode steps consume: the generated latents of either modality.
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated video latents, of shape `(1, latent_channels, num_latent_frames, "
                "latent_height, latent_width)`.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The generated audio latents, one batch item per stereo channel.",
            ),
        ]


# auto_docstring
class MiniMaxH3AutoDenoiseStep(ConditionalPipelineBlocks):
    """
    Denoise block.
       - `MiniMaxH3Ref2VACoreDenoiseStep` runs when `references` is provided (`ref2va`), against the `transformer_ref` checkpoint partition.
       - `MiniMaxH3FL2VACoreDenoiseStep` runs when a keyframe is provided (`fl2va`), against the `transformer` partition.
       - `MiniMaxH3CoreDenoiseStep` runs otherwise (`t2va`), against the `transformer` partition.

      Components:
          scheduler (`MiniMaxH3Scheduler`)
          audio_scheduler (`MiniMaxH3Scheduler`)
          transformer_ref (`MiniMaxH3Transformer3DModel`)
          transformer (`MiniMaxH3Transformer3DModel`)

      Configs:
          canvas_short_edge (default: 768)
          canvas_max_pixels (default: 1032192)

      Inputs:
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`.
          normalized_references (`list`, *optional*):
              The references normalized by the setup step, in packed order.
          condition_latents (`list`, *optional*):
              The encoded video conditioning latents, one per image and video reference in packed order. Their shape is where
              every reference block's geometry comes from.
          audio_condition_latents (`list`, *optional*):
              The encoded audio conditioning rows, one per audio-bearing reference in packed order.
          height (`int`, *optional*):
              Height of the generated video in pixels.
          width (`int`, *optional*):
              Width of the generated video in pixels.
          num_frames (`int`, *optional*, defaults to 124):
              Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE can decode;
              the resulting duration must stay between 5 and 15 seconds.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it first, one draw per condition, before the
              noise of the generated rows.
          latents (`Tensor`):
              Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used instead of the
              draw.
          audio_latents (`Tensor`):
              Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
          num_inference_steps (`int`):
              The number of denoising steps.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`, `position_ids` and
              the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          keyframe_anchors (`tuple`, *optional*, defaults to ()):
              Which end of the video every keyframe is anchored to, in packed order.

      Outputs:
          latents (`Tensor`):
              The generated video latents, of shape `(1, latent_channels, num_latent_frames, latent_height, latent_width)`.
          audio_latents (`Tensor`):
              The generated audio latents, one batch item per stereo channel.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VACoreDenoiseStep, MiniMaxH3FL2VACoreDenoiseStep, MiniMaxH3CoreDenoiseStep]
    block_names = ["ref2va", "fl2va", "t2va"]
    block_trigger_inputs = ["references", "image", "last_image"]
    default_block_name = "t2va"

    def select_block(self, **kwargs) -> str | None:
        if kwargs.get("references") is not None:
            return "ref2va"
        if kwargs.get("image") is not None or kwargs.get("last_image") is not None:
            return "fl2va"
        return None

    @property
    def description(self):
        return (
            "Denoise block.\n"
            + " - `MiniMaxH3Ref2VACoreDenoiseStep` runs when `references` is provided (`ref2va`), against the "
            "`transformer_ref` checkpoint partition.\n"
            + " - `MiniMaxH3FL2VACoreDenoiseStep` runs when a keyframe is provided (`fl2va`), against the "
            "`transformer` partition.\n"
            + " - `MiniMaxH3CoreDenoiseStep` runs otherwise (`t2va`), against the `transformer` partition."
        )


# auto_docstring
class MiniMaxH3DecodeStep(SequentialPipelineBlocks):
    """
    Decodes the denoised rows of the packed sequence into the generated video and its soundtrack.

      Components:
          vae (`AutoencoderKLMiniMaxH3`)
          video_processor (`VideoProcessor`)
          audio_vae (`AutoencoderKLMiniMaxH3Audio`)

      Inputs:
          latents (`Tensor`):
              The generated video latents.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np' or 'pt'.
          audio_latents (`Tensor`):
              The generated audio latents, one batch item per stereo channel.

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
class MiniMaxH3Blocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline blocks for joint video + audio generation with MiniMax-H3: the `t2va` (text only), `fl2va` (first and/or last keyframe) and `ref2va` (omni-reference) workflows, selected on the `references` and keyframe inputs. Without a `workflow=`, loading the components pulls **both** 61.7GB transformer partitions — pass one to load only what the task needs.

      Supported workflows:
        - `t2va`: requires `prompt`
        - `fl2va`: requires `prompt`, `image` or `prompt`, `last_image`
        - `ref2va`: requires `prompt`, `references`

      Components:
          image_processor (`VaeImageProcessor`)
          text_encoder (`Qwen3VLForConditionalGeneration`)
          tokenizer (`Qwen2Tokenizer`)
          processor (`Qwen3VLProcessor`)
          vae (`AutoencoderKLMiniMaxH3`)
          audio_vae (`AutoencoderKLMiniMaxH3Audio`)
          scheduler (`MiniMaxH3Scheduler`)
          audio_scheduler (`MiniMaxH3Scheduler`)
          transformer_ref (`MiniMaxH3Transformer3DModel`)
          transformer (`MiniMaxH3Transformer3DModel`)
          video_processor (`VideoProcessor`)

      Configs:
          canvas_short_edge (default: 768)
          canvas_max_pixels (default: 1032192)

      Inputs:
          references (`list`, *optional*):
              The references to condition on, **in the order the model should read them**: the order labels them in the prompt
              presentation and lays them out on the shared rotary clock, so a different order is a different request. One
              dataclass per modality, all holding in-memory media — a [`MiniMaxH3ImageReference`] (at most 9), a
              [`MiniMaxH3VideoReference`] at its own `fps` (at most 3, whose `audio` soundtrack is conditioned on as well), or a
              [`MiniMaxH3AudioReference`] at its own `sample_rate` (at most 3) — for at most 12 references in total, and audio
              references cannot be the only ones. These blocks never open a media file: decode with each class's `from_file`
              classmethod, which brings the rates along.
          height (`int`, *optional*):
              Height of the generated video in pixels, a multiple of 32.
          width (`int`, *optional*):
              Width of the generated video in pixels, a multiple of 32.
          num_frames (`int`, *optional*):
              Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE can decode;
              the resulting duration must stay between 5 and 15 seconds. To generate a video as long as a reference soundtrack,
              pass `round(samples / sample_rate * 24)`.
          image (`Image`, *optional*):
              Keyframe the video starts from. It is *stretched* onto the target canvas, which by default is derived from its own
              aspect ratio.
          last_image (`Image`, *optional*):
              Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. Combined with `image` it is the
              follower of the two and is cover-cropped onto the canvas.
          prompt (`str`):
              The prompt to guide generation, a single string.
          normalized_references (`list`, *optional*):
              The references normalized by the setup step, in packed order.
          keyframes (`list`, *optional*):
              The keyframes put onto the target canvas, in packed order.
          condition_latents (`list`, *optional*):
              The encoded video conditioning latents, one per image and video reference in packed order. Their shape is where
              every reference block's geometry comes from.
          audio_condition_latents (`list`, *optional*):
              The encoded audio conditioning rows, one per audio-bearing reference in packed order.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it first, one draw per condition, before the
              noise of the generated rows.
          latents (`Tensor`):
              Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used instead of the
              draw.
          audio_latents (`Tensor`):
              Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
          num_inference_steps (`int`):
              The number of denoising steps.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`, `position_ids` and
              the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          keyframe_anchors (`tuple`, *optional*, defaults to ()):
              Which end of the video every keyframe is anchored to, in packed order.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np' or 'pt'.

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
        MiniMaxH3AutoBeforeEncodeStep,
        MiniMaxH3AutoTextEncoderStep,
        MiniMaxH3AutoVaeEncoderStep,
        MiniMaxH3AutoDenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = [
        "before_encode",
        "text_encoder",
        "vae_encoder",
        "denoise",
        "decode",
    ]
    # One repository holds both checkpoint partitions (`transformer/` for `t2va`/`fl2va`, `transformer_ref/` for
    # `ref2va`), so the tasks are workflows over one shared `modular_model_index.json`, each declaring only the
    # components its own half loads. The keys below are the task names a `workflow=` argument to
    # `ModularPipeline.from_pretrained` takes, so that the component subset of a task can be resolved from the index
    # without instantiating the blocks first.
    _workflow_map = {
        "t2va": {"prompt": True},
        "fl2va": ({"prompt": True, "image": True}, {"prompt": True, "last_image": True}),
        "ref2va": {"prompt": True, "references": True},
    }

    @property
    def description(self):
        return (
            "Auto Modular pipeline blocks for joint video + audio generation with MiniMax-H3: the `t2va` (text "
            "only), `fl2va` (first and/or last keyframe) and `ref2va` (omni-reference) workflows, selected on the "
            "`references` and keyframe inputs. Without a `workflow=`, loading the components pulls **both** 61.7GB "
            "transformer partitions — pass one to load only what the task needs."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos", description="The generated video."),
            OutputParam(
                "audio",
                type_hint=torch.Tensor,
                description="The generated soundtrack, of shape `(1, 2, num_samples)`.",
            ),
            OutputParam("sampling_rate", type_hint=int, description="Sample rate of the generated soundtrack in Hz."),
        ]
