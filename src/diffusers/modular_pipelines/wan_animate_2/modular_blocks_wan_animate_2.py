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

from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import WanAnimate2PrepareSegmentsStep
from .decoders import WanAnimate2DecodeStep
from .denoise import WanAnimate2DenoiseStep
from .encoders import (
    WanAnimate2ImageClipEncoderStep,
    WanAnimate2ImageVaeEncoderStep,
    WanAnimate2ProcessImagesInputStep,
    WanAnimate2ProcessVideosInputStep,
    WanAnimate2TextEncoderStep,
    WanAnimate2VideoClipEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ====================
# 1. Encoder groups
# ====================


WanAnimate2ImageEncoderBlocks = InsertableDict(
    [
        ("preprocess", WanAnimate2ProcessImagesInputStep()),
        ("encode", WanAnimate2ImageClipEncoderStep()),
    ]
)


# auto_docstring
class WanAnimate2ImageEncodeStep(SequentialPipelineBlocks):
    """
    Image encoder step that letterboxes the reference character image to the resolved resolution and CLIP-encodes it into `encoder_hidden_states_image`.

      Components:
          image_processor (`WanAnimate2VideoProcessor`)
          image_encoder (`CLIPVisionModel`)

      Inputs:
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 800):
              Together with `width`, the target *area* of the generated video; the aspect ratio comes from `image`. Overwritten
              with the resolved frame height.
          width (`int`, *optional*, defaults to 640):
              See `height`. Overwritten with the resolved frame width.

      Outputs:
          image_pixels (`Tensor`):
              The letterboxed reference image as a `[1, 3, H, W]` tensor in `[-1, 1]`
          crop_region (`tuple`):
              `(top, left, height, width)` of the reference image content inside the letterboxed frame
          encoder_hidden_states_image (`Tensor`):
              CLIP vision features of the reference image, conditioning every denoising forward
    """

    model_name = "wan-animate-2"
    block_classes = WanAnimate2ImageEncoderBlocks.values()
    block_names = WanAnimate2ImageEncoderBlocks.keys()

    @property
    def description(self):
        return (
            "Image encoder step that letterboxes the reference character image to the resolved resolution and "
            "CLIP-encodes it into `encoder_hidden_states_image`."
        )


WanAnimate2VideoEncoderBlocks = InsertableDict(
    [
        ("preprocess", WanAnimate2ProcessVideosInputStep()),
        ("encode", WanAnimate2VideoClipEncoderStep()),
    ]
)


# auto_docstring
class WanAnimate2VideoEncodeStep(SequentialPipelineBlocks):
    """
    Video encoder step that preprocesses the driving video (fps resample, letterbox, zigzag padding to a whole number of segments) and CLIP-encodes its first frame into `condition_clip_context`.

      Components:
          video_processor (`WanAnimate2VideoProcessor`)
          image_encoder (`CLIPVisionModel`)

      Inputs:
          driving_video (`list`):
              The driving video that provides the motion, in any format accepted by `VideoProcessor.preprocess_video`.
          driving_video_fps (`float`, *optional*):
              The frame rate `driving_video` was captured at — `load_video(..., return_fps=True)` reports it. When set, the
              driving frames are resampled from it to `fps`; when `None` they are used as-is.
          fps (`int`, *optional*, defaults to 24):
              The frame rate the model generates at
          segment_frame_length (`int`, *optional*, defaults to 81):
              The number of frames in each inference segment
          prev_segment_conditioning_frames (`int`, *optional*, defaults to 1):
              The number of conditioning frames carried over from the previous segment
          height (`int`, *optional*, defaults to 800):
              The height the driving frames are letterboxed to; must match the reference image's resolved height. In the
              assembled pipeline the image preprocess step supplies the resolved value.
          width (`int`, *optional*, defaults to 640):
              See `height`.

      Outputs:
          driving_video_pixels (`Tensor`):
              The resampled, letterboxed, and zigzag-padded driving video, `[1, 3, T, height, width]` in `[-1, 1]`
          real_frame_len (`int`):
              Number of driving frames before zigzag padding; the output is trimmed to it
          num_segments (`int`):
              Number of inference segments
          effective_segment (`int`):
              Frames each segment advances by (`segment_frame_length - prev_segment_conditioning_frames`)
          condition_clip_context (`Tensor`):
              CLIP vision features of the driving video's first frame
    """

    model_name = "wan-animate-2"
    block_classes = WanAnimate2VideoEncoderBlocks.values()
    block_names = WanAnimate2VideoEncoderBlocks.keys()

    @property
    def description(self):
        return (
            "Video encoder step that preprocesses the driving video (fps resample, letterbox, zigzag padding to a "
            "whole number of segments) and CLIP-encodes its first frame into `condition_clip_context`."
        )


# ====================
# 2. Core denoise
# ====================


WanAnimate2CoreDenoiseBlocks = InsertableDict(
    [
        ("prepare_segments", WanAnimate2PrepareSegmentsStep()),
        ("denoise", WanAnimate2DenoiseStep()),
    ]
)


# auto_docstring
class WanAnimate2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that computes the segment-invariant geometry and runs the segment-by-segment denoising loop, decoding each segment inside the loop because the next segment conditions on its decoded pixels.

      Components:
          vae (`AutoencoderKLWan`)
          transformer (`WanAnimate2Transformer3DModel`)
          scheduler (`SchedulerMixin`)
          guider (`ClassifierFreeGuidance`)

      Inputs:
          segment_frame_length (`int`, *optional*, defaults to 81):
              TODO: Add description.
          reference_image_latents (`Tensor`):
              The reference conditioning tensor `[20, 1, latent_height, latent_width]`; provides the latent grid
          driving_video_pixels (`Tensor`):
              The preprocessed driving video `[1, 3, T, H, W]`, from the video preprocess step
          num_segments (`int`):
              TODO: Add description.
          effective_segment (`int`):
              TODO: Add description.
          prev_segment_conditioning_frames (`int`, *optional*, defaults to 1):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`int`, *optional*, defaults to 40):
              TODO: Add description.
          condition_clip_context (`Tensor`):
              TODO: Add description.
          prompt_ref_embeds (`Tensor`):
              TODO: Add description.
          height (`int`):
              TODO: Add description.
          width (`int`):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          grid_sizes_ref (`Tensor`):
              Post-patch latent grid `[[T, H/2, W/2]]` of a driving-video segment, used as the offset grid of the
              reference-extraction pass and the reference grid of the denoising passes
          max_seq_len (`int`):
              Packed sequence length of the generation tokens
          max_seq_len_ref (`int`):
              Packed sequence length of the reference tokens
          driving_video_latents (`Tensor`):
              VAE latents of this segment's driving-video slice
          driving_video_condition (`Tensor`):
              i2v mask + driving-slice latents, conditioning the reference-extraction pass
          reference_latents (`Tensor`):
              The full conditioning tensor: reference half stacked over the segment half
          latents (`Tensor`):
              This segment's initial noise
          kv_cache (`WanAnimate2KVCache`):
              Fresh per-segment cache for the reference K/V
          timesteps (`Tensor`):
              This segment's denoising timesteps
          out_frames (`Tensor`):
              This segment's decoded frames on device; the next segment conditions on its tail
          segment_frames (`list`):
              Per-segment decoded frames on CPU, each `[1, 3, T, H, W]`
    """

    model_name = "wan-animate-2"
    block_classes = WanAnimate2CoreDenoiseBlocks.values()
    block_names = WanAnimate2CoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step that computes the segment-invariant geometry and runs the segment-by-segment "
            "denoising loop, decoding each segment inside the loop because the next segment conditions on its "
            "decoded pixels."
        )


# ====================
# 3. Blocks
# ====================


BLOCKS = InsertableDict(
    [
        ("text_encoder", WanAnimate2TextEncoderStep()),
        ("image_encoder", WanAnimate2ImageEncodeStep()),
        ("video_encoder", WanAnimate2VideoEncodeStep()),
        ("vae_encoder", WanAnimate2ImageVaeEncoderStep()),
        ("denoise", WanAnimate2CoreDenoiseStep()),
        ("decode", WanAnimate2DecodeStep()),
    ]
)


# auto_docstring
class WanAnimate2Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for Wan-Animate-2 character animation: a reference character image and a driving video produce a video of the character following the driving motion.

      Components:
          text_encoder (`UMT5EncoderModel`)
          tokenizer (`AutoTokenizer`)
          image_processor (`WanAnimate2VideoProcessor`)
          image_encoder (`CLIPVisionModel`)
          video_processor (`WanAnimate2VideoProcessor`)
          vae (`AutoencoderKLWan`)
          transformer (`WanAnimate2Transformer3DModel`)
          scheduler (`SchedulerMixin`)
          guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              TODO: Add description.
          negative_prompt (`str`, *optional*):
              TODO: Add description.
          prompt_ref (`str`, *optional*, defaults to 人物动作的参考视频):
              The reference prompt for the driving video context
          max_sequence_length (`None`, *optional*, defaults to 512):
              TODO: Add description.
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 800):
              Together with `width`, the target *area* of the generated video; the aspect ratio comes from `image`. Overwritten
              with the resolved frame height.
          width (`int`, *optional*, defaults to 640):
              See `height`. Overwritten with the resolved frame width.
          driving_video (`list`):
              The driving video that provides the motion, in any format accepted by `VideoProcessor.preprocess_video`.
          driving_video_fps (`float`, *optional*):
              The frame rate `driving_video` was captured at — `load_video(..., return_fps=True)` reports it. When set, the
              driving frames are resampled from it to `fps`; when `None` they are used as-is.
          fps (`int`, *optional*, defaults to 24):
              The frame rate the model generates at
          segment_frame_length (`int`, *optional*, defaults to 81):
              The number of frames in each inference segment
          prev_segment_conditioning_frames (`int`, *optional*, defaults to 1):
              The number of conditioning frames carried over from the previous segment
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`int`, *optional*, defaults to 40):
              TODO: Add description.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          output_type (`str`, *optional*, defaults to np):
              The output type of the decoded videos

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "wan-animate-2"
    block_classes = BLOCKS.values()
    block_names = BLOCKS.keys()

    @property
    def description(self):
        return (
            "Modular pipeline blocks for Wan-Animate-2 character animation: a reference character image and a "
            "driving video produce a video of the character following the driving motion."
        )

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
