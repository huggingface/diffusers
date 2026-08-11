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
from .denoise import WanAnimate2DistilledDenoiseStep
from .encoders import WanAnimate2TextEncoderStep
from .modular_blocks_wan_animate_2 import (
    WanAnimate2ImageEncodeStep,
    WanAnimate2VaeEncodeStep,
    WanAnimate2VideoEncodeStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


WanAnimate2DistilledCoreDenoiseBlocks = InsertableDict(
    [
        ("prepare_segments", WanAnimate2PrepareSegmentsStep()),
        ("denoise", WanAnimate2DistilledDenoiseStep()),
    ]
)


# auto_docstring
class WanAnimate2DistilledCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step for the distilled Wan-Animate-2 checkpoint: computes the segment-invariant geometry and runs the segment-by-segment denoising loop in few steps without classifier-free guidance.

      Components:
          vae (`AutoencoderKLWan`)
          transformer (`WanAnimate2Transformer3DModel`)
          scheduler (`SchedulerMixin`)
          guider (`ClassifierFreeGuidance`)

      Inputs:
          segment_frame_length (`int`, *optional*, defaults to 81):
              TODO: Add description.
          latent_height (`int`):
              TODO: Add description.
          latent_width (`int`):
              TODO: Add description.
          num_segments (`int`):
              TODO: Add description.
          y_ref (`Tensor`):
              TODO: Add description.
          prev_segment_conditioning_frames (`int`, *optional*, defaults to 1):
              TODO: Add description.
          height (`int`):
              TODO: Add description.
          width (`int`):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`int`, *optional*, defaults to 40):
              TODO: Add description.
          condition_latents (`Tensor`):
              VAE latents of every segment's driving-video slice, `[num_segments, 16, T, H', W']`
          condition_y (`Tensor`):
              i2v mask + driving-slice latents per segment, `[num_segments, 20, T, H', W']`
          condition_clip_context (`Tensor`):
              TODO: Add description.
          prompt_ref_embeds (`Tensor`):
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
          y (`Tensor`):
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
    block_classes = WanAnimate2DistilledCoreDenoiseBlocks.values()
    block_names = WanAnimate2DistilledCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step for the distilled Wan-Animate-2 checkpoint: computes the segment-invariant geometry "
            "and runs the segment-by-segment denoising loop in few steps without classifier-free guidance."
        )


DISTILLED_BLOCKS = InsertableDict(
    [
        ("text_encoder", WanAnimate2TextEncoderStep()),
        ("image_encoder", WanAnimate2ImageEncodeStep()),
        ("video_encoder", WanAnimate2VideoEncodeStep()),
        ("vae_encoder", WanAnimate2VaeEncodeStep()),
        ("denoise", WanAnimate2DistilledCoreDenoiseStep()),
        ("decode", WanAnimate2DecodeStep()),
    ]
)


# auto_docstring
class WanAnimate2DistilledBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for distilled Wan-Animate-2 character animation, sampling in few steps without classifier-free guidance.

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
              Overwritten with the preprocessed `[1, 3, T, H, W]` tensor.
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
    block_classes = DISTILLED_BLOCKS.values()
    block_names = DISTILLED_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Modular pipeline blocks for distilled Wan-Animate-2 character animation, sampling in few steps "
            "without classifier-free guidance."
        )

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
