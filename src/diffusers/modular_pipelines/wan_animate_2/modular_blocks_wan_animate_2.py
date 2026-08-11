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
from ..modular_pipeline_utils import OutputParam
from .before_denoise import WanAnimate2PrepareSegmentsStep
from .decoders import WanAnimate2DecodeStep
from .denoise import WanAnimate2DenoiseStep
from .encoders import (
    WanAnimate2DrivingImageEncoderStep,
    WanAnimate2ImageEncoderStep,
    WanAnimate2ImageResizeStep,
    WanAnimate2RefVaeEncoderStep,
    WanAnimate2TextEncoderStep,
    WanAnimate2VideoPreprocessStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class WanAnimate2Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for Wan-Animate-2 character animation: a reference character image and a driving video produce a video of the character following the driving motion.

      Components:
          text_encoder (`UMT5EncoderModel`)
          tokenizer (`AutoTokenizer`)
          guider (`ClassifierFreeGuidance`)
          image_processor (`WanAnimate2VideoProcessor`)
          video_processor (`WanAnimate2VideoProcessor`)
          image_encoder (`CLIPVisionModel`)
          vae (`AutoencoderKLWan`)
          transformer (`WanAnimate2Transformer3DModel`)
          scheduler (`SchedulerMixin`)

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
          clip_len (`int`, *optional*, defaults to 81):
              The number of frames in each inference segment
          first_num (`int`, *optional*, defaults to 1):
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
    block_classes = [
        WanAnimate2TextEncoderStep,
        WanAnimate2ImageResizeStep,
        WanAnimate2VideoPreprocessStep,
        WanAnimate2ImageEncoderStep,
        WanAnimate2DrivingImageEncoderStep,
        WanAnimate2RefVaeEncoderStep,
        WanAnimate2PrepareSegmentsStep,
        WanAnimate2DenoiseStep,
        WanAnimate2DecodeStep,
    ]
    block_names = [
        "text_encoder",
        "image_resize",
        "video_preprocess",
        "image_encoder",
        "driving_image_encoder",
        "ref_vae_encoder",
        "prepare_segments",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Modular pipeline blocks for Wan-Animate-2 character animation: a reference character image and a "
            "driving video produce a video of the character following the driving motion."
        )

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
