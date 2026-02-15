# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    WanAdditionalInputsStep,
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
    WanTextInputStep,
)
from .decoders import WanVaeDecoderStep
from .denoise import (
    WanImage2VideoDenoiseStep,
)
from .encoders import (
    WanFirstLastFrameImageEncoderStep,
    WanFirstLastFrameVaeEncoderStep,
    WanImageCropResizeStep,
    WanImageEncoderStep,
    WanImageResizeStep,
    WanPrepareFirstFrameLatentsStep,
    WanPrepareFirstLastFrameLatentsStep,
    WanTextEncoderStep,
    WanVaeEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# ====================
# 1. IMAGE ENCODER
# ====================


# wan2.1 I2V (first frame only)
# auto_docstring
class WanImage2VideoImageEncoderStep(SequentialPipelineBlocks):
    """
    Image2Video Image Encoder step that resize the image and encode the image to generate the image embeddings

      Components:
          image_processor (`CLIPImageProcessor`) image_encoder (`CLIPVisionModel`)

      Inputs:
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.

      Outputs:
          resized_image (`Image`):
              TODO: Add description.
          image_embeds (`Tensor`):
              The image embeddings
    """

    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanImageEncoderStep]
    block_names = ["image_resize", "image_encoder"]

    @property
    def description(self):
        return "Image2Video Image Encoder step that resize the image and encode the image to generate the image embeddings"


# wan2.1 FLF2V (first and last frame)
# auto_docstring
class WanFLF2VImageEncoderStep(SequentialPipelineBlocks):
    """
    FLF2V Image Encoder step that resize and encode and encode the first and last frame images to generate the image
    embeddings

      Components:
          image_processor (`CLIPImageProcessor`) image_encoder (`CLIPVisionModel`)

      Inputs:
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.
          last_image (`Image`):
              The last frameimage

      Outputs:
          resized_image (`Image`):
              TODO: Add description.
          resized_last_image (`Image`):
              TODO: Add description.
          image_embeds (`Tensor`):
              The image embeddings
    """

    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanImageCropResizeStep, WanFirstLastFrameImageEncoderStep]
    block_names = ["image_resize", "last_image_resize", "image_encoder"]

    @property
    def description(self):
        return "FLF2V Image Encoder step that resize and encode and encode the first and last frame images to generate the image embeddings"


# wan2.1 Auto Image Encoder
# auto_docstring
class WanAutoImageEncoderStep(AutoPipelineBlocks):
    """
    Image Encoder step that encode the image to generate the image embeddingsThis is an auto pipeline block that works
    for image2video tasks. - `WanFLF2VImageEncoderStep` (flf2v) is used when `last_image` is provided. -
    `WanImage2VideoImageEncoderStep` (image2video) is used when `image` is provided. - if `last_image` or `image` is
    not provided, step will be skipped.

      Components:
          image_processor (`CLIPImageProcessor`) image_encoder (`CLIPVisionModel`)

      Inputs:
          image (`Image`, *optional*):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.
          last_image (`Image`, *optional*):
              The last frameimage

      Outputs:
          resized_image (`Image`):
              TODO: Add description.
          resized_last_image (`Image`):
              TODO: Add description.
          image_embeds (`Tensor`):
              The image embeddings
    """

    block_classes = [WanFLF2VImageEncoderStep, WanImage2VideoImageEncoderStep]
    block_names = ["flf2v_image_encoder", "image2video_image_encoder"]
    block_trigger_inputs = ["last_image", "image"]
    model_name = "wan-i2v"

    @property
    def description(self):
        return (
            "Image Encoder step that encode the image to generate the image embeddings"
            + "This is an auto pipeline block that works for image2video tasks."
            + " - `WanFLF2VImageEncoderStep` (flf2v) is used when `last_image` is provided."
            + " - `WanImage2VideoImageEncoderStep` (image2video) is used when `image` is provided."
            + " - if `last_image` or `image` is not provided, step will be skipped."
        )


# ====================
# 2. VAE ENCODER
# ====================


# wan2.1 I2V (first frame only)
# auto_docstring
class WanImage2VideoVaeEncoderStep(SequentialPipelineBlocks):
    """
    Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent
    representation

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.
          num_frames (`int`, *optional*, defaults to 81):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          resized_image (`Image`):
              TODO: Add description.
          first_frame_latents (`Tensor`):
              video latent representation with the first frame image condition
          image_condition_latents (`Tensor | NoneType`):
              TODO: Add description.
    """

    model_name = "wan-i2v"
    block_classes = [WanImageResizeStep, WanVaeEncoderStep, WanPrepareFirstFrameLatentsStep]
    block_names = ["image_resize", "vae_encoder", "prepare_first_frame_latents"]

    @property
    def description(self):
        return "Image2Video Vae Image Encoder step that resize the image and encode the first frame image to its latent representation"


# wan2.1 FLF2V (first and last frame)
# auto_docstring
class WanFLF2VVaeEncoderStep(SequentialPipelineBlocks):
    """
    FLF2V Vae Image Encoder step that resize and encode and encode the first and last frame images to generate the
    latent conditions

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image`):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.
          last_image (`Image`):
              The last frameimage
          num_frames (`int`, *optional*, defaults to 81):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          resized_image (`Image`):
              TODO: Add description.
          resized_last_image (`Image`):
              TODO: Add description.
          first_last_frame_latents (`Tensor`):
              video latent representation with the first and last frame images condition
          image_condition_latents (`Tensor | NoneType`):
              TODO: Add description.
    """

    model_name = "wan-i2v"
    block_classes = [
        WanImageResizeStep,
        WanImageCropResizeStep,
        WanFirstLastFrameVaeEncoderStep,
        WanPrepareFirstLastFrameLatentsStep,
    ]
    block_names = ["image_resize", "last_image_resize", "vae_encoder", "prepare_first_last_frame_latents"]

    @property
    def description(self):
        return "FLF2V Vae Image Encoder step that resize and encode and encode the first and last frame images to generate the latent conditions"


# wan2.1 Auto Vae Encoder
# auto_docstring
class WanAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    Vae Image Encoder step that encode the image to generate the image latentsThis is an auto pipeline block that works
    for image2video tasks. - `WanFLF2VVaeEncoderStep` (flf2v) is used when `last_image` is provided. -
    `WanImage2VideoVaeEncoderStep` (image2video) is used when `image` is provided. - if `last_image` or `image` is not
    provided, step will be skipped.

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image`, *optional*):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.
          last_image (`Image`, *optional*):
              The last frameimage
          num_frames (`int`, *optional*, defaults to 81):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          resized_image (`Image`):
              TODO: Add description.
          resized_last_image (`Image`):
              TODO: Add description.
          first_last_frame_latents (`Tensor`):
              video latent representation with the first and last frame images condition
          image_condition_latents (`Tensor | NoneType`):
              TODO: Add description.
          first_frame_latents (`Tensor`):
              video latent representation with the first frame image condition
    """

    model_name = "wan-i2v"
    block_classes = [WanFLF2VVaeEncoderStep, WanImage2VideoVaeEncoderStep]
    block_names = ["flf2v_vae_encoder", "image2video_vae_encoder"]
    block_trigger_inputs = ["last_image", "image"]

    @property
    def description(self):
        return (
            "Vae Image Encoder step that encode the image to generate the image latents"
            + "This is an auto pipeline block that works for image2video tasks."
            + " - `WanFLF2VVaeEncoderStep` (flf2v) is used when `last_image` is provided."
            + " - `WanImage2VideoVaeEncoderStep` (image2video) is used when `image` is provided."
            + " - if `last_image` or `image` is not provided, step will be skipped."
        )


# ====================
# 3. DENOISE (inputs -> set_timesteps -> prepare_latents -> denoise)
# ====================


# wan2.1 I2V core denoise (support both I2V and FLF2V)
# inputs (text + image_condition_latents) -> set_timesteps -> prepare_latents -> denoise (latents)
# auto_docstring
class WanImage2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    denoise block that takes encoded text and image latent conditions and runs the denoising process.

      Components:
          transformer (`WanTransformer3DModel`) scheduler (`UniPCMultistepScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          num_frames (`None`, *optional*):
              TODO: Add description.
          image_condition_latents (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          attention_kwargs (`None`, *optional*):
              TODO: Add description.
          image_embeds (`Tensor`):
              TODO: Add description.

      Outputs:
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_videos_per_prompt
          dtype (`dtype`):
              Data type of model tensor inputs (determined by `transformer.dtype`)
          latents (`Tensor`):
              The initial latents to use for the denoising process
    """

    model_name = "wan-i2v"
    block_classes = [
        WanTextInputStep,
        WanAdditionalInputsStep(image_latent_inputs=["image_condition_latents"]),
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        WanImage2VideoDenoiseStep,
    ]
    block_names = [
        "input",
        "additional_inputs",
        "set_timesteps",
        "prepare_latents",
        "denoise",
    ]

    @property
    def description(self):
        return "denoise block that takes encoded text and image latent conditions and runs the denoising process."


# ====================
# 4. BLOCKS (Wan2.1 Image2Video)
# ====================


# wan2.1 Image2Video Auto Blocks
# auto_docstring
class WanImage2VideoAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for image-to-video using Wan.

      Supported workflows:
        - `image2video`: requires `image`, `prompt`
        - `flf2v`: requires `last_image`, `image`, `prompt`

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) guider (`ClassifierFreeGuidance`)
          image_processor (`CLIPImageProcessor`) image_encoder (`CLIPVisionModel`) vae (`AutoencoderKLWan`)
          video_processor (`VideoProcessor`) transformer (`WanTransformer3DModel`) scheduler
          (`UniPCMultistepScheduler`)

      Inputs:
          prompt (`None`, *optional*):
              TODO: Add description.
          negative_prompt (`None`, *optional*):
              TODO: Add description.
          max_sequence_length (`None`, *optional*, defaults to 512):
              TODO: Add description.
          image (`Image`, *optional*):
              TODO: Add description.
          height (`int`, *optional*, defaults to 480):
              TODO: Add description.
          width (`int`, *optional*, defaults to 832):
              TODO: Add description.
          last_image (`Image`, *optional*):
              The last frameimage
          num_frames (`int`, *optional*, defaults to 81):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_videos_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          image_condition_latents (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          attention_kwargs (`None`, *optional*):
              TODO: Add description.
          image_embeds (`Tensor`):
              TODO: Add description.
          output_type (`str`, *optional*, defaults to np):
              The output type of the decoded videos

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "wan-i2v"
    block_classes = [
        WanTextEncoderStep,
        WanAutoImageEncoderStep,
        WanAutoVaeEncoderStep,
        WanImage2VideoCoreDenoiseStep,
        WanVaeDecoderStep,
    ]
    block_names = [
        "text_encoder",
        "image_encoder",
        "vae_encoder",
        "denoise",
        "decode",
    ]

    _workflow_map = {
        "image2video": {"image": True, "prompt": True},
        "flf2v": {"last_image": True, "image": True, "prompt": True},
    }

    @property
    def description(self):
        return "Auto Modular pipeline for image-to-video using Wan."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
