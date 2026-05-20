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
    HunyuanVideo15Image2VideoPrepareLatentsStep,
    HunyuanVideo15PrepareLatentsStep,
    HunyuanVideo15SetTimestepsStep,
    HunyuanVideo15TextInputStep,
)
from .decoders import HunyuanVideo15VaeDecoderStep
from .denoise import HunyuanVideo15DenoiseStep, HunyuanVideo15Image2VideoDenoiseStep
from .encoders import (
    HunyuanVideo15ImageEncoderStep,
    HunyuanVideo15TextEncoderStep,
    HunyuanVideo15VaeEncoderStep,
)


logger = logging.get_logger(__name__)


# auto_docstring
class HunyuanVideo15CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block that takes encoded conditions and runs the denoising process.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`HunyuanVideo15Transformer3DModel`)
          video_processor (`HunyuanVideo15ImageProcessor`) guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          batch_size (`int`, *optional*):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              Number of video frames to generate.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          negative_prompt_embeds (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds' field fed into the guider.
          prompt_embeds_mask (`Tensor`):
              Positive branch of the 'prompt_embeds_mask' field fed into the guider.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_mask' field fed into the guider.
          prompt_embeds_2 (`Tensor`):
              Positive branch of the 'prompt_embeds_2' field fed into the guider.
          negative_prompt_embeds_2 (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_2' field fed into the guider.
          prompt_embeds_mask_2 (`Tensor`):
              Positive branch of the 'prompt_embeds_mask_2' field fed into the guider.
          negative_prompt_embeds_mask_2 (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_mask_2' field fed into the guider.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextInputStep,
        HunyuanVideo15SetTimestepsStep,
        HunyuanVideo15PrepareLatentsStep,
        HunyuanVideo15DenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class HunyuanVideo15Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for HunyuanVideo 1.5 text-to-video.

      Components:
          text_encoder (`Qwen2_5_VLTextModel`) tokenizer (`Qwen2Tokenizer`) text_encoder_2 (`T5EncoderModel`)
          tokenizer_2 (`ByT5Tokenizer`) guider (`ClassifierFreeGuidance`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          transformer (`HunyuanVideo15Transformer3DModel`) video_processor (`HunyuanVideo15ImageProcessor`) vae
          (`AutoencoderKLHunyuanVideo15`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`, *optional*):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              Number of video frames to generate.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextEncoderStep,
        HunyuanVideo15CoreDenoiseStep,
        HunyuanVideo15VaeDecoderStep,
    ]
    block_names = ["text_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for HunyuanVideo 1.5 text-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]


# auto_docstring
class HunyuanVideo15Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block for image-to-video that takes encoded conditions and runs the denoising process.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`HunyuanVideo15Transformer3DModel`)
          video_processor (`HunyuanVideo15ImageProcessor`) guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          batch_size (`int`, *optional*):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              Number of video frames to generate.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image_latents (`Tensor`):
              Pre-encoded image latents from the VAE encoder step, used as conditioning for I2V.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          negative_prompt_embeds (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds' field fed into the guider.
          prompt_embeds_mask (`Tensor`):
              Positive branch of the 'prompt_embeds_mask' field fed into the guider.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_mask' field fed into the guider.
          prompt_embeds_2 (`Tensor`):
              Positive branch of the 'prompt_embeds_2' field fed into the guider.
          negative_prompt_embeds_2 (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_2' field fed into the guider.
          prompt_embeds_mask_2 (`Tensor`):
              Positive branch of the 'prompt_embeds_mask_2' field fed into the guider.
          negative_prompt_embeds_mask_2 (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_mask_2' field fed into the guider.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextInputStep,
        HunyuanVideo15SetTimestepsStep,
        HunyuanVideo15PrepareLatentsStep,
        HunyuanVideo15Image2VideoPrepareLatentsStep,
        HunyuanVideo15Image2VideoDenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "prepare_i2v_latents", "denoise"]

    @property
    def description(self):
        return "Denoise block for image-to-video that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class HunyuanVideo15AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the image input into its latent representation.
      This is an auto pipeline block that works for image-to-video tasks.
       - `HunyuanVideo15VaeEncoderStep` is used when `image` is provided.
       - If `image` is not provided, step will be skipped.

      Components:
          vae (`AutoencoderKLHunyuanVideo15`) video_processor (`HunyuanVideo15ImageProcessor`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.

      Outputs:
          image_latents (`Tensor`):
              Encoded image latents from the VAE encoder
          height (`int`):
              Target height resolved from image
          width (`int`):
              Target width resolved from image
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [HunyuanVideo15VaeEncoderStep]
    block_names = ["vae_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the image input into its latent representation.\n"
            "This is an auto pipeline block that works for image-to-video tasks.\n"
            " - `HunyuanVideo15VaeEncoderStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


# auto_docstring
class HunyuanVideo15AutoImageEncoderStep(AutoPipelineBlocks):
    """
    Siglip image encoder step that produces image_embeds.
      This is an auto pipeline block that works for image-to-video tasks.
       - `HunyuanVideo15ImageEncoderStep` is used when `image` is provided.
       - If `image` is not provided, step will be skipped.

      Components:
          image_encoder (`SiglipVisionModel`) feature_extractor (`SiglipImageProcessor`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.

      Outputs:
          image_embeds (`Tensor`):
              Image embeddings from the Siglip vision encoder
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [HunyuanVideo15ImageEncoderStep]
    block_names = ["image_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Siglip image encoder step that produces image_embeds.\n"
            "This is an auto pipeline block that works for image-to-video tasks.\n"
            " - `HunyuanVideo15ImageEncoderStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


# auto_docstring
class HunyuanVideo15AutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Auto denoise block that selects the appropriate denoise pipeline based on inputs.
       - `HunyuanVideo15Image2VideoCoreDenoiseStep` is used when `image_latents` is provided.
       - `HunyuanVideo15CoreDenoiseStep` is used otherwise (text-to-video).

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`HunyuanVideo15Transformer3DModel`)
          video_processor (`HunyuanVideo15ImageProcessor`) guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              Number of video frames to generate.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image_latents (`Tensor`, *optional*):
              Pre-encoded image latents from the VAE encoder step, used as conditioning for I2V.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          negative_prompt_embeds (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds' field fed into the guider.
          prompt_embeds_mask (`Tensor`):
              Positive branch of the 'prompt_embeds_mask' field fed into the guider.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_mask' field fed into the guider.
          prompt_embeds_2 (`Tensor`):
              Positive branch of the 'prompt_embeds_2' field fed into the guider.
          negative_prompt_embeds_2 (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_2' field fed into the guider.
          prompt_embeds_mask_2 (`Tensor`):
              Positive branch of the 'prompt_embeds_mask_2' field fed into the guider.
          negative_prompt_embeds_mask_2 (`Tensor`, *optional*):
              Negative branch of the 'negative_prompt_embeds_mask_2' field fed into the guider.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [HunyuanVideo15Image2VideoCoreDenoiseStep, HunyuanVideo15CoreDenoiseStep]
    block_names = ["image2video", "text2video"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Auto denoise block that selects the appropriate denoise pipeline based on inputs.\n"
            " - `HunyuanVideo15Image2VideoCoreDenoiseStep` is used when `image_latents` is provided.\n"
            " - `HunyuanVideo15CoreDenoiseStep` is used otherwise (text-to-video)."
        )


# auto_docstring
class HunyuanVideo15AutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks for HunyuanVideo 1.5 that support both text-to-video and image-to-video workflows.

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `image`, `prompt`

      Components:
          text_encoder (`Qwen2_5_VLTextModel`) tokenizer (`Qwen2Tokenizer`) text_encoder_2 (`T5EncoderModel`)
          tokenizer_2 (`ByT5Tokenizer`) guider (`ClassifierFreeGuidance`) vae (`AutoencoderKLHunyuanVideo15`)
          video_processor (`HunyuanVideo15ImageProcessor`) image_encoder (`SiglipVisionModel`) feature_extractor
          (`SiglipImageProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`) transformer
          (`HunyuanVideo15Transformer3DModel`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 121):
              Number of video frames to generate.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image_latents (`Tensor`, *optional*):
              Pre-encoded image latents from the VAE encoder step, used as conditioning for I2V.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextEncoderStep,
        HunyuanVideo15AutoVaeEncoderStep,
        HunyuanVideo15AutoImageEncoderStep,
        HunyuanVideo15AutoCoreDenoiseStep,
        HunyuanVideo15VaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "image_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
    }

    @property
    def description(self):
        return "Auto blocks for HunyuanVideo 1.5 that support both text-to-video and image-to-video workflows."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]


# auto_docstring
class HunyuanVideo15Image2VideoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for HunyuanVideo 1.5 image-to-video.

      Components:
          text_encoder (`Qwen2_5_VLTextModel`) tokenizer (`Qwen2Tokenizer`) text_encoder_2 (`T5EncoderModel`)
          tokenizer_2 (`ByT5Tokenizer`) guider (`ClassifierFreeGuidance`) vae (`AutoencoderKLHunyuanVideo15`)
          video_processor (`HunyuanVideo15ImageProcessor`) image_encoder (`SiglipVisionModel`) feature_extractor
          (`SiglipImageProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`) transformer
          (`HunyuanVideo15Transformer3DModel`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          batch_size (`int`, *optional*):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 121):
              Number of video frames to generate.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image_latents (`Tensor`):
              Pre-encoded image latents from the VAE encoder step, used as conditioning for I2V.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "hunyuan-video-1.5"
    block_classes = [
        HunyuanVideo15TextEncoderStep,
        HunyuanVideo15AutoVaeEncoderStep,
        HunyuanVideo15AutoImageEncoderStep,
        HunyuanVideo15Image2VideoCoreDenoiseStep,
        HunyuanVideo15VaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "image_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for HunyuanVideo 1.5 image-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
