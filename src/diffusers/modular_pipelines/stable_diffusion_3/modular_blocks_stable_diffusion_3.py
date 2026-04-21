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
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    StableDiffusion3Img2ImgPrepareLatentsStep,
    StableDiffusion3Img2ImgSetTimestepsStep,
    StableDiffusion3PrepareLatentsStep,
    StableDiffusion3SetTimestepsStep,
)
from .decoders import StableDiffusion3DecodeStep
from .denoise import StableDiffusion3DenoiseStep
from .encoders import (
    StableDiffusion3ProcessImagesInputStep,
    StableDiffusion3TextEncoderStep,
    StableDiffusion3VaeEncoderStep,
)
from .inputs import StableDiffusion3AdditionalInputsStep, StableDiffusion3TextInputStep


logger = logging.get_logger(__name__)


# auto_docstring
class StableDiffusion3Img2ImgVaeEncoderStep(SequentialPipelineBlocks):
    """
    Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`)

      Inputs:
          resized_image (`None`, *optional*):
              The pre-resized image input.
          image (`None`, *optional*):
              The input image to be used as the starting point for the image-to-image process.
          height (`None`, *optional*):
              The height in pixels of the generated image.
          width (`None`, *optional*):
              The width in pixels of the generated image.
          generator (`None`, *optional*):
              One or a list of torch generator(s) to make generation deterministic.

      Outputs:
          processed_image (`None`):
              The pre-processed image tensor.
          image_latents (`Tensor`):
              The latents representing the reference image
    """

    model_name = "stable-diffusion-3"
    block_classes = [
        StableDiffusion3ProcessImagesInputStep(),
        StableDiffusion3VaeEncoderStep(),
    ]
    block_names = ["preprocess", "encode"]


# auto_docstring
class StableDiffusion3AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`)

      Inputs:
          resized_image (`None`, *optional*):
              The pre-resized image input.
          image (`None`, *optional*):
              The input image to be used as the starting point for the image-to-image process.
          height (`None`, *optional*):
              The height in pixels of the generated image.
          width (`None`, *optional*):
              The width in pixels of the generated image.
          generator (`None`, *optional*):
              One or a list of torch generator(s) to make generation deterministic.

      Outputs:
          processed_image (`None`):
              The pre-processed image tensor.
          image_latents (`Tensor`):
              The latents representing the reference image
    """

    model_name = "stable-diffusion-3"
    block_classes = [StableDiffusion3Img2ImgVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs = ["image"]


# auto_docstring
class StableDiffusion3T2ICoreDenoiseStep(SequentialPipelineBlocks):
    """
    Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`SD3Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings.
          pooled_prompt_embeds (`Tensor`):
              Pre-generated pooled text embeddings.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings.
          negative_pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative pooled text embeddings.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy latents to be used as inputs for image generation.
          generator (`None`, *optional*):
              One or a list of torch generator(s) to make generation deterministic.
          num_inference_steps (`None`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`None`, *optional*):
              Custom timesteps to use for the denoising process.
          sigmas (`None`, *optional*):
              Custom sigmas to use for the denoising process.
          mu (`float`, *optional*):
              The mu value used for dynamic shifting. If not provided, it is dynamically calculated.
          joint_attention_kwargs (`dict`, *optional*):
              A kwargs dictionary passed along to the AttentionProcessor.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "stable-diffusion-3"
    block_classes = [
        StableDiffusion3TextInputStep(),
        StableDiffusion3PrepareLatentsStep(),
        StableDiffusion3SetTimestepsStep(),
        StableDiffusion3DenoiseStep(),
    ]
    block_names = ["text_inputs", "prepare_latents", "set_timesteps", "denoise"]

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class StableDiffusion3I2ICoreDenoiseStep(SequentialPipelineBlocks):
    """
    Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`SD3Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings.
          pooled_prompt_embeds (`Tensor`):
              Pre-generated pooled text embeddings.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings.
          negative_pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative pooled text embeddings.
          height (`None`, *optional*):
              The height in pixels of the generated image.
          width (`None`, *optional*):
              The width in pixels of the generated image.
          image_latents (`None`, *optional*):
              Latent input image_latents to be processed.
          latents (`Tensor | NoneType`, *optional*):
              Pre-generated noisy latents to be used as inputs for image generation.
          generator (`None`, *optional*):
              One or a list of torch generator(s) to make generation deterministic.
          num_inference_steps (`None`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`None`, *optional*):
              Custom timesteps to use for the denoising process.
          sigmas (`None`, *optional*):
              Custom sigmas to use for the denoising process.
          strength (`None`, *optional*, defaults to 0.6):
              Indicates extent to transform the reference image.
          mu (`float`, *optional*):
              The mu value used for dynamic shifting. If not provided, it is dynamically calculated.
          joint_attention_kwargs (`dict`, *optional*):
              A kwargs dictionary passed along to the AttentionProcessor.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "stable-diffusion-3"
    block_classes = [
        StableDiffusion3TextInputStep(),
        StableDiffusion3AdditionalInputsStep(),
        StableDiffusion3PrepareLatentsStep(),
        StableDiffusion3Img2ImgSetTimestepsStep(),
        StableDiffusion3Img2ImgPrepareLatentsStep(),
        StableDiffusion3DenoiseStep(),
    ]
    block_names = [
        "text_inputs",
        "additional_inputs",
        "prepare_latents",
        "set_timesteps",
        "prepare_img2img_latents",
        "denoise",
    ]

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class StableDiffusion3AutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer
          (`SD3Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings.
          pooled_prompt_embeds (`Tensor`):
              Pre-generated pooled text embeddings.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings.
          negative_pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative pooled text embeddings.
          height (`None`, *optional*):
              The height in pixels of the generated image.
          width (`None`, *optional*):
              The width in pixels of the generated image.
          image_latents (`None`, *optional*):
              Latent input image_latents to be processed.
          latents (`Tensor | NoneType`):
              Pre-generated noisy latents to be used as inputs for image generation.
          generator (`None`, *optional*):
              One or a list of torch generator(s) to make generation deterministic.
          num_inference_steps (`None`):
              The number of denoising steps.
          timesteps (`None`):
              Custom timesteps to use for the denoising process.
          sigmas (`None`, *optional*):
              Custom sigmas to use for the denoising process.
          strength (`None`, *optional*, defaults to 0.6):
              Indicates extent to transform the reference image.
          mu (`float`, *optional*):
              The mu value used for dynamic shifting. If not provided, it is dynamically calculated.
          joint_attention_kwargs (`dict`, *optional*):
              A kwargs dictionary passed along to the AttentionProcessor.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "stable-diffusion-3"
    block_classes = [
        StableDiffusion3I2ICoreDenoiseStep,
        StableDiffusion3T2ICoreDenoiseStep,
    ]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", StableDiffusion3TextEncoderStep()),
        ("vae_encoder", StableDiffusion3AutoVaeEncoderStep()),
        ("denoise", StableDiffusion3AutoCoreDenoiseStep()),
        ("decode", StableDiffusion3DecodeStep()),
    ]
)


# auto_docstring
class StableDiffusion3AutoBlocks(SequentialPipelineBlocks):
    """
    Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `image`, `prompt`

      Components:
          text_encoder (`CLIPTextModelWithProjection`) tokenizer (`CLIPTokenizer`) text_encoder_2
          (`CLIPTextModelWithProjection`) tokenizer_2 (`CLIPTokenizer`) text_encoder_3 (`T5EncoderModel`) tokenizer_3
          (`T5TokenizerFast`) image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`) transformer (`SD3Transformer2DModel`)

      Inputs:
          prompt (`None`, *optional*):
              The prompt or prompts to guide the image generation.
          prompt_2 (`None`, *optional*):
              The prompt or prompts to be sent to tokenizer_2 and text_encoder_2.
          prompt_3 (`None`, *optional*):
              The prompt or prompts to be sent to tokenizer_3 and text_encoder_3.
          negative_prompt (`None`, *optional*):
              The prompt or prompts not to guide the image generation.
          negative_prompt_2 (`None`, *optional*):
              The prompt or prompts not to guide the image generation for tokenizer_2.
          negative_prompt_3 (`None`, *optional*):
              The prompt or prompts not to guide the image generation for tokenizer_3.
          clip_skip (`int`, *optional*):
              Number of layers to be skipped from CLIP while computing the prompt embeddings.
          max_sequence_length (`int`, *optional*, defaults to 256):
              Maximum sequence length to use with the prompt.
          joint_attention_kwargs (`None`, *optional*):
              A kwargs dictionary passed along to the AttentionProcessor.
          resized_image (`None`, *optional*):
              The pre-resized image input.
          image (`None`, *optional*):
              The input image to be used as the starting point for the image-to-image process.
          height (`None`, *optional*):
              The height in pixels of the generated image.
          width (`None`, *optional*):
              The width in pixels of the generated image.
          generator (`None`, *optional*):
              One or a list of torch generator(s) to make generation deterministic.
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image_latents (`None`, *optional*):
              Latent input image_latents to be processed.
          latents (`Tensor | NoneType`):
              Pre-generated noisy latents to be used as inputs for image generation.
          num_inference_steps (`None`):
              The number of denoising steps.
          timesteps (`None`):
              Custom timesteps to use for the denoising process.
          sigmas (`None`, *optional*):
              Custom sigmas to use for the denoising process.
          strength (`None`, *optional*, defaults to 0.6):
              Indicates extent to transform the reference image.
          mu (`float`, *optional*):
              The mu value used for dynamic shifting. If not provided, it is dynamically calculated.
          output_type (`None`, *optional*, defaults to pil):
              The output format of the generated image (e.g., 'pil', 'pt', 'np').

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "stable-diffusion-3"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"image": True, "prompt": True},
    }

    @property
    def outputs(self):
        return [OutputParam.template("images")]
