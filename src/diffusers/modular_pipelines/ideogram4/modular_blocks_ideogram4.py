# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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


from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    Ideogram4ApplyStrengthStep,
    Ideogram4ImageInputsStep,
    Ideogram4MaskInputsStep,
    Ideogram4PrepareAdditionalInputsStep,
    Ideogram4PrepareLatentsStep,
    Ideogram4PrepareLatentsWithStrengthStep,
    Ideogram4PrepareMaskLatentsStep,
    Ideogram4SetTimestepsStep,
    Ideogram4TextInputsStep,
)
from .decoders import Ideogram4DecodeStep, Ideogram4InpaintDecodeStep
from .denoise import Ideogram4AfterDenoiseStep, Ideogram4DenoiseStep, Ideogram4InpaintDenoiseStep
from .encoders import (
    Ideogram4InpaintProcessImagesInputStep,
    Ideogram4ProcessImageInputStep,
    Ideogram4PromptUpsampleStep,
    Ideogram4TextEncoderStep,
    Ideogram4VaeEncoderStep,
)


# auto_docstring
class Ideogram4Img2ImgVaeEncoderStep(SequentialPipelineBlocks):
    """
    Preprocess and encode an image into packed Ideogram4 latents for image-to-image generation.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLFlux2`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          processed_image (`Tensor`):
              The image tensor resized and normalized for VAE encoding.
          height (`int`):
              The resolved image height in pixels.
          width (`int`):
              The resolved image width in pixels.
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "ideogram4"
    block_classes = [Ideogram4ProcessImageInputStep(), Ideogram4VaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Preprocess and encode an image into packed Ideogram4 latents for image-to-image generation."


# auto_docstring
class Ideogram4InpaintVaeEncoderStep(SequentialPipelineBlocks):
    """
    Preprocess an image and mask, then encode the image into packed Ideogram4 latents for inpainting.

      Components:
          image_processor (`VaeImageProcessor`) image_mask_processor (`InpaintProcessor`) vae (`AutoencoderKLFlux2`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          mask_image (`Image`):
              Mask image for inpainting.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          processed_image (`Tensor`):
              The image tensor resized and normalized for VAE encoding.
          processed_mask_image (`Tensor`):
              The binary mask tensor resized to the generation resolution.
          mask_overlay_kwargs (`dict`):
              Arguments used to composite a cropped inpaint result over the source image.
          height (`int`):
              The resolved image height in pixels.
          width (`int`):
              The resolved image width in pixels.
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "ideogram4"
    block_classes = [Ideogram4InpaintProcessImagesInputStep(), Ideogram4VaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Preprocess an image and mask, then encode the image into packed Ideogram4 latents for inpainting."


# auto_docstring
class Ideogram4AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    Encode image inputs for Ideogram4 image-to-image and inpainting workflows. The step is skipped for text-to-image
    generation.

      Components:
          image_processor (`VaeImageProcessor`) image_mask_processor (`InpaintProcessor`) vae (`AutoencoderKLFlux2`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          mask_image (`Image`, *optional*):
              Mask image for inpainting.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          processed_image (`Tensor`):
              The image tensor resized and normalized for VAE encoding.
          processed_mask_image (`Tensor`):
              The binary mask tensor resized to the generation resolution.
          mask_overlay_kwargs (`dict`):
              Arguments used to composite a cropped inpaint result over the source image.
          height (`int`):
              The resolved image height in pixels.
          width (`int`):
              The resolved image width in pixels.
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    block_classes = [Ideogram4InpaintVaeEncoderStep, Ideogram4Img2ImgVaeEncoderStep]
    block_names = ["inpaint", "img2img"]
    block_trigger_inputs = ["mask_image", "image"]

    @property
    def description(self) -> str:
        return (
            "Encode image inputs for Ideogram4 image-to-image and inpainting workflows. The step is skipped for "
            "text-to-image generation."
        )


TEXT2IMAGE_DENOISE_BLOCKS = InsertableDict(
    [
        ("input", Ideogram4TextInputsStep()),
        ("prepare_latents", Ideogram4PrepareLatentsStep()),
        ("set_timesteps", Ideogram4SetTimestepsStep()),
        ("prepare_additional_inputs", Ideogram4PrepareAdditionalInputsStep()),
        ("denoise", Ideogram4DenoiseStep()),
        ("after_denoise", Ideogram4AfterDenoiseStep()),
    ]
)


# auto_docstring
class Ideogram4CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core Ideogram4 text-to-image denoising workflow.

      Components:
          transformer (`Ideogram4Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          unconditional_transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          text_features (`Tensor`):
              Per-prompt text features from the encoder.
          text_lengths (`list`):
              Per-prompt text-token counts from the encoder.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).

      Outputs:
          latents (`Tensor`):
              Unpatchified latents ready for the VAE decoder.
    """

    model_name = "ideogram4"
    block_classes = list(TEXT2IMAGE_DENOISE_BLOCKS.values())
    block_names = list(TEXT2IMAGE_DENOISE_BLOCKS.keys())

    @property
    def description(self) -> str:
        return "Core Ideogram4 text-to-image denoising workflow."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents", description="Unpatchified latents ready for the VAE decoder.")]


IMAGE2IMAGE_DENOISE_BLOCKS = InsertableDict(
    [
        ("text_inputs", Ideogram4TextInputsStep()),
        ("image_inputs", Ideogram4ImageInputsStep()),
        ("prepare_latents", Ideogram4PrepareLatentsStep()),
        ("set_timesteps", Ideogram4SetTimestepsStep()),
        ("apply_strength", Ideogram4ApplyStrengthStep()),
        ("prepare_image_latents", Ideogram4PrepareLatentsWithStrengthStep()),
        ("prepare_additional_inputs", Ideogram4PrepareAdditionalInputsStep()),
        ("denoise", Ideogram4DenoiseStep()),
        ("after_denoise", Ideogram4AfterDenoiseStep()),
    ]
)


# auto_docstring
class Ideogram4Img2ImgCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core Ideogram4 image-to-image denoising workflow with strength-based latent initialization.

      Components:
          transformer (`Ideogram4Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          unconditional_transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          text_features (`Tensor`):
              Per-prompt text features from the encoder.
          text_lengths (`list`):
              Per-prompt text-token counts from the encoder.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.

      Outputs:
          latents (`Tensor`):
              Unpatchified latents ready for the VAE decoder.
    """

    model_name = "ideogram4"
    block_classes = list(IMAGE2IMAGE_DENOISE_BLOCKS.values())
    block_names = list(IMAGE2IMAGE_DENOISE_BLOCKS.keys())

    @property
    def description(self) -> str:
        return "Core Ideogram4 image-to-image denoising workflow with strength-based latent initialization."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents", description="Unpatchified latents ready for the VAE decoder.")]


INPAINT_DENOISE_BLOCKS = InsertableDict(
    [
        ("text_inputs", Ideogram4TextInputsStep()),
        ("image_inputs", Ideogram4ImageInputsStep()),
        ("mask_inputs", Ideogram4MaskInputsStep()),
        ("prepare_latents", Ideogram4PrepareLatentsStep()),
        ("set_timesteps", Ideogram4SetTimestepsStep()),
        ("apply_strength", Ideogram4ApplyStrengthStep()),
        ("prepare_image_latents", Ideogram4PrepareLatentsWithStrengthStep()),
        ("prepare_mask_latents", Ideogram4PrepareMaskLatentsStep()),
        ("prepare_additional_inputs", Ideogram4PrepareAdditionalInputsStep()),
        ("denoise", Ideogram4InpaintDenoiseStep()),
        ("after_denoise", Ideogram4AfterDenoiseStep()),
    ]
)


# auto_docstring
class Ideogram4InpaintCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core Ideogram4 inpaint denoising workflow with latent-mask blending at every step.

      Components:
          transformer (`Ideogram4Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          unconditional_transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          text_features (`Tensor`):
              Per-prompt text features from the encoder.
          text_lengths (`list`):
              Per-prompt text-token counts from the encoder.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`):
              The binary mask tensor resized to the generation resolution.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.

      Outputs:
          latents (`Tensor`):
              Unpatchified latents ready for the VAE decoder.
    """

    model_name = "ideogram4"
    block_classes = list(INPAINT_DENOISE_BLOCKS.values())
    block_names = list(INPAINT_DENOISE_BLOCKS.keys())

    @property
    def description(self) -> str:
        return "Core Ideogram4 inpaint denoising workflow with latent-mask blending at every step."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents", description="Unpatchified latents ready for the VAE decoder.")]


# auto_docstring
class Ideogram4AutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Select the Ideogram4 text-to-image, image-to-image, or inpaint denoising workflow.

      Components:
          transformer (`Ideogram4Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          unconditional_transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          text_features (`Tensor`):
              Per-prompt text features from the encoder.
          text_lengths (`list`):
              Per-prompt text-token counts from the encoder.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The binary mask tensor resized to the generation resolution.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.

      Outputs:
          latents (`Tensor`):
              Unpatchified latents ready for the VAE decoder.
    """

    block_classes = [Ideogram4InpaintCoreDenoiseStep, Ideogram4Img2ImgCoreDenoiseStep, Ideogram4CoreDenoiseStep]
    block_names = ["inpaint", "img2img", "text2img"]
    block_trigger_inputs = ["processed_mask_image", "image_latents"]
    default_block_name = "text2img"

    def select_block(self, processed_mask_image=None, image_latents=None) -> str | None:
        if processed_mask_image is not None:
            return "inpaint"
        if image_latents is not None:
            return "img2img"
        return None

    @property
    def description(self) -> str:
        return "Select the Ideogram4 text-to-image, image-to-image, or inpaint denoising workflow."


# auto_docstring
class Ideogram4AutoDecodeStep(AutoPipelineBlocks):
    """
    Decode Ideogram4 latents and apply the optional cropped-inpaint overlay.

      Components:
          vae (`AutoencoderKLFlux2`) image_mask_processor (`InpaintProcessor`) image_processor (`VaeImageProcessor`)

      Inputs:
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          latents (`Tensor`):
              The unpatchified latents to decode.
          mask_overlay_kwargs (`dict`, *optional*):
              Arguments used to composite a cropped inpaint result over the source image.

      Outputs:
          images (`list`):
              Generated images.
    """

    block_classes = [Ideogram4InpaintDecodeStep, Ideogram4DecodeStep]
    block_names = ["inpaint", "default"]
    block_trigger_inputs = ["mask_overlay_kwargs", None]

    @property
    def description(self) -> str:
        return "Decode Ideogram4 latents and apply the optional cropped-inpaint overlay."


AUTO_BLOCKS = InsertableDict(
    [
        ("vae_encoder", Ideogram4AutoVaeEncoderStep()),
        ("prompt_upsample", Ideogram4PromptUpsampleStep()),
        ("text_encoder", Ideogram4TextEncoderStep()),
        ("denoise", Ideogram4AutoCoreDenoiseStep()),
        ("decode", Ideogram4AutoDecodeStep()),
    ]
)


# auto_docstring
class Ideogram4AutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for Ideogram4 text-to-image, image-to-image, and inpainting workflows.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `prompt`, `image`
        - `inpainting`: requires `prompt`, `image`, `mask_image`

      Components:
          image_processor (`VaeImageProcessor`) image_mask_processor (`InpaintProcessor`) vae (`AutoencoderKLFlux2`)
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. tokenizer (`Qwen2Tokenizer`): The tokenizer paired
          with the text encoder. prompt_enhancer_head (`Ideogram4PromptEnhancerHead`): LM head grafted onto the text
          encoder for prompt upsampling. transformer (`Ideogram4Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) unconditional_transformer (`Ideogram4Transformer2DModel`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          mask_image (`Image`, *optional*):
              Mask image for inpainting.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          prompt (`str`):
              The prompt or prompts to guide image generation.
          prompt_upsampling (`bool`, *optional*, defaults to False):
              If True, rewrite the prompt into Ideogram4's native JSON caption before encoding.
          prompt_upsampling_temperature (`float`, *optional*, defaults to 1.0):
              Sampling temperature for prompt upsampling.
          max_sequence_length (`int`, *optional*, defaults to 2048):
              Maximum sequence length for prompt encoding.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The binary mask tensor resized to the generation resolution.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`, *optional*, defaults to 48):
              The number of denoising steps.
          mu (`float`, *optional*, defaults to 0.0):
              Base mean of the logit-normal schedule.
          std (`float`, *optional*, defaults to 1.5):
              Std of the logit-normal schedule.
          guidance_schedule (`list`, *optional*, defaults to (7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
          7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0)):
              Per-step guidance scale schedule (length num_inference_steps).
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          mask_overlay_kwargs (`dict`, *optional*):
              Arguments used to composite a cropped inpaint result over the source image.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "ideogram4"
    block_classes = list(AUTO_BLOCKS.values())
    block_names = list(AUTO_BLOCKS.keys())

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"prompt": True, "image": True},
        "inpainting": {"prompt": True, "image": True, "mask_image": True},
    }

    @property
    def description(self) -> str:
        return "Auto Modular pipeline for Ideogram4 text-to-image, image-to-image, and inpainting workflows."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]
