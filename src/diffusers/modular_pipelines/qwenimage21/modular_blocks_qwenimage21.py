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

from ..modular_pipeline import ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    QwenImage21PrepareInpaintStep,
    QwenImage21PrepareLatentsStep,
    QwenImage21SetTimestepsStep,
    QwenImage21TextInputsStep,
)
from .decoders import QwenImage21DecodeStep
from .denoise import QwenImage21DenoiseStep, QwenImage21InpaintDenoiseStep
from .encoders import QwenImage21InpaintVaeEncoderStep, QwenImage21TextEncoderStep, QwenImage21VaeEncoderStep
from .inputs import QwenImage21ProcessImagesStep, QwenImage21ProcessInpaintStep


QwenImage21CoreDenoiseBlocks = InsertableDict(
    [
        ("input", QwenImage21TextInputsStep()),
        ("prepare_latents", QwenImage21PrepareLatentsStep()),
        ("set_timesteps", QwenImage21SetTimestepsStep()),
        ("denoise", QwenImage21DenoiseStep()),
    ]
)


# auto_docstring
class QwenImage21CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Prepare and denoise target latents from reusable text and image embeddings.

      Components:
          transformer (`QwenImage21Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`, *optional*):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          image_pad_mask (`Tensor`):
              Positive prompt vision positions.
          negative_image_pad_mask (`Tensor`, *optional*):
              Negative prompt vision positions.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          output_resolution (`int`, *optional*, defaults to 1024):
              Default output side length.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          condition_latents (`Tensor`, *optional*):
              Packed condition tokens.
          condition_shapes (`list`, *optional*):
              Spatial shape of each condition image.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21CoreDenoiseBlocks.values()
    block_names = QwenImage21CoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Prepare and denoise target latents from reusable text and image embeddings."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


QwenImage21InpaintCoreDenoiseBlocks = InsertableDict(
    [
        ("input", QwenImage21TextInputsStep()),
        ("prepare_latents", QwenImage21PrepareLatentsStep()),
        ("set_timesteps", QwenImage21SetTimestepsStep()),
        ("prepare_inpaint", QwenImage21PrepareInpaintStep()),
        ("denoise", QwenImage21InpaintDenoiseStep()),
    ]
)


# auto_docstring
class QwenImage21InpaintCoreDenoiseStep(QwenImage21CoreDenoiseStep):
    """
    Prepare and denoise a masked source with optional reference-image conditioning.

      Components:
          transformer (`QwenImage21Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              negative text embeddings used to guide the image generation. Can be generated from text_encoder step.
          prompt_embeds_mask (`Tensor`, *optional*):
              mask for the text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              mask for the negative text embeddings. Can be generated from text_encoder step.
          image_pad_mask (`Tensor`):
              Positive prompt vision positions.
          negative_image_pad_mask (`Tensor`, *optional*):
              Negative prompt vision positions.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          output_resolution (`int`, *optional*, defaults to 1024):
              Default output side length.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          condition_latents (`Tensor`, *optional*):
              Packed condition tokens.
          condition_shapes (`list`, *optional*):
              Spatial shape of each condition image.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          source_latents (`Tensor`):
              Encoded source tokens.
          processed_mask (`Tensor`):
              Binary repaint mask.
          strength (`float`, *optional*, defaults to 1.0):
              Strength for img2img/inpainting.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    block_classes = QwenImage21InpaintCoreDenoiseBlocks.values()
    block_names = QwenImage21InpaintCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Prepare and denoise a masked source with optional reference-image conditioning."


QwenImage21Text2ImageBlocks = InsertableDict(
    [
        ("text_encoder", QwenImage21TextEncoderStep()),
        ("denoise", QwenImage21CoreDenoiseStep()),
        ("decode", QwenImage21DecodeStep()),
    ]
)


# auto_docstring
class QwenImage21Text2ImageStep(SequentialPipelineBlocks):
    """
    Generate RGBA images from text with Qwen-Image 2.1.

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) processor (`Qwen3VLProcessor`) guider
          (`ClassifierFreeGuidance`) transformer (`QwenImage21Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) vae (`AutoencoderKLQwenImage21`) image_processor (`VaeImageProcessor`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          condition_images (`list`, *optional*):
              Resized images to encode with each prompt.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          output_resolution (`int`, *optional*, defaults to 1024):
              Default output side length.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          condition_latents (`Tensor`, *optional*):
              Packed condition tokens.
          condition_shapes (`list`, *optional*):
              Spatial shape of each condition image.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
          image_pad_mask (`Tensor`):
              Vision token positions in each positive prompt.
          negative_image_pad_mask (`Tensor`):
              Vision token positions in each negative prompt.
          latents (`Tensor`):
              Initial target noise.
          height (`int`):
              Output height in pixels.
          width (`int`):
              Output width in pixels.
          condition_latents (`Tensor`):
              Condition tokens expanded to the output batch.
          img_shapes (`list`):
              Condition and target grid shapes for rotary embeddings.
          img_mask (`Tensor`):
              Joint positive prompt and target vision positions.
          negative_img_mask (`Tensor`):
              Joint negative prompt and target vision positions.
          timesteps (`Tensor`):
              Denoising timesteps.
          num_inference_steps (`int`):
              Number of denoising timesteps.
          noise_pred (`Tensor`):
              Guided target flow prediction.
          kv_caches (`dict`):
              Temporary KV caches, cleared after generation.
          cache_enabled (`bool`):
              Whether causal-condition caching is active.
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21Text2ImageBlocks.values()
    block_names = QwenImage21Text2ImageBlocks.keys()

    @property
    def description(self):
        return "Generate RGBA images from text with Qwen-Image 2.1."


QwenImage21ImageConditionedBlocks = InsertableDict(
    [
        ("preprocess", QwenImage21ProcessImagesStep()),
        ("text_encoder", QwenImage21TextEncoderStep()),
        ("vae_encoder", QwenImage21VaeEncoderStep()),
        ("denoise", QwenImage21CoreDenoiseStep()),
        ("decode", QwenImage21DecodeStep()),
    ]
)


# auto_docstring
class QwenImage21ImageConditionedStep(SequentialPipelineBlocks):
    """
    Generate images conditioned on one or more reference images.

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) processor
          (`Qwen3VLProcessor`) guider (`ClassifierFreeGuidance`) vae (`AutoencoderKLQwenImage21`) transformer
          (`QwenImage21Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to resize condition images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          condition_images (`list`):
              Resized RGBA images for joint vision and text encoding.
          vae_images (`list`):
              Normalized RGBA condition tensors.
          height (`int`):
              Output height in pixels.
          width (`int`):
              Output width in pixels.
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
          image_pad_mask (`Tensor`):
              Vision token positions in each positive prompt.
          negative_image_pad_mask (`Tensor`):
              Vision token positions in each negative prompt.
          condition_latents (`Tensor`):
              Concatenated condition image tokens.
          condition_shapes (`list`):
              Frame, height and width for each condition latent grid.
          latents (`Tensor`):
              Initial target noise.
          img_shapes (`list`):
              Condition and target grid shapes for rotary embeddings.
          img_mask (`Tensor`):
              Joint positive prompt and target vision positions.
          negative_img_mask (`Tensor`):
              Joint negative prompt and target vision positions.
          timesteps (`Tensor`):
              Denoising timesteps.
          num_inference_steps (`int`):
              Number of denoising timesteps.
          noise_pred (`Tensor`):
              Guided target flow prediction.
          kv_caches (`dict`):
              Temporary KV caches, cleared after generation.
          cache_enabled (`bool`):
              Whether causal-condition caching is active.
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21ImageConditionedBlocks.values()
    block_names = QwenImage21ImageConditionedBlocks.keys()

    @property
    def description(self):
        return "Generate images conditioned on one or more reference images."


QwenImage21InpaintBlocks = InsertableDict(
    [
        ("preprocess", QwenImage21ProcessInpaintStep()),
        ("text_encoder", QwenImage21TextEncoderStep()),
        ("vae_encoder", QwenImage21VaeEncoderStep()),
        ("source_encoder", QwenImage21InpaintVaeEncoderStep()),
        ("denoise", QwenImage21InpaintCoreDenoiseStep()),
        ("decode", QwenImage21DecodeStep()),
    ]
)


# auto_docstring
class QwenImage21InpaintStep(SequentialPipelineBlocks):
    """
    Repaint one source image using a mask and optional additional references.

      Components:
          image_processor (`VaeImageProcessor`) mask_processor (`VaeImageProcessor`) text_encoder
          (`Qwen3VLForConditionalGeneration`) processor (`Qwen3VLProcessor`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKLQwenImage21`) transformer (`QwenImage21Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to resize condition images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          mask_image (`Image`):
              Mask image for inpainting.
          reference_images (`list`, *optional*):
              Additional PIL or numpy reference images, shared by every prompt. The source is always the first
              condition image.
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          strength (`float`, *optional*, defaults to 1.0):
              Strength for img2img/inpainting.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          condition_images (`list`):
              Resized RGBA images for joint vision and text encoding.
          vae_images (`list`):
              Normalized RGBA condition tensors.
          height (`int`):
              Output height in pixels.
          width (`int`):
              Output width in pixels.
          source_image (`Tensor`):
              Normalized source RGBA image at the output resolution.
          processed_mask (`Tensor`):
              Binary repaint mask at the output resolution; white is repainted.
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
          image_pad_mask (`Tensor`):
              Vision token positions in each positive prompt.
          negative_image_pad_mask (`Tensor`):
              Vision token positions in each negative prompt.
          condition_latents (`Tensor`):
              Concatenated condition image tokens.
          condition_shapes (`list`):
              Frame, height and width for each condition latent grid.
          source_latents (`Tensor`):
              Source tokens used to preserve the unmasked area.
          latents (`Tensor`):
              Initial target noise.
          img_shapes (`list`):
              Condition and target grid shapes for rotary embeddings.
          img_mask (`Tensor`):
              Joint positive prompt and target vision positions.
          negative_img_mask (`Tensor`):
              Joint negative prompt and target vision positions.
          timesteps (`Tensor`):
              Denoising timesteps.
          num_inference_steps (`int`):
              Number of denoising timesteps.
          initial_noise (`Tensor`):
              Noise reused when preserving the source.
          mask (`Tensor`):
              Repaint weights for target latent tokens.
          noise_pred (`Tensor`):
              Guided target flow prediction.
          kv_caches (`dict`):
              Temporary KV caches, cleared after generation.
          cache_enabled (`bool`):
              Whether causal-condition caching is active.
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage21"
    block_classes = QwenImage21InpaintBlocks.values()
    block_names = QwenImage21InpaintBlocks.keys()

    @property
    def description(self):
        return "Repaint one source image using a mask and optional additional references."


# auto_docstring
class QwenImage21AutoBlocks(ConditionalPipelineBlocks):
    """
    Qwen-Image 2.1 text-to-image, image-conditioned generation, and inpainting with optional references.

      Components:
          image_processor (`VaeImageProcessor`) mask_processor (`VaeImageProcessor`) text_encoder
          (`Qwen3VLForConditionalGeneration`) processor (`Qwen3VLProcessor`) guider (`ClassifierFreeGuidance`) vae
          (`AutoencoderKLQwenImage21`) transformer (`QwenImage21Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          output_resolution (`int`, *optional*, defaults to 1024):
              Target side length used to resize condition images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          mask_image (`Image`, *optional*):
              Mask image for inpainting.
          reference_images (`list`, *optional*):
              Additional PIL or numpy reference images, shared by every prompt. The source is always the first
              condition image.
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          num_inference_steps (`int`, *optional*, defaults to 40):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          strength (`float`, *optional*, defaults to 1.0):
              Strength for img2img/inpainting.
          use_kv_cache (`bool`, *optional*, defaults to True):
              Cache step-independent text and condition-image keys and values.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          condition_images (`list`, *optional*):
              Resized images to encode with each prompt.
          condition_latents (`Tensor`, *optional*):
              Packed condition tokens.
          condition_shapes (`list`, *optional*):
              Spatial shape of each condition image.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "qwenimage21"
    block_classes = [QwenImage21InpaintStep, QwenImage21ImageConditionedStep, QwenImage21Text2ImageStep]
    block_names = ["inpainting", "image_conditioned", "text2image"]
    block_trigger_inputs = ["mask_image", "image", "reference_images"]
    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
        "inpainting": {"prompt": True, "image": True, "mask_image": True},
    }

    def select_block(self, mask_image=None, image=None, reference_images=None):
        if mask_image is not None:
            if image is None:
                raise ValueError("`mask_image` requires a source `image`.")
            return "inpainting"
        if reference_images is not None:
            raise ValueError(
                "`reference_images` requires inpainting with `image` and `mask_image`. For unmasked generation, pass the condition images as `image`."
            )
        return "image_conditioned" if image is not None else "text2image"

    @property
    def available_workflows(self):
        return list(self._workflow_map)

    def get_workflow(self, workflow_name: str):
        if workflow_name not in self._workflow_map:
            raise ValueError(f"Unknown workflow {workflow_name!r}. Available workflows: {self.available_workflows}")
        return self.get_execution_blocks(**self._workflow_map[workflow_name])

    @property
    def description(self):
        return "Qwen-Image 2.1 text-to-image, image-conditioned generation, and inpainting with optional references."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
