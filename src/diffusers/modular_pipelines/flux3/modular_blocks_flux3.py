# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
    Flux3PrepareImageCondLatentsStep,
    Flux3PrepareLatentsStep,
    Flux3SetTimestepsStep,
    Flux3TextRoPEInputsStep,
)
from .decoders import Flux3DecodeStep, Flux3UnpackLatentsStep
from .denoise import Flux3DenoiseStep
from .encoders import Flux3ProcessImagesInputStep, Flux3TextEncoderStep, Flux3VaeEncoderStep
from .inputs import Flux3ImageCondInputStep, Flux3TextInputStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


###
### VAE encoder (reference images)
###


# auto_docstring
class Flux3VaeEncoderSequentialStep(SequentialPipelineBlocks):
    """
    VAE encoder step that preprocesses and encodes the reference images into their latent representations.

      Components:
          image_processor (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`)

      Inputs:
          image (`Image | list`):
              The reference image or images to condition the generation on; each keeps its own resolution
              (center-cropped to the 16-pixel grid)

      Outputs:
          condition_images (`list`):
              The preprocessed reference images, one tensor of shape (1, 3, height_i, width_i) in [-1, 1] per image
          image_cond_latents (`list`):
              Normalized packed latents of the reference images, one tensor of shape (1, 128, height_i//16,
              width_i//16) per image
    """

    model_name = "flux3"
    block_classes = [Flux3ProcessImagesInputStep(), Flux3VaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "VAE encoder step that preprocesses and encodes the reference images into their latent representations."


# auto_docstring
class Flux3AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the reference images into their latent representations.
       - `Flux3VaeEncoderSequentialStep` is used when `image` is provided (image-to-image).
       - If `image` is not provided, step will be skipped.

      Components:
          image_processor (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`)

      Inputs:
          image (`Image | list`, *optional*):
              The reference image or images to condition the generation on; each keeps its own resolution
              (center-cropped to the 16-pixel grid)

      Outputs:
          condition_images (`list`):
              The preprocessed reference images, one tensor of shape (1, 3, height_i, width_i) in [-1, 1] per image
          image_cond_latents (`list`):
              Normalized packed latents of the reference images, one tensor of shape (1, 128, height_i//16,
              width_i//16) per image
    """

    model_name = "flux3"
    block_classes = [Flux3VaeEncoderSequentialStep]
    block_names = ["image_conditioning"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the reference images into their latent representations.\n"
            " - `Flux3VaeEncoderSequentialStep` is used when `image` is provided (image-to-image).\n"
            " - If `image` is not provided, step will be skipped."
        )


###
### Core denoise
###

Flux3CoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux3TextInputStep()),
        ("prepare_latents", Flux3PrepareLatentsStep()),
        ("set_timesteps", Flux3SetTimestepsStep()),
        ("prepare_rope_inputs", Flux3TextRoPEInputsStep()),
        ("denoise", Flux3DenoiseStep()),
        ("after_denoise", Flux3UnpackLatentsStep()),
    ]
)


# auto_docstring
class Flux3CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for FLUX.3, for text-to-image generation.

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3"
    block_classes = Flux3CoreDenoiseBlocks.values()
    block_names = Flux3CoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Core denoise step that performs the denoising process for FLUX.3, for text-to-image generation."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


Flux3ImageConditionedCoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux3TextInputStep()),
        ("cond_input", Flux3ImageCondInputStep()),
        ("prepare_latents", Flux3PrepareLatentsStep()),
        ("prepare_image_cond_latents", Flux3PrepareImageCondLatentsStep()),
        ("set_timesteps", Flux3SetTimestepsStep()),
        ("prepare_rope_inputs", Flux3TextRoPEInputsStep()),
        ("denoise", Flux3DenoiseStep()),
        ("after_denoise", Flux3UnpackLatentsStep()),
    ]
)


# auto_docstring
class Flux3ImageConditionedCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for FLUX.3 with reference images (image-to-image).

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          image_cond_latents (`list`):
              Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16)
              per image
          height (`int`, *optional*):
              The height in pixels of the generated image; defaults to the first reference image's aspect ratio at the
              default area
          width (`int`, *optional*):
              The width in pixels of the generated image; defaults to the first reference image's aspect ratio at the
              default area
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3"
    block_classes = Flux3ImageConditionedCoreDenoiseBlocks.values()
    block_names = Flux3ImageConditionedCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step that performs the denoising process for FLUX.3 with reference images (image-to-image)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# auto_docstring
class Flux3AutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Core denoise step that selects the FLUX.3 denoising workflow by the conditioning inputs.
       - `Flux3ImageConditionedCoreDenoiseStep` is used when `image_cond_latents` is provided (image-to-image).
       - `Flux3CoreDenoiseStep` is used otherwise (text-to-image).

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          image_cond_latents (`list`, *optional*):
              Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16)
              per image
          height (`int`, *optional*, defaults to None or 1024, depending on the workflow):
              The height in pixels of the generated image; defaults to the first reference image's aspect ratio at the
              default area
          width (`int`, *optional*, defaults to None or 1024, depending on the workflow):
              The width in pixels of the generated image; defaults to the first reference image's aspect ratio at the
              default area
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
              Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3"
    block_classes = [Flux3ImageConditionedCoreDenoiseStep, Flux3CoreDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_cond_latents", None]

    @property
    def description(self):
        return (
            "Core denoise step that selects the FLUX.3 denoising workflow by the conditioning inputs.\n"
            " - `Flux3ImageConditionedCoreDenoiseStep` is used when `image_cond_latents` is provided (image-to-image).\n"
            " - `Flux3CoreDenoiseStep` is used otherwise (text-to-image)."
        )


###
### Auto blocks
###

AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux3TextEncoderStep()),
        ("vae_encoder", Flux3AutoVaeEncoderStep()),
        ("denoise", Flux3AutoCoreDenoiseStep()),
        ("decode", Flux3DecodeStep()),
    ]
)


# auto_docstring
class Flux3AutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks that perform text-to-image and image-to-image generation using FLUX.3.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image_conditioned`: requires `prompt`, `image`

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer (`AutoProcessor`) image_processor
          (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`) transformer (`Flux3Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt to guide the generation (one prompt per call)
          negative_prompt (`str`, *optional*):
              The prompt not to guide the generation
          max_sequence_length (`int`, *optional*, defaults to 8192):
              Maximum number of prompt tokens; longer prompts are truncated.
          text_encoder_out_layers (`tuple`, *optional*, defaults to (4, 8, 12, 16, 20, 24, 28, 32)):
              Indices of the text-encoder hidden layers whose states are stacked to form the text embeddings
          image (`Image | list`, *optional*):
              The reference image or images to condition the generation on; each keeps its own resolution
              (center-cropped to the 16-pixel grid)
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          image_cond_latents (`list`, *optional*):
              Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16)
              per image
          height (`int`, *optional*, defaults to None or 1024, depending on the workflow):
              The height in pixels of the generated image; defaults to the first reference image's aspect ratio at the
              default area
          width (`int`, *optional*, defaults to None or 1024, depending on the workflow):
              The width in pixels of the generated image; defaults to the first reference image's aspect ratio at the
              default area
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
              Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()
    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
    }

    @property
    def description(self):
        return "Auto blocks that perform text-to-image and image-to-image generation using FLUX.3."

    @property
    def outputs(self):
        return [
            OutputParam.template("images"),
            OutputParam.template("latents"),
        ]
