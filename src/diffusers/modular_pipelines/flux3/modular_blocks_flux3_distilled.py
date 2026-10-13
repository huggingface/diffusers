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
    Flux3DistilledTextRoPEInputsStep,
    Flux3PrepareImageCondLatentsStep,
    Flux3PrepareLatentsStep,
    Flux3SetTimestepsStep,
)
from .decoders import Flux3DecodeStep, Flux3UnpackLatentsStep
from .denoise import Flux3DistilledDenoiseStep
from .encoders import Flux3DistilledTextEncoderStep
from .inputs import Flux3DistilledTextInputStep, Flux3ImageCondInputStep
from .modular_blocks_flux3 import Flux3AutoVaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# The step-distilled (flow-map) FLUX.3 checkpoints: 16 steps and no classifier-free guidance. The text encoder,
# text input, text RoPE and denoise steps have distilled versions without the negative prompt; the denoise loop
# passes the transformer the timestep each step jumps to and runs a single pass. Everything else is shared with the
# base model.

###
### Core denoise
###

Flux3DistilledCoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux3DistilledTextInputStep()),
        ("prepare_latents", Flux3PrepareLatentsStep()),
        ("set_timesteps", Flux3SetTimestepsStep(num_inference_steps=16)),
        ("prepare_rope_inputs", Flux3DistilledTextRoPEInputsStep()),
        ("denoise", Flux3DistilledDenoiseStep()),
        ("after_denoise", Flux3UnpackLatentsStep()),
    ]
)


# auto_docstring
class Flux3DistilledCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for the step-distilled FLUX.3 model, for text-to-image
    generation.

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 16):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3-distilled"
    block_classes = Flux3DistilledCoreDenoiseBlocks.values()
    block_names = Flux3DistilledCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step that performs the denoising process for the step-distilled FLUX.3 model, for "
            "text-to-image generation."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


Flux3DistilledImageConditionedCoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux3DistilledTextInputStep()),
        ("cond_input", Flux3ImageCondInputStep()),
        ("prepare_latents", Flux3PrepareLatentsStep()),
        ("prepare_image_cond_latents", Flux3PrepareImageCondLatentsStep()),
        ("set_timesteps", Flux3SetTimestepsStep(num_inference_steps=16)),
        ("prepare_rope_inputs", Flux3DistilledTextRoPEInputsStep()),
        ("denoise", Flux3DistilledDenoiseStep()),
        ("after_denoise", Flux3UnpackLatentsStep()),
    ]
)


# auto_docstring
class Flux3DistilledImageConditionedCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for the step-distilled FLUX.3 model with reference images
    (image-to-image).

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
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
          num_inference_steps (`int`, *optional*, defaults to 16):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3-distilled"
    block_classes = Flux3DistilledImageConditionedCoreDenoiseBlocks.values()
    block_names = Flux3DistilledImageConditionedCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step that performs the denoising process for the step-distilled FLUX.3 model with reference "
            "images (image-to-image)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# auto_docstring
class Flux3DistilledAutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Core denoise step that selects the step-distilled FLUX.3 denoising workflow by the conditioning inputs.
       - `Flux3DistilledImageConditionedCoreDenoiseStep` is used when `image_cond_latents` is provided
         (image-to-image).
       - `Flux3DistilledCoreDenoiseStep` is used otherwise (text-to-image).

      Components:
          transformer (`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
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

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3-distilled"
    block_classes = [Flux3DistilledImageConditionedCoreDenoiseStep, Flux3DistilledCoreDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_cond_latents", None]

    @property
    def description(self):
        return (
            "Core denoise step that selects the step-distilled FLUX.3 denoising workflow by the conditioning inputs.\n"
            " - `Flux3DistilledImageConditionedCoreDenoiseStep` is used when `image_cond_latents` is provided "
            "(image-to-image).\n"
            " - `Flux3DistilledCoreDenoiseStep` is used otherwise (text-to-image)."
        )


###
### Auto blocks
###

DISTILLED_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", Flux3DistilledTextEncoderStep()),
        ("vae_encoder", Flux3AutoVaeEncoderStep()),
        ("denoise", Flux3DistilledAutoCoreDenoiseStep()),
        ("decode", Flux3DecodeStep()),
    ]
)


# auto_docstring
class Flux3DistilledAutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks that perform text-to-image and image-to-image generation using the step-distilled FLUX.3 model (16
    steps, no classifier-free guidance).

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image_conditioned`: requires `prompt`, `image`

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer (`AutoProcessor`) image_processor
          (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`) transformer (`Flux3Transformer2DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          prompt (`str`):
              The prompt to guide the generation (one prompt per call)
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
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux3-distilled"
    block_classes = DISTILLED_AUTO_BLOCKS.values()
    block_names = DISTILLED_AUTO_BLOCKS.keys()
    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
    }

    @property
    def description(self):
        return (
            "Auto blocks that perform text-to-image and image-to-image generation using the step-distilled FLUX.3 "
            "model (16 steps, no classifier-free guidance)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("images"),
            OutputParam.template("latents"),
        ]
