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
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    Flux2PrepareImageLatentsStep,
    Flux2PrepareLatentsStep,
    Flux2RoPEInputsStep,
    Flux2SetTimestepsStep,
)
from .decoders import Flux2DecodeStep, Flux2UnpackLatentsStep
from .denoise import Flux2KleinDenoiseStep
from .encoders import (
    Flux2KleinTextEncoderStep,
    Flux2VaeEncoderStep,
)
from .inputs import (
    Flux2ProcessImagesInputStep,
    Flux2TextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

################
# VAE encoder
################


# auto_docstring
class Flux2KleinVaeEncoderSequentialStep(SequentialPipelineBlocks):
    """
    VAE encoder step that preprocesses and encodes the image inputs into their latent representations.

      Components:
          image_processor (`Flux2ImageProcessor`) vae (`AutoencoderKLFlux2`)

      Inputs:
          image (`None`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          condition_images (`list`):
              TODO: Add description.
          image_latents (`list`):
              List of latent representations for each reference image
    """

    model_name = "flux2-klein"

    block_classes = [Flux2ProcessImagesInputStep(), Flux2VaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "VAE encoder step that preprocesses and encodes the image inputs into their latent representations."


# auto_docstring
class Flux2KleinAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the image inputs into their latent representations.
      This is an auto pipeline block that works for image conditioning tasks.
       - `Flux2KleinVaeEncoderSequentialStep` is used when `image` is provided.
       - If `image` is not provided, step will be skipped.

      Components:
          image_processor (`Flux2ImageProcessor`) vae (`AutoencoderKLFlux2`)

      Inputs:
          image (`None`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          condition_images (`list`):
              TODO: Add description.
          image_latents (`list`):
              List of latent representations for each reference image
    """

    model_name = "flux2-klein"

    block_classes = [Flux2KleinVaeEncoderSequentialStep]
    block_names = ["img_conditioning"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the image inputs into their latent representations.\n"
            "This is an auto pipeline block that works for image conditioning tasks.\n"
            " - `Flux2KleinVaeEncoderSequentialStep` is used when `image` is provided.\n"
            " - If `image` is not provided, step will be skipped."
        )


###
### Core denoise
###

Flux2KleinCoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux2TextInputStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2KleinDenoiseStep()),
        ("after_denoise", Flux2UnpackLatentsStep()),
    ]
)


# auto_docstring
class Flux2KleinCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for Flux2-Klein (distilled model), for text-to-image
    generation.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`Flux2Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          joint_attention_kwargs (`None`, *optional*):
              TODO: Add description.
          image_latents (`Tensor`, *optional*):
              Packed image latents for conditioning. Shape: (B, img_seq_len, C)
          image_latent_ids (`Tensor`, *optional*):
              Position IDs for image latents. Shape: (B, img_seq_len, 4)

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux2-klein"

    block_classes = Flux2KleinCoreDenoiseBlocks.values()
    block_names = Flux2KleinCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Core denoise step that performs the denoising process for Flux2-Klein (distilled model), for text-to-image generation."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


Flux2KleinImageConditionedCoreDenoiseBlocks = InsertableDict(
    [
        ("input", Flux2TextInputStep()),
        ("prepare_image_latents", Flux2PrepareImageLatentsStep()),
        ("prepare_latents", Flux2PrepareLatentsStep()),
        ("set_timesteps", Flux2SetTimestepsStep()),
        ("prepare_rope_inputs", Flux2RoPEInputsStep()),
        ("denoise", Flux2KleinDenoiseStep()),
        ("after_denoise", Flux2UnpackLatentsStep()),
    ]
)


# auto_docstring
class Flux2KleinImageConditionedCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that performs the denoising process for Flux2-Klein (distilled model) with image conditioning.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`Flux2Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          image_latents (`list`, *optional*):
              TODO: Add description.
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          joint_attention_kwargs (`None`, *optional*):
              TODO: Add description.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux2-klein"

    block_classes = Flux2KleinImageConditionedCoreDenoiseBlocks.values()
    block_names = Flux2KleinImageConditionedCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Core denoise step that performs the denoising process for Flux2-Klein (distilled model) with image conditioning."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# auto_docstring
class Flux2KleinAutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Auto core denoise step that performs the denoising process for Flux2-Klein.
      This is an auto pipeline block that works for text-to-image and image-conditioned generation.
       - `Flux2KleinCoreDenoiseStep` is used for text-to-image generation.
       - `Flux2KleinImageConditionedCoreDenoiseStep` is used for image-conditioned generation.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`Flux2Transformer2DModel`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          image_latents (`list`, *optional*):
              TODO: Add description.
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`):
              TODO: Add description.
          timesteps (`None`):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          joint_attention_kwargs (`None`, *optional*):
              TODO: Add description.
          image_latent_ids (`Tensor`, *optional*):
              Position IDs for image latents. Shape: (B, img_seq_len, 4)

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux2-klein"
    block_classes = [Flux2KleinImageConditionedCoreDenoiseStep, Flux2KleinCoreDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Auto core denoise step that performs the denoising process for Flux2-Klein.\n"
            "This is an auto pipeline block that works for text-to-image and image-conditioned generation.\n"
            " - `Flux2KleinCoreDenoiseStep` is used for text-to-image generation.\n"
            " - `Flux2KleinImageConditionedCoreDenoiseStep` is used for image-conditioned generation.\n"
        )


###
### Auto blocks
###


# auto_docstring
class Flux2KleinAutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks that perform the text-to-image and image-conditioned generation using Flux2-Klein.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image_conditioned`: requires `image`, `prompt`

      Components:
          text_encoder (`Qwen3ForCausalLM`) tokenizer (`Qwen2TokenizerFast`) image_processor (`Flux2ImageProcessor`)
          vae (`AutoencoderKLFlux2`) scheduler (`FlowMatchEulerDiscreteScheduler`) transformer
          (`Flux2Transformer2DModel`)

      Configs:
          is_distilled (default: True)

      Inputs:
          prompt (`None`, *optional*):
              TODO: Add description.
          max_sequence_length (`int`, *optional*, defaults to 512):
              TODO: Add description.
          text_encoder_out_layers (`tuple`, *optional*, defaults to (9, 18, 27)):
              TODO: Add description.
          image (`None`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          image_latents (`list`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`):
              TODO: Add description.
          num_inference_steps (`None`):
              TODO: Add description.
          timesteps (`None`):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          joint_attention_kwargs (`None`, *optional*):
              TODO: Add description.
          image_latent_ids (`Tensor`, *optional*):
              Position IDs for image latents. Shape: (B, img_seq_len, 4)
          output_type (`None`, *optional*, defaults to pil):
              TODO: Add description.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "flux2-klein"
    block_classes = [
        Flux2KleinTextEncoderStep(),
        Flux2KleinAutoVaeEncoderStep(),
        Flux2KleinAutoCoreDenoiseStep(),
        Flux2DecodeStep(),
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"image": True, "prompt": True},
    }

    @property
    def description(self):
        return "Auto blocks that perform the text-to-image and image-conditioned generation using Flux2-Klein."

    @property
    def outputs(self):
        return [
            OutputParam.template("images"),
        ]
