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
    FluxImg2ImgPrepareLatentsStep,
    FluxImg2ImgSetTimestepsStep,
    FluxPrepareLatentsStep,
    FluxRoPEInputsStep,
    FluxSetTimestepsStep,
)
from .decoders import FluxDecodeStep
from .denoise import FluxDenoiseStep
from .encoders import (
    FluxProcessImagesInputStep,
    FluxTextEncoderStep,
    FluxVaeEncoderStep,
)
from .inputs import (
    FluxAdditionalInputsStep,
    FluxTextInputStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# vae encoder (run before before_denoise)


# auto_docstring
class FluxImg2ImgVaeEncoderStep(SequentialPipelineBlocks):
    """
    Vae encoder step that preprocess andencode the image inputs into their latent representations.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`)

      Inputs:
          resized_image (`None`, *optional*):
              TODO: Add description.
          image (`None`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          processed_image (`None`):
              TODO: Add description.
          image_latents (`Tensor`):
              The latents representing the reference image
    """

    model_name = "flux"

    block_classes = [FluxProcessImagesInputStep(), FluxVaeEncoderStep()]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Vae encoder step that preprocess andencode the image inputs into their latent representations."


# auto_docstring
class FluxAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    Vae encoder step that encode the image inputs into their latent representations.
      This is an auto pipeline block that works for img2img tasks.
       - `FluxImg2ImgVaeEncoderStep` (img2img) is used when only `image` is provided. - if `image` is not provided,
         step will be skipped.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`)

      Inputs:
          resized_image (`None`, *optional*):
              TODO: Add description.
          image (`None`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.

      Outputs:
          processed_image (`None`):
              TODO: Add description.
          image_latents (`Tensor`):
              The latents representing the reference image
    """

    model_name = "flux"
    block_classes = [FluxImg2ImgVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for img2img tasks.\n"
            + " - `FluxImg2ImgVaeEncoderStep` (img2img) is used when only `image` is provided."
            + " - if `image` is not provided, step will be skipped."
        )


# before_denoise: text2img
# auto_docstring
class FluxBeforeDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepares the inputs for the denoise step in text-to-image generation.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`.
              Can be generated in input step.
          dtype (`dtype`, *optional*):
              The dtype of the model inputs
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          guidance_scale (`None`, *optional*, defaults to 3.5):
              TODO: Add description.
          prompt_embeds (`None`, *optional*):
              TODO: Add description.

      Outputs:
          latents (`Tensor`):
              The initial latents to use for the denoising process
          timesteps (`Tensor`):
              The timesteps to use for inference
          num_inference_steps (`int`):
              The number of denoising steps to perform at inference time
          guidance (`Tensor`):
              Optional guidance to be used.
          txt_ids (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation.
          img_ids (`list`):
              The sequence lengths of the image latents, used for RoPE calculation.
    """

    model_name = "flux"
    block_classes = [FluxPrepareLatentsStep(), FluxSetTimestepsStep(), FluxRoPEInputsStep()]
    block_names = ["prepare_latents", "set_timesteps", "prepare_rope_inputs"]

    @property
    def description(self):
        return "Before denoise step that prepares the inputs for the denoise step in text-to-image generation."


# before_denoise: img2img
# auto_docstring
class FluxImg2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    """
    Before denoise step that prepare the inputs for the denoise step for img2img task.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`.
              Can be generated in input step.
          dtype (`dtype`, *optional*):
              The dtype of the model inputs
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          strength (`None`, *optional*, defaults to 0.6):
              TODO: Add description.
          guidance_scale (`None`, *optional*, defaults to 3.5):
              TODO: Add description.
          image_latents (`Tensor`):
              The image latents to use for the denoising process. Can be generated in vae encoder and packed in input
              step.
          prompt_embeds (`None`, *optional*):
              TODO: Add description.

      Outputs:
          latents (`Tensor`):
              The initial latents to use for the denoising process
          timesteps (`Tensor`):
              The timesteps to use for inference
          num_inference_steps (`int`):
              The number of denoising steps to perform at inference time
          guidance (`Tensor`):
              Optional guidance to be used.
          initial_noise (`Tensor`):
              The initial random noised used for inpainting denoising.
          txt_ids (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation.
          img_ids (`list`):
              The sequence lengths of the image latents, used for RoPE calculation.
    """

    model_name = "flux"
    block_classes = [
        FluxPrepareLatentsStep(),
        FluxImg2ImgSetTimestepsStep(),
        FluxImg2ImgPrepareLatentsStep(),
        FluxRoPEInputsStep(),
    ]
    block_names = ["prepare_latents", "set_timesteps", "prepare_img2img_latents", "prepare_rope_inputs"]

    @property
    def description(self):
        return "Before denoise step that prepare the inputs for the denoise step for img2img task."


# before_denoise: all task (text2img, img2img)
# auto_docstring
class FluxAutoBeforeDenoiseStep(AutoPipelineBlocks):
    """
    Before denoise step that prepare the inputs for the denoise step.
      This is an auto pipeline block that works for text2image.
       - `FluxBeforeDenoiseStep` (text2image) is used.
       - `FluxImg2ImgBeforeDenoiseStep` (img2img) is used when only `image_latents` is provided.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          height (`int`):
              TODO: Add description.
          width (`int`):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be `batch_size * num_images_per_prompt`.
              Can be generated in input step.
          dtype (`dtype`, *optional*):
              The dtype of the model inputs
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          strength (`None`, *optional*, defaults to 0.6):
              TODO: Add description.
          guidance_scale (`None`, *optional*, defaults to 3.5):
              TODO: Add description.
          image_latents (`Tensor`, *optional*):
              The image latents to use for the denoising process. Can be generated in vae encoder and packed in input
              step.
          prompt_embeds (`None`, *optional*):
              TODO: Add description.

      Outputs:
          latents (`Tensor`):
              The initial latents to use for the denoising process
          timesteps (`Tensor`):
              The timesteps to use for inference
          num_inference_steps (`int`):
              The number of denoising steps to perform at inference time
          guidance (`Tensor`):
              Optional guidance to be used.
          initial_noise (`Tensor`):
              The initial random noised used for inpainting denoising.
          txt_ids (`list`):
              The sequence lengths of the prompt embeds, used for RoPE calculation.
          img_ids (`list`):
              The sequence lengths of the image latents, used for RoPE calculation.
    """

    model_name = "flux"
    block_classes = [FluxImg2ImgBeforeDenoiseStep, FluxBeforeDenoiseStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2image.\n"
            + " - `FluxBeforeDenoiseStep` (text2image) is used.\n"
            + " - `FluxImg2ImgBeforeDenoiseStep` (img2img) is used when only `image_latents` is provided.\n"
        )


# inputs: text2image/img2img


# auto_docstring
class FluxImg2ImgInputStep(SequentialPipelineBlocks):
    """
    Input step that prepares the inputs for the img2img denoising step. It:

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated pooled text embeddings. Can be generated from text_encoder step.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          image_latents (`None`, *optional*):
              TODO: Add description.

      Outputs:
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt
          dtype (`dtype`):
              Data type of model tensor inputs (determined by `prompt_embeds`)
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation
          pooled_prompt_embeds (`Tensor`):
              pooled text embeddings used to guide the image generation
          image_height (`int`):
              The height of the image latents
          image_width (`int`):
              The width of the image latents
    """

    model_name = "flux"
    block_classes = [FluxTextInputStep(), FluxAdditionalInputsStep()]
    block_names = ["text_inputs", "additional_inputs"]

    @property
    def description(self):
        return "Input step that prepares the inputs for the img2img denoising step. It:\n"
        " - make sure the text embeddings have consistent batch size as well as the additional inputs (`image_latents`).\n"
        " - update height/width based `image_latents`, patchify `image_latents`."


# auto_docstring
class FluxAutoInputStep(AutoPipelineBlocks):
    """
    Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size,
    and patchified.
       This is an auto pipeline block that works for text2image/img2img tasks.
       - `FluxImg2ImgInputStep` (img2img) is used when `image_latents` is provided.
       - `FluxTextInputStep` (text2image) is used when `image_latents` are not provided.

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated pooled text embeddings. Can be generated from text_encoder step.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          image_latents (`None`, *optional*):
              TODO: Add description.

      Outputs:
          batch_size (`int`):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt
          dtype (`dtype`):
              Data type of model tensor inputs (determined by `prompt_embeds`)
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation
          pooled_prompt_embeds (`Tensor`):
              pooled text embeddings used to guide the image generation
          image_height (`int`):
              The height of the image latents
          image_width (`int`):
              The width of the image latents
    """

    model_name = "flux"

    block_classes = [FluxImg2ImgInputStep, FluxTextInputStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Input step that standardize the inputs for the denoising step, e.g. make sure inputs have consistent batch size, and patchified. \n"
            " This is an auto pipeline block that works for text2image/img2img tasks.\n"
            + " - `FluxImg2ImgInputStep` (img2img) is used when `image_latents` is provided.\n"
            + " - `FluxTextInputStep` (text2image) is used when `image_latents` are not provided.\n"
        )


# auto_docstring
class FluxCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core step that performs the denoising process for Flux.
      This step supports text-to-image and image-to-image tasks for Flux:
       - for image-to-image generation, you need to provide `image_latents`
       - for text-to-image generation, all you need to provide is prompt embeddings.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`FluxTransformer2DModel`)

      Inputs:
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          pooled_prompt_embeds (`Tensor`, *optional*):
              Pre-generated pooled text embeddings. Can be generated from text_encoder step.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          image_latents (`None`, *optional*):
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
          strength (`None`, *optional*, defaults to 0.6):
              TODO: Add description.
          guidance_scale (`None`, *optional*, defaults to 3.5):
              TODO: Add description.
          joint_attention_kwargs (`None`, *optional*):
              TODO: Add description.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "flux"
    block_classes = [FluxAutoInputStep, FluxAutoBeforeDenoiseStep, FluxDenoiseStep]
    block_names = ["input", "before_denoise", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process for Flux.\n"
            + "This step supports text-to-image and image-to-image tasks for Flux:\n"
            + " - for image-to-image generation, you need to provide `image_latents`\n"
            + " - for text-to-image generation, all you need to provide is prompt embeddings."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
        ]


# Auto blocks (text2image and img2img)
AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", FluxTextEncoderStep()),
        ("vae_encoder", FluxAutoVaeEncoderStep()),
        ("denoise", FluxCoreDenoiseStep()),
        ("decode", FluxDecodeStep()),
    ]
)


# auto_docstring
class FluxAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image and image-to-image using Flux.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `image`, `prompt`

      Components:
          text_encoder (`CLIPTextModel`) tokenizer (`CLIPTokenizer`) text_encoder_2 (`T5EncoderModel`) tokenizer_2
          (`T5TokenizerFast`) image_processor (`VaeImageProcessor`) vae (`AutoencoderKL`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) transformer (`FluxTransformer2DModel`)

      Inputs:
          prompt (`None`, *optional*):
              TODO: Add description.
          prompt_2 (`None`, *optional*):
              TODO: Add description.
          max_sequence_length (`int`, *optional*, defaults to 512):
              TODO: Add description.
          joint_attention_kwargs (`None`, *optional*):
              TODO: Add description.
          resized_image (`None`, *optional*):
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
          image_latents (`None`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          strength (`None`, *optional*, defaults to 0.6):
              TODO: Add description.
          guidance_scale (`None`, *optional*, defaults to 3.5):
              TODO: Add description.
          output_type (`None`, *optional*, defaults to pil):
              TODO: Add description.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "flux"

    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"image": True, "prompt": True},
    }

    @property
    def description(self):
        return "Auto Modular pipeline for text-to-image and image-to-image using Flux."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
