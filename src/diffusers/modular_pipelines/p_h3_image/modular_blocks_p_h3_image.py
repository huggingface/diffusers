# Copyright 2026 Pruna AI and The HuggingFace Team. All rights reserved.
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

import torch

from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InputParam, OutputParam
from .before_denoise import (
    PH3ImageEditPrepareLatentsStep,
    PH3ImageEditPrepareLayoutStep,
    PH3ImagePrepareConditionLatentsStep,
    PH3ImagePrepareLatentsStep,
    PH3ImagePrepareLayoutStep,
    PH3ImageSetTimestepsStep,
)
from .before_encoder import PH3ImageEditResizeStep
from .decoders import PH3ImageAfterDenoiseStep, PH3ImageDecodeStep
from .denoise import PH3ImageDenoiseStep
from .encoders import PH3ImageEditTextEncoderStep, PH3ImageEditVaeEncoderStep, PH3ImageTextEncoderStep
from .prompt_upsampler import PH3ImageAutoPromptUpsampleStep


# auto_docstring
class PH3ImageAutoBeforeEncodeStep(AutoPipelineBlocks):
    """
    Image preparation block.
       - `PH3ImageEditResizeStep` runs when `image` is provided (`image_conditioned`), resizing the image to edit onto
         the canvas.
       - a `text2image` request skips this block.

      Components:
          image_processor (`VaeImageProcessor`)

      Configs:
          reference_min_pixels (default: 1048576) reference_max_pixels (default: 8388608) canvas_min_pixels (default:
          4194304) vision_pixels (default: 1000000)

      Inputs:
          image (`Image | list`, *optional*, depends on the workflow):
              The image to edit, or a list of at most 8 images: the first is edited, the others are references.
          height (`int`, *optional*):
              Height of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for `text2image`,
              which works best at 8 megapixels. An edit defaults to the first image's size, at least
              `canvas_min_pixels`, and works best at 4 or 8 megapixels.
          width (`int`, *optional*):
              Width of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for `text2image`,
              which works best at 8 megapixels. An edit defaults to the first image's size, at least
              `canvas_min_pixels`, and works best at 4 or 8 megapixels.

      Outputs:
          height (`int`):
              Resolved height of the edited image in pixels.
          width (`int`):
              Resolved width of the edited image in pixels.
          reference_images (`list`):
              The images of the edit resized for the VAE.
          vision_images (`list`):
              The images of the edit resized for the text encoder.
    """

    model_name = "p-h3-image"
    block_classes = [PH3ImageEditResizeStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Image preparation block.\n"
            + " - `PH3ImageEditResizeStep` runs when `image` is provided (`image_conditioned`), resizing the image "
            "to edit onto the canvas.\n" + " - a `text2image` request skips this block."
        )


# auto_docstring
class PH3ImageAutoTextEncoderStep(AutoPipelineBlocks):
    """
    Text encoder block.
       - `PH3ImageEditTextEncoderStep` runs when `image` is provided (`image_conditioned`).
       - `PH3ImageTextEncoderStep` runs otherwise (`text2image`).

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) text_connector (`PH3ImageTextConnector`) tokenizer
          (`Qwen2Tokenizer`) processor (`Qwen3VLProcessor`)

      Inputs:
          prompt (`str`):
              The prompt, or the edit instruction when `image` is passed. A single string.
          vision_images (`list`, *optional*, depends on the workflow):
              The images of the edit resized for the text encoder, the image to edit first.

      Outputs:
          prompt_embeds (`Tensor`):
              The conditioning MiniMax-H3's transformer reads, of shape `(1, num_text_tokens, 5120)`.
          text_token_tags (`Tensor`):
              The modality tag of every row of `prompt_embeds`; vision tokens are tagged as video.
    """

    model_name = "p-h3-image"
    block_classes = [PH3ImageEditTextEncoderStep, PH3ImageTextEncoderStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self):
        return (
            "Text encoder block.\n"
            + " - `PH3ImageEditTextEncoderStep` runs when `image` is provided (`image_conditioned`).\n"
            + " - `PH3ImageTextEncoderStep` runs otherwise (`text2image`)."
        )


# auto_docstring
class PH3ImageAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder block.
       - `PH3ImageEditVaeEncoderStep` runs when `image` is provided (`image_conditioned`).
       - a `text2image` request skips this block.

      Components:
          vae (`AutoencoderKLPH3Image`)

      Inputs:
          reference_images (`list`, *optional*, depends on the workflow):
              The images of the edit resized for the VAE, the image to edit first.

      Outputs:
          condition_latents (`list`):
              The conditioning latents of the images of the edit, one `(1, latent_channels, 1, latent_height,
              latent_width)` tensor per image.
    """

    model_name = "p-h3-image"
    block_classes = [PH3ImageEditVaeEncoderStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder block.\n"
            + " - `PH3ImageEditVaeEncoderStep` runs when `image` is provided (`image_conditioned`).\n"
            + " - a `text2image` request skips this block."
        )


# auto_docstring
class PH3ImageCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for `text2image`: builds the packed layout, draws the noise, runs the denoising loop and
    unpacks the latents.

      Components:
          scheduler (`MiniMaxH3Scheduler`) audio_scheduler (`MiniMaxH3Scheduler`) transformer
          (`MiniMaxH3Transformer3DModel`)

      Configs:
          sample_sigmas (default: [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]):
          The checkpoint's sampling grid, used when `sigmas` is not passed.

      Inputs:
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`.
          height (`int`, *optional*, defaults to 2912):
              Height of the generated image in pixels, rounded to a multiple of 32. The model works best at 8
              megapixels.
          width (`int`, *optional*, defaults to 2912):
              Width of the generated image in pixels, rounded to a multiple of 32. The model works best at 8
              megapixels.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated image noise of shape `(1, latent_channels, 1, latent_height, latent_width)`, used instead
              of the draw.
          sigmas (`list`, *optional*):
              A custom sampling grid, strictly decreasing and ending at 0. Defaults to the checkpoint's
              `sample_sigmas`.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The generated latents, of shape `(1, latent_channels, 1, latent_height, latent_width)`.
    """

    model_name = "p-h3-image"
    block_classes = [
        PH3ImagePrepareLayoutStep,
        PH3ImagePrepareLatentsStep,
        PH3ImageSetTimestepsStep,
        PH3ImageDenoiseStep,
        PH3ImageAfterDenoiseStep,
    ]
    block_names = ["prepare_layout", "prepare_latents", "set_timesteps", "denoise", "after_denoise"]

    @property
    def description(self):
        return (
            "Core denoising workflow for `text2image`: builds the packed layout, draws the noise, runs the denoising "
            "loop and unpacks the latents."
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated latents, of shape `(1, latent_channels, 1, latent_height, latent_width)`.",
            ),
        ]


# auto_docstring
class PH3ImageEditCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for `image_conditioned`: as for `text2image`, with the conditioning latents of the image to
    edit packed in front of the generated rows.

      Components:
          scheduler (`MiniMaxH3Scheduler`) audio_scheduler (`MiniMaxH3Scheduler`) transformer
          (`MiniMaxH3Transformer3DModel`)

      Configs:
          sample_sigmas (default: [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]):
          The checkpoint's sampling grid, used when `sigmas` is not passed.

      Inputs:
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`.
          condition_latents (`list`):
              The conditioning latents of the images of the edit. Their shapes give the geometry of the reference
              blocks.
          height (`int`):
              Height of the edited image in pixels.
          width (`int`):
              Width of the edited image in pixels.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it first, one draw per condition,
              before the noise of the generated rows.
          latents (`Tensor`, *optional*):
              Pre-generated image noise of shape `(1, latent_channels, 1, latent_height, latent_width)`, used instead
              of the draw.
          sigmas (`list`, *optional*):
              A custom sampling grid, strictly decreasing and ending at 0. Defaults to the checkpoint's
              `sample_sigmas`.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The generated latents, of shape `(1, latent_channels, 1, latent_height, latent_width)`.
    """

    model_name = "p-h3-image"
    block_classes = [
        PH3ImageEditPrepareLayoutStep,
        PH3ImagePrepareConditionLatentsStep,
        PH3ImagePrepareLatentsStep,
        PH3ImageEditPrepareLatentsStep,
        PH3ImageSetTimestepsStep,
        PH3ImageDenoiseStep,
        PH3ImageAfterDenoiseStep,
    ]
    block_names = [
        "prepare_layout",
        "prepare_condition_latents",
        "prepare_latents",
        "prepare_latents_image_conditioned",
        "set_timesteps",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return (
            "Core denoising workflow for `image_conditioned`: as for `text2image`, with the conditioning latents of the "
            "image to edit packed in front of the generated rows."
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated latents, of shape `(1, latent_channels, 1, latent_height, latent_width)`.",
            ),
        ]


# auto_docstring
class PH3ImageAutoDenoiseStep(AutoPipelineBlocks):
    """
    Denoise block.
       - `PH3ImageEditCoreDenoiseStep` runs when `image` is provided (`image_conditioned`).
       - `PH3ImageCoreDenoiseStep` runs otherwise (`text2image`).

      Components:
          scheduler (`MiniMaxH3Scheduler`) audio_scheduler (`MiniMaxH3Scheduler`) transformer
          (`MiniMaxH3Transformer3DModel`)

      Configs:
          sample_sigmas (default: [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]):
          The checkpoint's sampling grid, used when `sigmas` is not passed.

      Inputs:
          text_token_tags (`Tensor`):
              The per-row modality tag of every row of `prompt_embeds`.
          condition_latents (`list`, *optional*, depends on the workflow):
              The conditioning latents of the images of the edit. Their shapes give the geometry of the reference
              blocks.
          height (`int`, *optional*, depends on the workflow):
              Height of the edited image in pixels.
          width (`int`, *optional*, depends on the workflow):
              Width of the edited image in pixels.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it first, one draw per condition,
              before the noise of the generated rows.
          latents (`Tensor`, *optional*):
              Pre-generated image noise of shape `(1, latent_channels, 1, latent_height, latent_width)`, used instead
              of the draw.
          sigmas (`list`, *optional*):
              A custom sampling grid, strictly decreasing and ending at 0. Defaults to the checkpoint's
              `sample_sigmas`.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The generated latents, of shape `(1, latent_channels, 1, latent_height, latent_width)`.
    """

    model_name = "p-h3-image"
    block_classes = [PH3ImageEditCoreDenoiseStep, PH3ImageCoreDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self):
        return (
            "Denoise block.\n"
            + " - `PH3ImageEditCoreDenoiseStep` runs when `image` is provided (`image_conditioned`).\n"
            + " - `PH3ImageCoreDenoiseStep` runs otherwise (`text2image`)."
        )


# auto_docstring
class PH3ImageBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for P-H3-Image: `text2image`, and instruction-based editing (`image_conditioned`) when
    `image` is passed. `prompt` goes through the prompt upsampler first; pop the `prompt_upsampler` step to render it
    as given.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image_conditioned`: requires `prompt`, `image`

      Components:
          image_processor (`VaeImageProcessor`) prompt_upsampler (`Qwen3_5ForConditionalGeneration`): The prompt
          upsampler. prompt_upsampler_processor (`AutoProcessor`): Its processor. text_encoder
          (`Qwen3VLForConditionalGeneration`) text_connector (`PH3ImageTextConnector`) tokenizer (`Qwen2Tokenizer`)
          processor (`Qwen3VLProcessor`) vae (`AutoencoderKLPH3Image`) scheduler (`MiniMaxH3Scheduler`) audio_scheduler
          (`MiniMaxH3Scheduler`) transformer (`MiniMaxH3Transformer3DModel`)

      Configs:
          reference_min_pixels (default: 1048576) reference_max_pixels (default: 8388608) canvas_min_pixels (default:
          4194304) vision_pixels (default: 1000000) sample_sigmas (default: [1.0, 0.984753, 0.967905, 0.951688,
          0.925372, 0.881923, 0.810083, 0.528718, 0.0]): The checkpoint's sampling grid, used when `sigmas` is not
          passed.

      Inputs:
          image (`Image | list`, *optional*, depends on the workflow):
              The image to edit, or a list of at most 8 images: the first is edited, the others are references.
          height (`int`, *optional*):
              Height of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for `text2image`,
              which works best at 8 megapixels. An edit defaults to the first image's size, at least
              `canvas_min_pixels`, and works best at 4 or 8 megapixels.
          width (`int`, *optional*):
              Width of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for `text2image`,
              which works best at 8 megapixels. An edit defaults to the first image's size, at least
              `canvas_min_pixels`, and works best at 4 or 8 megapixels.
          prompt (`str`):
              The prompt, or the edit instruction when `image` is passed. A single string.
          max_new_tokens (`int`, *optional*, depends on the workflow):
              The most tokens the upsampler may generate, its reasoning included.
          vision_images (`list`, *optional*, depends on the workflow):
              The images of the edit resized for the text encoder, the image to edit first.
          reference_images (`list`, *optional*, depends on the workflow):
              The images of the edit resized for the VAE, the image to edit first.
          condition_latents (`list`, *optional*, depends on the workflow):
              The conditioning latents of the images of the edit. Their shapes give the geometry of the reference
              blocks.
          generator (`Generator`, *optional*):
              The generator of the request. The conditioning noise is drawn from it first, one draw per condition,
              before the noise of the generated rows.
          latents (`Tensor`, *optional*):
              Pre-generated image noise of shape `(1, latent_channels, 1, latent_height, latent_width)`, used instead
              of the draw.
          sigmas (`list`, *optional*):
              A custom sampling grid, strictly decreasing and ending at 0. Defaults to the checkpoint's
              `sample_sigmas`.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the three row-index tensors.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "p-h3-image"
    block_classes = [
        PH3ImageAutoBeforeEncodeStep,
        PH3ImageAutoPromptUpsampleStep,
        PH3ImageAutoTextEncoderStep,
        PH3ImageAutoVaeEncoderStep,
        PH3ImageAutoDenoiseStep,
        PH3ImageDecodeStep,
    ]
    block_names = ["before_encode", "prompt_upsampler", "text_encoder", "vae_encoder", "denoise", "decode"]
    # `before_encode` reads `height`/`width` with no default (an edit derives them from the image) and the prompt
    # upsampler defaults them to 2912, so the pipeline declares them itself: optional, no default, each workflow keeps
    # its own
    input_overrides = [
        InputParam.template(
            "height",
            description=(
                "Height of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for "
                "`text2image`, which works best at 8 megapixels. An edit defaults to the first image's size, at "
                "least `canvas_min_pixels`, and works best at 4 or 8 megapixels."
            ),
        ),
        InputParam.template(
            "width",
            description=(
                "Width of the generated image in pixels, rounded to a multiple of 32. Defaults to 2912 for "
                "`text2image`, which works best at 8 megapixels. An edit defaults to the first image's size, at "
                "least `canvas_min_pixels`, and works best at 4 or 8 megapixels."
            ),
        ),
    ]
    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline blocks for P-H3-Image: `text2image`, and instruction-based editing (`image_conditioned`) "
            "when `image` is passed. `prompt` goes through the prompt upsampler first; pop the `prompt_upsampler` step "
            "to render it as given."
        )

    @property
    def outputs(self):
        return [OutputParam.template("images")]
