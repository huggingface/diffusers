# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    MiniMaxH3ImageEditPrepareLatentsStep,
    MiniMaxH3ImageEditPrepareLayoutStep,
    MiniMaxH3ImagePrepareCanvasStep,
    MiniMaxH3ImagePrepareLatentsStep,
    MiniMaxH3ImagePrepareLayoutStep,
    MiniMaxH3ImageSetTimestepsStep,
)
from .before_encoder import MiniMaxH3ImageEditResizeStep
from .decoders import MiniMaxH3ImageAfterDenoiseStep, MiniMaxH3ImageDecodeStep
from .denoise import MiniMaxH3ImageDenoiseStep
from .encoders import (
    MiniMaxH3ImageEditTextEncoderStep,
    MiniMaxH3ImageEditVaeEncoderStep,
    MiniMaxH3ImageTextEncoderStep,
)


# auto_docstring
class MiniMaxH3ImageAutoBeforeEncodeStep(AutoPipelineBlocks):
    """
    Image preparation block.
       - `MiniMaxH3ImageEditResizeStep` runs when `image` is provided (`image_conditioned`), resizing the reference
         images for the VAE and for the text encoder.
       - a `text2image` request skips this block.

      Components:
          image_processor (`VaeImageProcessor`)

      Configs:
          reference_min_short_edge (default: 1024) reference_max_short_edge (default: 2048) vision_min_short_edge
          (default: 1024) vision_max_short_edge (default: 1024)

      Inputs:
          image (`Image | list`, *optional*):
              The reference image of the edit, or a list of them. The prompt refers to them in order as `<Picture 1>`,
              `<Picture 2>`, ...

      Outputs:
          reference_images (`list`):
              The reference images resized for the VAE.
          vision_images (`list`):
              The reference images resized for the text encoder.
    """

    model_name = "minimax-h3-image"
    block_classes = [MiniMaxH3ImageEditResizeStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "Image preparation block.\n"
            + " - `MiniMaxH3ImageEditResizeStep` runs when `image` is provided (`image_conditioned`), resizing the "
            "reference images for the VAE and for the text encoder.\n" + " - a `text2image` request skips this block."
        )


# auto_docstring
class MiniMaxH3ImageAutoTextEncoderStep(AutoPipelineBlocks):
    """
    Text encoder block.
       - `MiniMaxH3ImageEditTextEncoderStep` runs when `image` is provided (`image_conditioned`), putting a label and a
         vision block per reference image in front of the prompt.
       - `MiniMaxH3ImageTextEncoderStep` runs otherwise (`text2image`).

      Components:
          text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer (`Qwen2Tokenizer`) processor (`Qwen3VLProcessor`)

      Inputs:
          prompt (`str`):
              The prompt to guide generation, a single string.
          vision_images (`list`, *optional*):
              The reference images resized for the text encoder.

      Outputs:
          prompt_embeds (`Tensor`):
              The hidden state the transformer conditions on, of shape `(1, num_text_tokens, 5120)`, read after the
              50th decoder layer of the Qwen3-VL conditioner.
          text_token_tags (`Tensor`):
              The modality tag of every row of `prompt_embeds`; a vision block is tagged as image.
    """

    model_name = "minimax-h3-image"
    block_classes = [MiniMaxH3ImageEditTextEncoderStep, MiniMaxH3ImageTextEncoderStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self):
        return (
            "Text encoder block.\n"
            + " - `MiniMaxH3ImageEditTextEncoderStep` runs when `image` is provided (`image_conditioned`), putting a "
            "label and a vision block per reference image in front of the prompt.\n"
            + " - `MiniMaxH3ImageTextEncoderStep` runs otherwise (`text2image`)."
        )


# auto_docstring
class MiniMaxH3ImageAutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder block.
       - `MiniMaxH3ImageEditVaeEncoderStep` runs when `image` is provided (`image_conditioned`), encoding the reference
         images.
       - a `text2image` request skips this block.

      Components:
          vae (`AutoencoderKLMiniMaxH3Image`)

      Inputs:
          reference_images (`list`, *optional*):
              The reference images resized for the VAE.
          generator (`Generator`, *optional*):
              The generator of the request. One posterior sample per reference is drawn from it.

      Outputs:
          reference_latents (`list`):
              The conditioning latents of the reference images, one `(1, latent_channels, latent_height, latent_width)`
              tensor per reference, each at the reference's own size.
    """

    model_name = "minimax-h3-image"
    block_classes = [MiniMaxH3ImageEditVaeEncoderStep]
    block_names = ["image_conditioned"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder block.\n"
            + " - `MiniMaxH3ImageEditVaeEncoderStep` runs when `image` is provided (`image_conditioned`), encoding "
            "the reference images.\n" + " - a `text2image` request skips this block."
        )


# auto_docstring
class MiniMaxH3ImageCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for `text2image`: resolves the canvas, builds the packed layout, draws the noise, runs the
    denoising loop and unpacks the latents.

      Components:
          scheduler (`MiniMaxH3Scheduler`) transformer (`MiniMaxH3ImageTransformer2DModel`)

      Configs:
          sample_sigmas (default: [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724,
          0.23200887441635132, 0.0]): The sampling grid the checkpoint's per-step output heads were distilled on.

      Inputs:
          height (`int`, *optional*, defaults to 2048):
              Height of the returned image in pixels.
          width (`int`, *optional*, defaults to 2048):
              Width of the returned image in pixels.
          min_generation_area (`int`, *optional*, defaults to 4194304):
              The smallest area, in pixels, an image is generated at. A smaller `height x width` is generated on a
              canvas scaled up to this area and resized back to `height x width` after decoding. Pass 1024 to always
              generate at the requested size.
          text_token_tags (`Tensor`):
              The modality tag of every row of `prompt_embeds`.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noise of shape `(1, latent_channels, latent_height, latent_width)`, used instead of the
              draw.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the two row-index tensors.

      Outputs:
          latents (`Tensor`):
              The generated latents, of shape `(1, latent_channels, latent_height, latent_width)`.
    """

    model_name = "minimax-h3-image"
    block_classes = [
        MiniMaxH3ImagePrepareCanvasStep,
        MiniMaxH3ImagePrepareLayoutStep,
        MiniMaxH3ImagePrepareLatentsStep,
        MiniMaxH3ImageSetTimestepsStep,
        MiniMaxH3ImageDenoiseStep,
        MiniMaxH3ImageAfterDenoiseStep,
    ]
    block_names = ["prepare_canvas", "prepare_layout", "prepare_latents", "set_timesteps", "denoise", "after_denoise"]

    @property
    def description(self):
        return (
            "Core denoising workflow for `text2image`: resolves the canvas, builds the packed layout, draws the noise, "
            "runs the denoising loop and unpacks the latents."
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated latents, of shape `(1, latent_channels, latent_height, latent_width)`.",
            ),
        ]


# auto_docstring
class MiniMaxH3ImageEditCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for `image_conditioned`: resolves the canvas, builds the packed layout, draws the noise,
    puts the noised reference rows in front of the generated rows, runs the denoising loop and unpacks the latents.

      Components:
          scheduler (`MiniMaxH3Scheduler`) transformer (`MiniMaxH3ImageTransformer2DModel`)

      Configs:
          sample_sigmas (default: [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724,
          0.23200887441635132, 0.0]): The sampling grid the checkpoint's per-step output heads were distilled on.

      Inputs:
          height (`int`, *optional*, defaults to 2048):
              Height of the returned image in pixels.
          width (`int`, *optional*, defaults to 2048):
              Width of the returned image in pixels.
          min_generation_area (`int`, *optional*, defaults to 4194304):
              The smallest area, in pixels, an image is generated at. A smaller `height x width` is generated on a
              canvas scaled up to this area and resized back to `height x width` after decoding. Pass 1024 to always
              generate at the requested size.
          text_token_tags (`Tensor`):
              The modality tag of every row of `prompt_embeds`.
          reference_latents (`list`):
              The conditioning latents of the reference images. Their shape is where every reference's geometry comes
              from.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noise of shape `(1, latent_channels, latent_height, latent_width)`, used instead of the
              draw.
          reference_noise (`list`, *optional*):
              Pre-generated noise for the reference images, one tensor of the shape of each entry of
              `reference_latents`, used instead of the draws.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the two row-index tensors.

      Outputs:
          latents (`Tensor`):
              The generated latents, of shape `(1, latent_channels, latent_height, latent_width)`.
    """

    model_name = "minimax-h3-image"
    block_classes = [
        MiniMaxH3ImagePrepareCanvasStep,
        MiniMaxH3ImageEditPrepareLayoutStep,
        MiniMaxH3ImagePrepareLatentsStep,
        MiniMaxH3ImageEditPrepareLatentsStep,
        MiniMaxH3ImageSetTimestepsStep,
        MiniMaxH3ImageDenoiseStep,
        MiniMaxH3ImageAfterDenoiseStep,
    ]
    block_names = [
        "prepare_canvas",
        "prepare_layout",
        "prepare_latents",
        "prepare_condition_latents",
        "set_timesteps",
        "denoise",
        "after_denoise",
    ]

    @property
    def description(self):
        return (
            "Core denoising workflow for `image_conditioned`: resolves the canvas, builds the packed layout, draws "
            "the noise, puts the noised reference rows in front of the generated rows, runs the denoising loop and "
            "unpacks the latents."
        )

    @property
    def outputs(self):
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The generated latents, of shape `(1, latent_channels, latent_height, latent_width)`.",
            ),
        ]


# auto_docstring
class MiniMaxH3ImageAutoDenoiseStep(AutoPipelineBlocks):
    """
    Denoise block.
       - `MiniMaxH3ImageEditCoreDenoiseStep` runs when `image` is provided (`image_conditioned`).
       - `MiniMaxH3ImageCoreDenoiseStep` runs otherwise (`text2image`).

      Components:
          scheduler (`MiniMaxH3Scheduler`) transformer (`MiniMaxH3ImageTransformer2DModel`)

      Configs:
          sample_sigmas (default: [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724,
          0.23200887441635132, 0.0]): The sampling grid the checkpoint's per-step output heads were distilled on.

      Inputs:
          height (`int`):
              Height of the returned image in pixels.
          width (`int`):
              Width of the returned image in pixels.
          min_generation_area (`int`, *optional*, defaults to 4194304):
              The smallest area, in pixels, an image is generated at. A smaller `height x width` is generated on a
              canvas scaled up to this area and resized back to `height x width` after decoding. Pass 1024 to always
              generate at the requested size.
          text_token_tags (`Tensor`):
              The modality tag of every row of `prompt_embeds`.
          reference_latents (`list`, *optional*):
              The conditioning latents of the reference images. Their shape is where every reference's geometry comes
              from.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`):
              Pre-generated noise of shape `(1, latent_channels, latent_height, latent_width)`, used instead of the
              draw.
          reference_noise (`list`, *optional*):
              Pre-generated noise for the reference images, one tensor of the shape of each entry of
              `reference_latents`, used instead of the draws.
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the two row-index tensors.

      Outputs:
          latents (`Tensor`):
              The generated latents, of shape `(1, latent_channels, latent_height, latent_width)`.
    """

    model_name = "minimax-h3-image"
    block_classes = [MiniMaxH3ImageEditCoreDenoiseStep, MiniMaxH3ImageCoreDenoiseStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self):
        return (
            "Denoise block.\n"
            + " - `MiniMaxH3ImageEditCoreDenoiseStep` runs when `image` is provided (`image_conditioned`).\n"
            + " - `MiniMaxH3ImageCoreDenoiseStep` runs otherwise (`text2image`)."
        )


# auto_docstring
class MiniMaxH3ImageBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for MiniMax-H3-Image: `text2image`, and reference-image editing (`image_conditioned`) when
    `image` is passed.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image_conditioned`: requires `prompt`, `image`

      Components:
          image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer
          (`Qwen2Tokenizer`) processor (`Qwen3VLProcessor`) vae (`AutoencoderKLMiniMaxH3Image`) scheduler
          (`MiniMaxH3Scheduler`) transformer (`MiniMaxH3ImageTransformer2DModel`)

      Configs:
          reference_min_short_edge (default: 1024) reference_max_short_edge (default: 2048) vision_min_short_edge
          (default: 1024) vision_max_short_edge (default: 1024) sample_sigmas (default: [1.0, 0.9675236940383911,
          0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]): The sampling grid the checkpoint's
          per-step output heads were distilled on.

      Inputs:
          image (`Image | list`, *optional*):
              The reference image of the edit, or a list of them. The prompt refers to them in order as `<Picture 1>`,
              `<Picture 2>`, ...
          prompt (`str`):
              The prompt to guide generation, a single string.
          vision_images (`list`, *optional*):
              The reference images resized for the text encoder.
          reference_images (`list`, *optional*):
              The reference images resized for the VAE.
          generator (`Generator`, *optional*):
              The generator of the request. One posterior sample per reference is drawn from it.
          height (`int`):
              Height of the returned image in pixels.
          width (`int`):
              Width of the returned image in pixels.
          min_generation_area (`int`, *optional*, defaults to 4194304):
              The smallest area, in pixels, an image is generated at. A smaller `height x width` is generated on a
              canvas scaled up to this area and resized back to `height x width` after decoding. Pass 1024 to always
              generate at the requested size.
          reference_latents (`list`, *optional*):
              The conditioning latents of the reference images. Their shape is where every reference's geometry comes
              from.
          latents (`Tensor`):
              Pre-generated noise of shape `(1, latent_channels, latent_height, latent_width)`, used instead of the
              draw.
          reference_noise (`list`, *optional*):
              Pre-generated noise for the reference images, one tensor of the shape of each entry of
              `reference_latents`, used instead of the draws.
          **denoiser_input_fields (`None`, *optional*):
              The structural description of the packed sequence the transformer reads by name: `token_tags`,
              `position_ids` and the two row-index tensors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "minimax-h3-image"
    block_classes = [
        MiniMaxH3ImageAutoBeforeEncodeStep,
        MiniMaxH3ImageAutoTextEncoderStep,
        MiniMaxH3ImageAutoVaeEncoderStep,
        MiniMaxH3ImageAutoDenoiseStep,
        MiniMaxH3ImageDecodeStep,
    ]
    block_names = ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2image": {"prompt": True},
        "image_conditioned": {"prompt": True, "image": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline blocks for MiniMax-H3-Image: `text2image`, and reference-image editing "
            "(`image_conditioned`) when `image` is passed."
        )

    @property
    def outputs(self):
        return [OutputParam.template("images")]
