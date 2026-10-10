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


import numpy as np
import torch

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import TripoSplatImageInputStep, TripoSplatPrepareLatentsStep, TripoSplatSetTimestepsStep
from .decoders import TripoSplatGaussianDecodeStep
from .denoise import TripoSplatDenoiseStep
from .encoders import TripoSplatImageEncoderStep, TripoSplatImagePreprocessStep, TripoSplatVaeEncoderStep


TripoSplatDenoiseBlocks = InsertableDict(
    [
        ("inputs", TripoSplatImageInputStep()),
        ("timesteps", TripoSplatSetTimestepsStep()),
        ("prepare", TripoSplatPrepareLatentsStep()),
        ("loop", TripoSplatDenoiseStep()),
    ]
)


# auto_docstring
class TripoSplatCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Prepare noise and denoise with image conditioning.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`TripoSplatTransformer3DModel`) guider
          (`TripoSplatClassifierFreeGuidance`)

      Inputs:
          encoder_hidden_states (`Tensor`):
              DINOv3 image features.
          image_latents (`Tensor`):
              Packed VAE image latents.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 20):
              The number of denoising steps.
          generator (`Generator | list`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          camera_latents (`Tensor`, *optional*):
              Initial camera noise, with shape (batch, 1, 5).

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          camera_latents (`Tensor`):
              Denoised camera latents.
    """

    model_name = "triposplat"
    block_classes = TripoSplatDenoiseBlocks.values()
    block_names = TripoSplatDenoiseBlocks.keys()

    @property
    def description(self):
        return "Prepare noise and denoise with image conditioning."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("camera_latents", type_hint=torch.Tensor, description="Denoised camera latents."),
        ]


TripoSplatBlocks = InsertableDict(
    [
        ("preprocess", TripoSplatImagePreprocessStep()),
        ("image_encoder", TripoSplatImageEncoderStep()),
        ("vae_encoder", TripoSplatVaeEncoderStep()),
        ("denoise", TripoSplatCoreDenoiseStep()),
        ("decode", TripoSplatGaussianDecodeStep()),
    ]
)


# auto_docstring
class TripoSplatAutoBlocks(SequentialPipelineBlocks):
    """
    Generate Gaussian splats from foreground images.

      Components:
          image_processor (`TripoSplatImageProcessor`) background_remover (`BiRefNetModel`) image_encoder
          (`DINOv3ViTModel`) vae (`AutoencoderKLFlux2`) scheduler (`FlowMatchEulerDiscreteScheduler`) transformer
          (`TripoSplatTransformer3DModel`) guider (`TripoSplatClassifierFreeGuidance`) decoder
          (`TripoSplatGaussianDecoder`)

      Configs:
          num_prefix_tokens (default: 5)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          erode_radius (`int`, *optional*, defaults to 1):
              Radius of the foreground mask erosion.
          is_preprocessed (`bool`, *optional*, defaults to False):
              Whether inputs are already cropped RGB images composited on black.
          generator (`Generator | list`, *optional*):
              Torch generator for deterministic generation.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 20):
              The number of denoising steps.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          camera_latents (`Tensor`, *optional*):
              Initial camera noise, with shape (batch, 1, 5).
          num_gaussians (`int | list`, *optional*, defaults to 262144):
              Gaussian counts between 32768 and 262144.
          decoder_generator (`Generator | list`, *optional*):
              Random generator or per-sample generators for octree sampling.
          output_type (`str`, *optional*, defaults to pt):
              Output format: 'pt' or 'np'.

      Outputs:
          gaussians (`Tensor | ndarray | list`):
              Decoded Gaussian parameters.
          latents (`Tensor`):
              Denoised latents.
          camera_latents (`Tensor`):
              Denoised camera latents.
          preprocessed_images (`list`):
              Prepared RGB images.
    """

    model_name = "triposplat"
    block_classes = TripoSplatBlocks.values()
    block_names = TripoSplatBlocks.keys()

    @property
    def description(self):
        return "Generate Gaussian splats from foreground images."

    @property
    def outputs(self):
        return [
            OutputParam(
                "gaussians", type_hint=torch.Tensor | np.ndarray | list, description="Decoded Gaussian parameters."
            ),
            OutputParam.template("latents"),
            OutputParam("camera_latents", type_hint=torch.Tensor, description="Denoised camera latents."),
            OutputParam("preprocessed_images", type_hint=list, description="Prepared RGB images."),
        ]
