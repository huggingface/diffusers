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

from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch

from ...image_processor import PipelineImageInput
from ...loaders import ModularIPAdapterMixin, StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin
from ...pipelines.pipeline_utils import StableDiffusionMixin
from ...pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from ...utils import logging
from ..modular_pipeline import ModularPipeline
from ..modular_pipeline_utils import InputParam, OutputParam


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# YiYi TODO: move to a different file? stable_diffusion_xl_module should have its own folder?
# YiYi Notes: model specific components:
## (1) it should inherit from ModularPipeline
## (2) acts like a container that holds components and configs
## (3) define default config (related to components), e.g. default_sample_size, vae_scale_factor, num_channels_unet, num_channels_latents
## (4) inherit from model-specic loader class (e.g. StableDiffusionXLLoraLoaderMixin)
## (5) how to use together with Components_manager?
class StableDiffusionXLModularPipeline(
    ModularPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
    ModularIPAdapterMixin,
):
    """
    A ModularPipeline for Stable Diffusion XL.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "StableDiffusionXLAutoBlocks"

    @property
    def default_height(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_width(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_sample_size(self):
        default_sample_size = 128
        if hasattr(self, "unet") and self.unet is not None:
            default_sample_size = self.unet.config.sample_size
        return default_sample_size

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if hasattr(self, "vae") and self.vae is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    # YiYi TODO: change to num_channels_latents
    @property
    def num_channels_unet(self):
        num_channels_unet = 4
        if hasattr(self, "unet") and self.unet is not None:
            num_channels_unet = self.unet.config.in_channels
        return num_channels_unet

    @property
    def num_channels_latents(self):
        num_channels_latents = 4
        if hasattr(self, "vae") and self.vae is not None:
            num_channels_latents = self.vae.config.latent_channels
        return num_channels_latents


# YiYi/Sayak TODO: not used yet, maintain a list of schema that can be used across all pipeline blocks
# auto_docstring
SDXL_INPUTS_SCHEMA = {
    "prompt": InputParam(
        "prompt", type_hint=Union[str, List[str]], description="The prompt or prompts to guide the image generation"
    ),
    "prompt_2": InputParam(
        "prompt_2",
        type_hint=Union[str, List[str]],
        description="The prompt or prompts to be sent to the tokenizer_2 and text_encoder_2",
    ),
    "negative_prompt": InputParam(
        "negative_prompt",
        type_hint=Union[str, List[str]],
        description="The prompt or prompts not to guide the image generation",
    ),
    "negative_prompt_2": InputParam(
        "negative_prompt_2",
        type_hint=Union[str, List[str]],
        description="The negative prompt or prompts for text_encoder_2",
    ),
    "cross_attention_kwargs": InputParam(
        "cross_attention_kwargs",
        type_hint=Optional[dict],
        description="Kwargs dictionary passed to the AttentionProcessor",
    ),
    "clip_skip": InputParam(
        "clip_skip", type_hint=Optional[int], description="Number of layers to skip in CLIP text encoder"
    ),
    "image": InputParam(
        "image",
        type_hint=PipelineImageInput,
        required=True,
        description="The image(s) to modify for img2img or inpainting",
    ),
    "mask_image": InputParam(
        "mask_image",
        type_hint=PipelineImageInput,
        required=True,
        description="Mask image for inpainting, white pixels will be repainted",
    ),
    "generator": InputParam(
        "generator",
        type_hint=Optional[Union[torch.Generator, List[torch.Generator]]],
        description="Generator(s) for deterministic generation",
    ),
    "height": InputParam("height", type_hint=Optional[int], description="Height in pixels of the generated image"),
    "width": InputParam("width", type_hint=Optional[int], description="Width in pixels of the generated image"),
    "num_images_per_prompt": InputParam(
        "num_images_per_prompt", type_hint=int, default=1, description="Number of images to generate per prompt"
    ),
    "num_inference_steps": InputParam(
        "num_inference_steps", type_hint=int, default=50, description="Number of denoising steps"
    ),
    "timesteps": InputParam(
        "timesteps", type_hint=Optional[torch.Tensor], description="Custom timesteps for the denoising process"
    ),
    "sigmas": InputParam(
        "sigmas", type_hint=Optional[torch.Tensor], description="Custom sigmas for the denoising process"
    ),
    "denoising_end": InputParam(
        "denoising_end",
        type_hint=Optional[float],
        description="Fraction of denoising process to complete before termination",
    ),
    # YiYi Notes: img2img defaults to 0.3, inpainting defaults to 0.9999
    "strength": InputParam(
        "strength", type_hint=float, default=0.3, description="How much to transform the reference image"
    ),
    "denoising_start": InputParam(
        "denoising_start", type_hint=Optional[float], description="Starting point of the denoising process"
    ),
    "latents": InputParam(
        "latents", type_hint=Optional[torch.Tensor], description="Pre-generated noisy latents for image generation"
    ),
    "padding_mask_crop": InputParam(
        "padding_mask_crop",
        type_hint=Optional[Tuple[int, int]],
        description="Size of margin in crop for image and mask",
    ),
    "original_size": InputParam(
        "original_size",
        type_hint=Optional[Tuple[int, int]],
        description="Original size of the image for SDXL's micro-conditioning",
    ),
    "target_size": InputParam(
        "target_size", type_hint=Optional[Tuple[int, int]], description="Target size for SDXL's micro-conditioning"
    ),
    "negative_original_size": InputParam(
        "negative_original_size",
        type_hint=Optional[Tuple[int, int]],
        description="Negative conditioning based on image resolution",
    ),
    "negative_target_size": InputParam(
        "negative_target_size",
        type_hint=Optional[Tuple[int, int]],
        description="Negative conditioning based on target resolution",
    ),
    "crops_coords_top_left": InputParam(
        "crops_coords_top_left",
        type_hint=Tuple[int, int],
        default=(0, 0),
        description="Top-left coordinates for SDXL's micro-conditioning",
    ),
    "negative_crops_coords_top_left": InputParam(
        "negative_crops_coords_top_left",
        type_hint=Tuple[int, int],
        default=(0, 0),
        description="Negative conditioning crop coordinates",
    ),
    "aesthetic_score": InputParam(
        "aesthetic_score", type_hint=float, default=6.0, description="Simulates aesthetic score of generated image"
    ),
    "negative_aesthetic_score": InputParam(
        "negative_aesthetic_score", type_hint=float, default=2.0, description="Simulates negative aesthetic score"
    ),
    "eta": InputParam("eta", type_hint=float, default=0.0, description="Parameter η in the DDIM paper"),
    "output_type": InputParam(
        "output_type", type_hint=str, default="pil", description="Output format (pil/tensor/np.array)"
    ),
    "ip_adapter_image": InputParam(
        "ip_adapter_image",
        type_hint=PipelineImageInput,
        required=True,
        description="Image(s) to be used as IP adapter",
    ),
    "control_image": InputParam(
        "control_image", type_hint=PipelineImageInput, required=True, description="ControlNet input condition"
    ),
    "control_guidance_start": InputParam(
        "control_guidance_start",
        type_hint=Union[float, List[float]],
        default=0.0,
        description="When ControlNet starts applying",
    ),
    "control_guidance_end": InputParam(
        "control_guidance_end",
        type_hint=Union[float, List[float]],
        default=1.0,
        description="When ControlNet stops applying",
    ),
    "controlnet_conditioning_scale": InputParam(
        "controlnet_conditioning_scale",
        type_hint=Union[float, List[float]],
        default=1.0,
        description="Scale factor for ControlNet outputs",
    ),
    "guess_mode": InputParam(
        "guess_mode",
        type_hint=bool,
        default=False,
        description="Enables ControlNet encoder to recognize input without prompts",
    ),
    "control_mode": InputParam(
        "control_mode", type_hint=List[int], required=True, description="Control mode for union controlnet"
    ),
    "prompt_embeds": InputParam(
        "prompt_embeds",
        type_hint=torch.Tensor,
        required=True,
        description="Text embeddings used to guide image generation",
    ),
    "negative_prompt_embeds": InputParam(
        "negative_prompt_embeds", type_hint=torch.Tensor, description="Negative text embeddings"
    ),
    "pooled_prompt_embeds": InputParam(
        "pooled_prompt_embeds", type_hint=torch.Tensor, required=True, description="Pooled text embeddings"
    ),
    "negative_pooled_prompt_embeds": InputParam(
        "negative_pooled_prompt_embeds", type_hint=torch.Tensor, description="Negative pooled text embeddings"
    ),
    "batch_size": InputParam("batch_size", type_hint=int, required=True, description="Number of prompts"),
    "dtype": InputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
    "preprocess_kwargs": InputParam(
        "preprocess_kwargs", type_hint=Optional[dict], description="Kwargs for ImageProcessor"
    ),
    "latent_timestep": InputParam(
        "latent_timestep", type_hint=torch.Tensor, required=True, description="Initial noise level timestep"
    ),
    "image_latents": InputParam(
        "image_latents", type_hint=torch.Tensor, required=True, description="Latents representing reference image"
    ),
    "mask": InputParam("mask", type_hint=torch.Tensor, required=True, description="Mask for inpainting"),
    "masked_image_latents": InputParam(
        "masked_image_latents", type_hint=torch.Tensor, description="Masked image latents for inpainting"
    ),
    "add_time_ids": InputParam(
        "add_time_ids", type_hint=torch.Tensor, required=True, description="Time ids for conditioning"
    ),
    "negative_add_time_ids": InputParam(
        "negative_add_time_ids", type_hint=torch.Tensor, description="Negative time ids"
    ),
    "timestep_cond": InputParam("timestep_cond", type_hint=torch.Tensor, description="Timestep conditioning for LCM"),
    "noise": InputParam("noise", type_hint=torch.Tensor, description="Noise added to image latents"),
    "crops_coords": InputParam("crops_coords", type_hint=Optional[Tuple[int]], description="Crop coordinates"),
    "ip_adapter_embeds": InputParam(
        "ip_adapter_embeds", type_hint=List[torch.Tensor], description="Image embeddings for IP-Adapter"
    ),
    "negative_ip_adapter_embeds": InputParam(
        "negative_ip_adapter_embeds",
        type_hint=List[torch.Tensor],
        description="Negative image embeddings for IP-Adapter",
    ),
    "images": InputParam(
        "images",
        type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]],
        required=True,
        description="Generated images",
    ),
}


SDXL_INTERMEDIATE_OUTPUTS_SCHEMA = {
    "prompt_embeds": OutputParam(
        "prompt_embeds", type_hint=torch.Tensor, description="Text embeddings used to guide image generation"
    ),
    "negative_prompt_embeds": OutputParam(
        "negative_prompt_embeds", type_hint=torch.Tensor, description="Negative text embeddings"
    ),
    "pooled_prompt_embeds": OutputParam(
        "pooled_prompt_embeds", type_hint=torch.Tensor, description="Pooled text embeddings"
    ),
    "negative_pooled_prompt_embeds": OutputParam(
        "negative_pooled_prompt_embeds", type_hint=torch.Tensor, description="Negative pooled text embeddings"
    ),
    "batch_size": OutputParam("batch_size", type_hint=int, description="Number of prompts"),
    "dtype": OutputParam("dtype", type_hint=torch.dtype, description="Data type of model tensor inputs"),
    "image_latents": OutputParam(
        "image_latents", type_hint=torch.Tensor, description="Latents representing reference image"
    ),
    "mask": OutputParam("mask", type_hint=torch.Tensor, description="Mask for inpainting"),
    "masked_image_latents": OutputParam(
        "masked_image_latents", type_hint=torch.Tensor, description="Masked image latents for inpainting"
    ),
    "crops_coords": OutputParam("crops_coords", type_hint=Optional[Tuple[int]], description="Crop coordinates"),
    "timesteps": OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps for inference"),
    "num_inference_steps": OutputParam("num_inference_steps", type_hint=int, description="Number of denoising steps"),
    "latent_timestep": OutputParam(
        "latent_timestep", type_hint=torch.Tensor, description="Initial noise level timestep"
    ),
    "add_time_ids": OutputParam("add_time_ids", type_hint=torch.Tensor, description="Time ids for conditioning"),
    "negative_add_time_ids": OutputParam(
        "negative_add_time_ids", type_hint=torch.Tensor, description="Negative time ids"
    ),
    "timestep_cond": OutputParam("timestep_cond", type_hint=torch.Tensor, description="Timestep conditioning for LCM"),
    "latents": OutputParam("latents", type_hint=torch.Tensor, description="Denoised latents"),
    "noise": OutputParam("noise", type_hint=torch.Tensor, description="Noise added to image latents"),
    "ip_adapter_embeds": OutputParam(
        "ip_adapter_embeds", type_hint=List[torch.Tensor], description="Image embeddings for IP-Adapter"
    ),
    "negative_ip_adapter_embeds": OutputParam(
        "negative_ip_adapter_embeds",
        type_hint=List[torch.Tensor],
        description="Negative image embeddings for IP-Adapter",
    ),
    "images": OutputParam(
        "images",
        type_hint=Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]],
        description="Generated images",
    ),
}


SDXL_OUTPUTS_SCHEMA = {
    "images": OutputParam(
        "images",
        type_hint=Union[
            Tuple[Union[List[PIL.Image.Image], List[torch.Tensor], List[np.array]]], StableDiffusionXLPipelineOutput
        ],
        description="The final generated images",
    )
}
