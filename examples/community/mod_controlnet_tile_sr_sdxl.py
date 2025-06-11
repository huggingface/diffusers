# Copyright 2025 The DEVAIEXP Team and The HuggingFace Team. All rights reserved.
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

import inspect
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import (
    AutoencoderKL,
    ControlNetModel,
    ControlNetUnionModel,
    MultiControlNetModel,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers, LMSDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.import_utils import is_invisible_watermark_available
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

from diffusers.utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        import torch
        from diffusers import DiffusionPipeline, ControlNetUnionModel, AutoencoderKL, UniPCMultistepScheduler
        from diffusers.utils import load_image
        from PIL import Image

        device = "cuda"

        # Initialize the models and pipeline
        controlnet = ControlNetUnionModel.from_pretrained(
            "brad-twinkl/controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16
        ).to(device=device)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device=device)

        model_id = "SG161222/RealVisXL_V5.0"
        pipe = StableDiffusionXLControlNetTileSRPipeline.from_pretrained(
            model_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        ).to(device)

        pipe.enable_model_cpu_offload()  # << Enable this if you have limited VRAM
        pipe.enable_vae_tiling() # << Enable this if you have limited VRAM
        pipe.enable_vae_slicing() # << Enable this if you have limited VRAM

        # Set selected scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Load image
        control_image = load_image("https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1.jpg")
        original_height = control_image.height
        original_width = control_image.width
        print(f"Current resolution: H:{original_height} x W:{original_width}")

        # Pre-upscale image for tiling
        resolution = 4096
        tile_gaussian_sigma = 0.3
        max_tile_size = 1024 # or 1280

        current_size = max(control_image.size)
        scale_factor = max(2, resolution / current_size)
        new_size = (int(control_image.width * scale_factor), int(control_image.height * scale_factor))
        image = control_image.resize(new_size, Image.LANCZOS)

        # Update target height and width
        target_height = image.height
        target_width = image.width
        print(f"Target resolution: H:{target_height} x W:{target_width}")

        # Calculate overlap size
        normal_tile_overlap, border_tile_overlap = calculate_overlap(target_width, target_height)

        # Set other params
        tile_weighting_method = TileWeightingMethod.COSINE.value
        guidance_scale = 4
        num_inference_steps = 35
        denoising_strenght = 0.65
        controlnet_strength = 1.0
        prompt = "high-quality, noise-free edges, high quality, 4k, hd, 8k"
        negative_prompt = "blurry, pixelated, noisy, low resolution, artifacts, poor details"

        # Image generation
        control_image = pipe(
            image=image,
            control_image=control_image,
            control_mode=[6],
            controlnet_conditioning_scale=float(controlnet_strength),
            prompt=prompt,
            negative_prompt=negative_prompt,
            normal_tile_overlap=normal_tile_overlap,
            border_tile_overlap=border_tile_overlap,
            height=target_height,
            width=target_width,
            original_size=(original_width, original_height),
            target_size=(target_width, target_height),
            guidance_scale=guidance_scale,
            strength=float(denoising_strenght),
            tile_weighting_method=tile_weighting_method,
            max_tile_size=max_tile_size,
            tile_gaussian_sigma=float(tile_gaussian_sigma),
            num_inference_steps=num_inference_steps,
        )["images"][0]
        ```
"""


# This function was copied and adapted from https://huggingface.co/spaces/gokaygokay/TileUpscalerV2, licensed under Apache 2.0.
def _adaptive_tile_size(image_size, base_tile_size=512, max_tile_size=1280):
    """
    Calculate the adaptive tile size based on the image dimensions, ensuring the tile
    respects the aspect ratio and stays within the specified size limits.
    """
    width, height = image_size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # Landscape orientation
        tile_width = min(width, max_tile_size)
        tile_height = min(int(tile_width / aspect_ratio), max_tile_size)
    else:
        # Portrait or square orientation
        tile_height = min(height, max_tile_size)
        tile_width = min(int(tile_height * aspect_ratio), max_tile_size)

    # Ensure the tile size is not smaller than the base_tile_size
    tile_width = max(tile_width, base_tile_size)
    tile_height = max(tile_height, base_tile_size)

    return tile_width, tile_height


# Copied and adapted from https://github.com/huggingface/diffusers/blob/main/examples/community/mixture_tiling.py
def _tile2pixel_indices(
    tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, image_width, image_height
):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    # Calculate initial indices
    px_row_init = 0 if tile_row == 0 else tile_row * (tile_height - tile_row_overlap)
    px_col_init = 0 if tile_col == 0 else tile_col * (tile_width - tile_col_overlap)

    # Calculate end indices
    px_row_end = px_row_init + tile_height
    px_col_end = px_col_init + tile_width

    # Ensure the last tile does not exceed the image dimensions
    px_row_end = min(px_row_end, image_height)
    px_col_end = min(px_col_end, image_width)

    return px_row_init, px_row_end, px_col_init, px_col_end


# Copied and adapted from https://github.com/huggingface/diffusers/blob/main/examples/community/mixture_tiling.py
def _tile2latent_indices(
    tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, image_width, image_height
):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    # Get pixel indices
    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
        tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, image_width, image_height
    )

    # Convert to latent space
    latent_row_init = px_row_init // 8
    latent_row_end = px_row_end // 8
    latent_col_init = px_col_init // 8
    latent_col_end = px_col_end // 8
    latent_height = image_height // 8
    latent_width = image_width // 8

    # Ensure the last tile does not exceed the latent dimensions
    latent_row_end = min(latent_row_end, latent_height)
    latent_col_end = min(latent_col_end, latent_width)

    return latent_row_init, latent_row_end, latent_col_init, latent_col_end


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class StableDiffusionXLControlNetTileSRPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for image-to-image generation using Stable Diffusion XL with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetUnionModel`]):
            Provides additional conditioning to the unet during the denoising process.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
            Whether the `unet` requires an `aesthetic_score` condition to be passed during inference. Also see the
            config of `stabilityai/stable-diffusion-xl-refiner-1-0`.
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetUnionModel,
        scheduler: KarrasDiffusionSchedulers,
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        if not isinstance(controlnet, ControlNetUnionModel):
            raise ValueError("Expected `controlnet` to be of type `ControlNetUnionModel`.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

    def calculate_overlap(self, width, height, base_overlap=128):
        """
        Calculates dynamic overlap based on the image's aspect ratio.

        Args:
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            base_overlap (int, optional): Base overlap value in pixels. Defaults to 128.

        Returns:
            tuple: A tuple containing:
                - row_overlap (int): Overlap between tiles in consecutive rows.
                - col_overlap (int): Overlap between tiles in consecutive columns.
        """
        ratio = height / width
        if ratio < 1:  # Image is wider than tall
            return base_overlap // 2, base_overlap
        else:  # Image is taller than wide
            return base_overlap, base_overlap * 2

    class TileWeightingMethod(Enum):
        """Mode in which the tile weights will be generated"""

        COSINE = "Cosine"
        GAUSSIAN = "Gaussian"

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        dtype = text_encoders[0].dtype
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )
                text_encoder.to(dtype)
                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                if pooled_prompt_embeds is None and prompt_embeds[0].ndim == 2:
                    pooled_prompt_embeds = prompt_embeds[0]

                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                if negative_pooled_prompt_embeds is None and negative_prompt_embeds[0].ndim == 2:
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        image,
        strength,
        num_inference_steps,
        normal_tile_overlap,
        border_tile_overlap,
        max_tile_size,
        tile_gaussian_sigma,
        tile_weighting_method,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        if num_inference_steps is None:
            raise ValueError("`num_inference_steps` cannot be None.")
        elif not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError(
                f"`num_inference_steps` has to be a positive integer but is {num_inference_steps} of type"
                f" {type(num_inference_steps)}."
            )
        if normal_tile_overlap is None:
            raise ValueError("`normal_tile_overlap` cannot be None.")
        elif not isinstance(normal_tile_overlap, int) or normal_tile_overlap < 64:
            raise ValueError(
                f"`normal_tile_overlap` has to be greater than 64 but is {normal_tile_overlap} of type"
                f" {type(normal_tile_overlap)}."
            )
        if border_tile_overlap is None:
            raise ValueError("`border_tile_overlap` cannot be None.")
        elif not isinstance(border_tile_overlap, int) or border_tile_overlap < 128:
            raise ValueError(
                f"`border_tile_overlap` has to be greater than 128 but is {border_tile_overlap} of type"
                f" {type(border_tile_overlap)}."
            )
        if max_tile_size is None:
            raise ValueError("`max_tile_size` cannot be None.")
        elif not isinstance(max_tile_size, int) or max_tile_size not in (1024, 1280):
            raise ValueError(
                f"`max_tile_size` has to be in 1024 or 1280 but is {max_tile_size} of type {type(max_tile_size)}."
            )
        if tile_gaussian_sigma is None:
            raise ValueError("`tile_gaussian_sigma` cannot be None.")
        elif not isinstance(tile_gaussian_sigma, float) or tile_gaussian_sigma <= 0:
            raise ValueError(
                f"`tile_gaussian_sigma` has to be a positive float but is {tile_gaussian_sigma} of type"
                f" {type(tile_gaussian_sigma)}."
            )
        if tile_weighting_method is None:
            raise ValueError("`tile_weighting_method` cannot be None.")
        elif not isinstance(tile_weighting_method, str) or tile_weighting_method not in [
            t.value for t in self.TileWeightingMethod
        ]:
            raise ValueError(
                f"`tile_weighting_method` has to be a string in ({[t.value for t in self.TileWeightingMethod]}) but is {tile_weighting_method} of type"
                f" {type(tile_weighting_method)}."
            )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt)
        elif (
            isinstance(self.controlnet, ControlNetUnionModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetUnionModel)
        ):
            self.check_image(image, prompt)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetUnionModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetUnionModel)
        ) or (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.check_image
    def check_image(self, image, prompt):
        image_is_pil = isinstance(image, Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.prepare_image
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.prepare_latents
    def prepare_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
    ):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        latents_mean = latents_std = None
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)

        # Offload text encoder if `enable_model_cpu_offload` was enabled
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                    image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            if latents_mean is not None and latents_std is not None:
                latents_mean = latents_mean.to(device=device, dtype=dtype)
                latents_std = latents_std.to(device=device, dtype=dtype)
                init_latents = (init_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
            else:
                init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def _generate_cosine_weights(self, tile_width, tile_height, nbatches, device, dtype):
        """
        Generates cosine weights as a PyTorch tensor for blending tiles.

        Args:
            tile_width (int): Width of the tile in pixels.
            tile_height (int): Height of the tile in pixels.
            nbatches (int): Number of batches.
            device (torch.device): Device where the tensor will be allocated (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type of the tensor (e.g., torch.float32).

        Returns:
            torch.Tensor: A tensor containing cosine weights for blending tiles, expanded to match batch and channel dimensions.
        """
        # Convert tile dimensions to latent space
        latent_width = tile_width // 8
        latent_height = tile_height // 8

        # Generate x and y coordinates in latent space
        x = np.arange(0, latent_width)
        y = np.arange(0, latent_height)

        # Calculate midpoints
        midpoint_x = (latent_width - 1) / 2
        midpoint_y = (latent_height - 1) / 2

        # Compute cosine probabilities for x and y
        x_probs = np.cos(np.pi * (x - midpoint_x) / latent_width)
        y_probs = np.cos(np.pi * (y - midpoint_y) / latent_height)

        # Create a 2D weight matrix using the outer product
        weights_np = np.outer(y_probs, x_probs)

        # Convert to a PyTorch tensor with the correct device and dtype
        weights_torch = torch.tensor(weights_np, device=device, dtype=dtype)

        # Expand for batch and channel dimensions
        tile_weights_expanded = torch.tile(weights_torch, (nbatches, self.unet.config.in_channels, 1, 1))

        return tile_weights_expanded

    def _generate_gaussian_weights(self, tile_width, tile_height, nbatches, device, dtype, sigma=0.05):
        """
        Generates Gaussian weights as a PyTorch tensor for blending tiles in latent space.

        Args:
            tile_width (int): Width of the tile in pixels.
            tile_height (int): Height of the tile in pixels.
            nbatches (int): Number of batches.
            device (torch.device): Device where the tensor will be allocated (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type of the tensor (e.g., torch.float32).
            sigma (float, optional): Standard deviation of the Gaussian distribution. Controls the smoothness of the weights. Defaults to 0.05.

        Returns:
            torch.Tensor: A tensor containing Gaussian weights for blending tiles, expanded to match batch and channel dimensions.
        """
        # Convert tile dimensions to latent space
        latent_width = tile_width // 8
        latent_height = tile_height // 8

        # Generate Gaussian weights in latent space
        x = np.linspace(-1, 1, latent_width)
        y = np.linspace(-1, 1, latent_height)
        xx, yy = np.meshgrid(x, y)
        gaussian_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Convert to a PyTorch tensor with the correct device and dtype
        weights_torch = torch.tensor(gaussian_weight, device=device, dtype=dtype)

        # Expand for batch and channel dimensions
        weights_expanded = weights_torch.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        weights_expanded = weights_expanded.expand(nbatches, -1, -1, -1)  # Expand to the number of batches

        return weights_expanded

    def _get_num_tiles(self, height, width, tile_height, tile_width, normal_tile_overlap, border_tile_overlap):
        """
        Calculates the number of tiles needed to cover an image, choosing the appropriate formula based on the
        ratio between the image size and the tile size.

        This function automatically selects between two formulas:
        1. A universal formula for typical cases (image-to-tile ratio <= 6:1).
        2. A specialized formula with border tile overlap for larger or atypical cases (image-to-tile ratio > 6:1).

        Args:
            height (int): Height of the image in pixels.
            width (int): Width of the image in pixels.
            tile_height (int): Height of each tile in pixels.
            tile_width (int): Width of each tile in pixels.
            normal_tile_overlap (int): Overlap between tiles in pixels for normal (non-border) tiles.
            border_tile_overlap (int): Overlap between tiles in pixels for border tiles.

        Returns:
            tuple: A tuple containing:
                - grid_rows (int): Number of rows in the tile grid.
                - grid_cols (int): Number of columns in the tile grid.

        Notes:
            - The function uses the universal formula (without border_tile_overlap) for typical cases where the
            image-to-tile ratio is 6:1 or smaller.
            - For larger or atypical cases (image-to-tile ratio > 6:1), it uses a specialized formula that includes
            border_tile_overlap to ensure complete coverage of the image, especially at the edges.
        """
        # Calculate the ratio between the image size and the tile size
        height_ratio = height / tile_height
        width_ratio = width / tile_width

        # If the ratio is greater than 6:1, use the formula with border_tile_overlap
        if height_ratio > 6 or width_ratio > 6:
            grid_rows = int(np.ceil((height - border_tile_overlap) / (tile_height - normal_tile_overlap))) + 1
            grid_cols = int(np.ceil((width - border_tile_overlap) / (tile_width - normal_tile_overlap))) + 1
        else:
            # Otherwise, use the universal formula
            grid_rows = int(np.ceil((height - normal_tile_overlap) / (tile_height - normal_tile_overlap)))
            grid_cols = int(np.ceil((width - normal_tile_overlap) / (tile_width - normal_tile_overlap)))

        return grid_rows, grid_cols

    def prepare_tiles(
        self,
        grid_rows,
        grid_cols,
        tile_weighting_method,
        tile_width,
        tile_height,
        normal_tile_overlap,
        border_tile_overlap,
        width,
        height,
        tile_sigma,
        batch_size,
        device,
        dtype,
    ):
        """
        Processes image tiles by dynamically adjusting overlap and calculating Gaussian or cosine weights.

        Args:
            grid_rows (int): Number of rows in the tile grid.
            grid_cols (int): Number of columns in the tile grid.
            tile_weighting_method (str): Method for weighting tiles. Options: "Gaussian" or "Cosine".
            tile_width (int): Width of each tile in pixels.
            tile_height (int): Height of each tile in pixels.
            normal_tile_overlap (int): Overlap between tiles in pixels for normal tiles.
            border_tile_overlap (int): Overlap between tiles in pixels for border tiles.
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            tile_sigma (float): Sigma parameter for Gaussian weighting.
            batch_size (int): Batch size for weight tiles.
            device (torch.device): Device where tensors will be allocated (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type of the tensors (e.g., torch.float32).

        Returns:
            tuple: A tuple containing:
                - tile_weights (np.ndarray): Array of weights for each tile.
                - tile_row_overlaps (np.ndarray): Array of row overlaps for each tile.
                - tile_col_overlaps (np.ndarray): Array of column overlaps for each tile.
        """

        # Create arrays to store dynamic overlaps and weights
        tile_row_overlaps = np.full((grid_rows, grid_cols), normal_tile_overlap)
        tile_col_overlaps = np.full((grid_rows, grid_cols), normal_tile_overlap)
        tile_weights = np.empty((grid_rows, grid_cols), dtype=object)  # Stores Gaussian or cosine weights

        # Iterate over tiles to adjust overlap and calculate weights
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate the size of the current tile
                px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
                    row, col, tile_width, tile_height, normal_tile_overlap, normal_tile_overlap, width, height
                )
                current_tile_width = px_col_end - px_col_init
                current_tile_height = px_row_end - px_row_init
                sigma = tile_sigma

                # Adjust overlap for smaller tiles
                if current_tile_width < tile_width:
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
                        row, col, tile_width, tile_height, border_tile_overlap, border_tile_overlap, width, height
                    )
                    current_tile_width = px_col_end - px_col_init
                    tile_col_overlaps[row, col] = border_tile_overlap
                    sigma = tile_sigma * 1.2
                if current_tile_height < tile_height:
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
                        row, col, tile_width, tile_height, border_tile_overlap, border_tile_overlap, width, height
                    )
                    current_tile_height = px_row_end - px_row_init
                    tile_row_overlaps[row, col] = border_tile_overlap
                    sigma = tile_sigma * 1.2

                # Calculate weights for the current tile
                if tile_weighting_method == self.TileWeightingMethod.COSINE.value:
                    tile_weights[row, col] = self._generate_cosine_weights(
                        tile_width=current_tile_width,
                        tile_height=current_tile_height,
                        nbatches=batch_size,
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    tile_weights[row, col] = self._generate_gaussian_weights(
                        tile_width=current_tile_width,
                        tile_height=current_tile_height,
                        nbatches=batch_size,
                        device=device,
                        dtype=dtype,
                        sigma=sigma,
                    )

        return tile_weights, tile_row_overlaps, tile_col_overlaps

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_mode: Optional[Union[int, List[int]]] = None,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        normal_tile_overlap: int = 64,
        border_tile_overlap: int = 128,
        max_tile_size: int = 1024,
        tile_gaussian_sigma: float = 0.05,
        tile_weighting_method: str = "Cosine",
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`, *optional*):
                The initial image to be used as the starting point for the image generation process. Can also accept
                image latents as `image`, if passing latents directly, they will not be encoded again.
            control_image (`PipelineImageInput`, *optional*):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance for Unet.
                If the type is specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also
                be accepted as an image. The dimensions of the output image default to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                init, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*):
                The height in pixels of the generated image. If not provided, defaults to the height of `control_image`.
            width (`int`, *optional*):
                The width in pixels of the generated image. If not provided, defaults to the width of `control_image`.
            strength (`float`, *optional*, defaults to 0.9999):
                Indicates the extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point, and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum, and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487).
                Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages generating
                images closely linked to the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/):
                `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original UNet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_mode (`int` or `List[int]`, *optional*):
                The mode of ControlNet guidance. Can be used to specify different behaviors for multiple ControlNets.
            original_size (`Tuple[int, int]`, *optional*):
                If `original_size` is not the same as `target_size`, the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning.
            crops_coords_top_left (`Tuple[int, int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning.
            target_size (`Tuple[int, int]`, *optional*):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified, it will default to `(height, width)`. Part of SDXL's micro-conditioning.
            negative_original_size (`Tuple[int, int]`, *optional*):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning.
            negative_crops_coords_top_left (`Tuple[int, int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning.
            negative_target_size (`Tuple[int, int]`, *optional*):
                To negatively condition the generation process based on a target image resolution. It should be the same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning.
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Used to simulate an aesthetic score of the generated image by influencing the negative text condition.
                Part of SDXL's micro-conditioning.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            normal_tile_overlap (`int`, *optional*, defaults to 64):
                Number of overlapping pixels between tiles in consecutive rows.
            border_tile_overlap (`int`, *optional*, defaults to 128):
                Number of overlapping pixels between tiles at the borders.
            max_tile_size (`int`, *optional*, defaults to 1024):
                Maximum size of a tile in pixels.
            tile_gaussian_sigma (`float`, *optional*, defaults to 0.3):
                Sigma parameter for Gaussian weighting of tiles.
            tile_weighting_method (`str`, *optional*, defaults to "Cosine"):
                Method for weighting tiles. Options: "Cosine" or "Gaussian".

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`
            containing the output images.
        """

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]

        if not isinstance(control_image, list):
            control_image = [control_image]
        else:
            control_image = control_image.copy()

        if control_mode is None or isinstance(control_mode, list) and len(control_mode) == 0:
            raise ValueError("The value for `control_mode` is expected!")

        if not isinstance(control_mode, list):
            control_mode = [control_mode]

        if len(control_image) != len(control_mode):
            raise ValueError("Expected len(control_image) == len(control_mode)")

        num_control_type = controlnet.config.num_control_type

        # 0. Set internal use parameters
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        negative_original_size = negative_original_size or original_size
        negative_target_size = negative_target_size or target_size
        control_type = [0 for _ in range(num_control_type)]
        control_type = torch.Tensor(control_type)
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False
        batch_size = 1
        device = self._execution_device
        global_pool_conditions = controlnet.config.global_pool_conditions
        guess_mode = guess_mode or global_pool_conditions

        # 1. Check inputs
        for _image, control_idx in zip(control_image, control_mode):
            control_type[control_idx] = 1
            self.check_inputs(
                prompt,
                height,
                width,
                _image,
                strength,
                num_inference_steps,
                normal_tile_overlap,
                border_tile_overlap,
                max_tile_size,
                tile_gaussian_sigma,
                tile_weighting_method,
                controlnet_conditioning_scale,
                control_guidance_start,
                control_guidance_end,
            )

        # 2 Get tile width and tile height size
        tile_width, tile_height = _adaptive_tile_size((width, height), max_tile_size=max_tile_size)

        # 2.1 Calculate the number of tiles needed
        grid_rows, grid_cols = self._get_num_tiles(
            height, width, tile_height, tile_width, normal_tile_overlap, border_tile_overlap
        )

        # 2.2 Expand prompt to number of tiles
        if not isinstance(prompt, list):
            prompt = [[prompt] * grid_cols] * grid_rows

        # 2.3 Update height and width tile size by tile size and tile overlap size
        width = (grid_cols - 1) * (tile_width - normal_tile_overlap) + min(
            tile_width, width - (grid_cols - 1) * (tile_width - normal_tile_overlap)
        )
        height = (grid_rows - 1) * (tile_height - normal_tile_overlap) + min(
            tile_height, height - (grid_rows - 1) * (tile_height - normal_tile_overlap)
        )

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        text_embeddings = [
            [
                self.encode_prompt(
                    prompt=col,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    negative_pooled_prompt_embeds=None,
                    lora_scale=text_encoder_lora_scale,
                    clip_skip=self.clip_skip,
                )
                for col in row
            ]
            for row in prompt
        ]

        # 4. Prepare latent image
        image_tensor = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        # 4.1 Prepare controlnet_conditioning_image
        control_image = self.prepare_control_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        control_type = (
            control_type.reshape(1, -1)
            .to(device, dtype=controlnet.dtype)
            .repeat(batch_size * num_images_per_prompt * 2, 1)
        )

        # 5. Prepare timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1
        self.scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        dtype = text_embeddings[0][0][0].dtype
        if latents is None:
            latents = self.prepare_latents(
                image_tensor,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                dtype,
                device,
                generator,
                True,
            )

        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            controlnet_keep.append(
                1.0
                - float(i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end)
            )

        # 8.1 Prepare added time ids & embeddings
        # text_embeddings order: prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        embeddings_and_added_time = []
        crops_coords_top_left = negative_crops_coords_top_left = (tile_width, tile_height)
        for row in range(grid_rows):
            addition_embed_type_row = []
            for col in range(grid_cols):
                # extract generated values
                prompt_embeds = text_embeddings[row][col][0]
                negative_prompt_embeds = text_embeddings[row][col][1]
                pooled_prompt_embeds = text_embeddings[row][col][2]
                negative_pooled_prompt_embeds = text_embeddings[row][col][3]

                if negative_original_size is None:
                    negative_original_size = original_size
                if negative_target_size is None:
                    negative_target_size = target_size
                add_text_embeds = pooled_prompt_embeds

                if self.text_encoder_2 is None:
                    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                else:
                    text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

                add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    aesthetic_score,
                    negative_aesthetic_score,
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
                add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                    add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device)
                add_text_embeds = add_text_embeds.to(device)
                add_time_ids = add_time_ids.to(device)
                addition_embed_type_row.append((prompt_embeds, add_text_embeds, add_time_ids))

            embeddings_and_added_time.append(addition_embed_type_row)

        # 9. Prepare tiles weights and latent overlaps size to denoising process
        tile_weights, tile_row_overlaps, tile_col_overlaps = self.prepare_tiles(
            grid_rows,
            grid_cols,
            tile_weighting_method,
            tile_width,
            tile_height,
            normal_tile_overlap,
            border_tile_overlap,
            width,
            height,
            tile_gaussian_sigma,
            batch_size,
            device,
            dtype,
        )

        # 10. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Diffuse each tile
                noise_preds = []
                for row in range(grid_rows):
                    noise_preds_row = []
                    for col in range(grid_cols):
                        if self.interrupt:
                            continue
                        tile_row_overlap = tile_row_overlaps[row, col]
                        tile_col_overlap = tile_col_overlaps[row, col]

                        px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(
                            row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, width, height
                        )

                        tile_latents = latents[:, :, px_row_init:px_row_end, px_col_init:px_col_end]

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = (
                            torch.cat([tile_latents] * 2)
                            if self.do_classifier_free_guidance
                            else tile_latents  # 1, 4, ...
                        )
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        added_cond_kwargs = {
                            "text_embeds": embeddings_and_added_time[row][col][1],
                            "time_ids": embeddings_and_added_time[row][col][2],
                        }

                        # controlnet(s) inference
                        if guess_mode and self.do_classifier_free_guidance:
                            # Infer ControlNet only for the conditional batch.
                            control_model_input = tile_latents
                            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                            controlnet_prompt_embeds = embeddings_and_added_time[row][col][0].chunk(2)[1]
                            controlnet_added_cond_kwargs = {
                                "text_embeds": embeddings_and_added_time[row][col][1].chunk(2)[1],
                                "time_ids": embeddings_and_added_time[row][col][2].chunk(2)[1],
                            }
                        else:
                            control_model_input = latent_model_input
                            controlnet_prompt_embeds = embeddings_and_added_time[row][col][0]
                            controlnet_added_cond_kwargs = added_cond_kwargs

                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * controlnet_keep[i]

                        px_row_init_pixel, px_row_end_pixel, px_col_init_pixel, px_col_end_pixel = _tile2pixel_indices(
                            row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, width, height
                        )

                        tile_control_image = control_image[
                            :, :, px_row_init_pixel:px_row_end_pixel, px_col_init_pixel:px_col_end_pixel
                        ]

                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=[tile_control_image],
                            control_type=control_type,
                            control_type_idx=control_mode,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            added_cond_kwargs=controlnet_added_cond_kwargs,
                            return_dict=False,
                        )

                        if guess_mode and self.do_classifier_free_guidance:
                            # Inferred ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [
                                torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
                            ]
                            mid_block_res_sample = torch.cat(
                                [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                            )

                        # predict the noise residual
                        with torch.amp.autocast(device.type, dtype=dtype, enabled=dtype != self.unet.dtype):
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=embeddings_and_added_time[row][col][0],
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred_tile = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )
                            noise_preds_row.append(noise_pred_tile)
                    noise_preds.append(noise_preds_row)

                # Stitch noise predictions for all tiles
                noise_pred = torch.zeros(latents.shape, device=device)
                contributors = torch.zeros(latents.shape, device=device)

                # Add each tile contribution to overall latents
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        tile_row_overlap = tile_row_overlaps[row, col]
                        tile_col_overlap = tile_col_overlaps[row, col]
                        px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(
                            row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, width, height
                        )
                        tile_weights_resized = tile_weights[row, col]

                        noise_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += (
                            noise_preds[row][col] * tile_weights_resized
                        )
                        contributors[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += tile_weights_resized

                # Average overlapping areas with more than 1 contributor
                noise_pred /= contributors
                noise_pred = noise_pred.to(dtype)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        result = StableDiffusionXLPipelineOutput(images=image)
        if not return_dict:
            return (image,)

        return result
