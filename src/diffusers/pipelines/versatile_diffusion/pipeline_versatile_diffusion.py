import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch

import PIL.Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import logging
from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline
from .pipeline_versatile_diffusion_text_to_image import VersatileDiffusionTextToImagePipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VersatileDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionMegaSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    tokenizer: CLIPTokenizer
    image_feature_extractor: CLIPFeatureExtractor
    text_encoder: CLIPTextModel
    image_encoder: CLIPVisionModel
    image_unet: UNet2DConditionModel
    text_unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        image_feature_extractor: CLIPFeatureExtractor,
        text_encoder: CLIPTextModel,
        image_encoder: CLIPVisionModel,
        image_unet: UNet2DConditionModel,
        text_unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            image_feature_extractor=image_feature_extractor,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_unet=image_unet,
            text_unet=text_unet,
            vae=vae,
            scheduler=scheduler,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.image_unet.config.attention_head_dim // 2
        self.image_unet.set_attention_slice(slice_size)
        self.text_unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def image_variation(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        expected_components = inspect.signature(VersatileDiffusionImageVariationPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        return VersatileDiffusionImageVariationPipeline(**components)(
            image=image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )

    @torch.no_grad()
    def text_to_image(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        expected_components = inspect.signature(VersatileDiffusionTextToImagePipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        return VersatileDiffusionTextToImagePipeline(**components)(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )

    @torch.no_grad()
    def dual_guided(
        self,
        first_prompt: Union[str, List[str], PIL.Image.Image, List[PIL.Image.Image]],
        second_prompt: Union[str, List[str], PIL.Image.Image, List[PIL.Image.Image]],
        prompt_mix_ratio: float = 0.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        expected_components = inspect.signature(VersatileDiffusionDualGuidedPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = VersatileDiffusionDualGuidedPipeline(**components)
        output = temp_pipeline(
            first_prompt=first_prompt,
            second_prompt=second_prompt,
            prompt_mix_ratio=prompt_mix_ratio,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )
        temp_pipeline._revert_dual_attention()

        return output
