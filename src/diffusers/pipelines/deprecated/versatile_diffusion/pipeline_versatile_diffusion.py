import inspect
from typing import Callable, List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from ....models import AutoencoderKL, UNet2DConditionModel
from ....schedulers import KarrasDiffusionSchedulers
from ....utils import logging
from ...pipeline_utils import DiffusionPipeline
from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline
from .pipeline_versatile_diffusion_text_to_image import VersatileDiffusionTextToImagePipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VersatileDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    tokenizer: CLIPTokenizer
    image_feature_extractor: CLIPImageProcessor
    text_encoder: CLIPTextModel
    image_encoder: CLIPVisionModel
    image_unet: UNet2DConditionModel
    text_unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler: KarrasDiffusionSchedulers

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        image_feature_extractor: CLIPImageProcessor,
        text_encoder: CLIPTextModel,
        image_encoder: CLIPVisionModel,
        image_unet: UNet2DConditionModel,
        text_unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def image_variation(
        self,
        image: Union[torch.Tensor, PIL.Image.Image],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image`, `List[PIL.Image.Image]` or `torch.Tensor`):
                The image prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Examples:

        ```py
        >>> from diffusers import VersatileDiffusionPipeline
        >>> import torch
        >>> import requests
        >>> from io import BytesIO
        >>> from PIL import Image

        >>> # let's download an initial image
        >>> url = "https://huggingface.co/datasets/diffusers/images/resolve/main/benz.jpg"

        >>> response = requests.get(url)
        >>> image = Image.open(BytesIO(response.content)).convert("RGB")

        >>> pipe = VersatileDiffusionPipeline.from_pretrained(
        ...     "shi-labs/versatile-diffusion", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.Generator(device="cuda").manual_seed(0)
        >>> image = pipe.image_variation(image, generator=generator).images[0]
        >>> image.save("./car_variation.png")
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
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
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Examples:

        ```py
        >>> from diffusers import VersatileDiffusionPipeline
        >>> import torch

        >>> pipe = VersatileDiffusionPipeline.from_pretrained(
        ...     "shi-labs/versatile-diffusion", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.Generator(device="cuda").manual_seed(0)
        >>> image = pipe.text_to_image("an astronaut riding on a horse on mars", generator=generator).images[0]
        >>> image.save("./astronaut.png")
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        expected_components = inspect.signature(VersatileDiffusionTextToImagePipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = VersatileDiffusionTextToImagePipeline(**components)
        output = temp_pipeline(
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
        # swap the attention blocks back to the original state
        temp_pipeline._swap_unet_attention_blocks()

        return output

    @torch.no_grad()
    def dual_guided(
        self,
        prompt: Union[PIL.Image.Image, List[PIL.Image.Image]],
        image: Union[str, List[str]],
        text_to_image_strength: float = 0.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Examples:

        ```py
        >>> from diffusers import VersatileDiffusionPipeline
        >>> import torch
        >>> import requests
        >>> from io import BytesIO
        >>> from PIL import Image

        >>> # let's download an initial image
        >>> url = "https://huggingface.co/datasets/diffusers/images/resolve/main/benz.jpg"

        >>> response = requests.get(url)
        >>> image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> text = "a red car in the sun"

        >>> pipe = VersatileDiffusionPipeline.from_pretrained(
        ...     "shi-labs/versatile-diffusion", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.Generator(device="cuda").manual_seed(0)
        >>> text_to_image_strength = 0.75

        >>> image = pipe.dual_guided(
        ...     prompt=text, image=image, text_to_image_strength=text_to_image_strength, generator=generator
        ... ).images[0]
        >>> image.save("./car_variation.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        expected_components = inspect.signature(VersatileDiffusionDualGuidedPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = VersatileDiffusionDualGuidedPipeline(**components)
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            text_to_image_strength=text_to_image_strength,
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
