import inspect
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint

import PIL

from ...models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class LDMSuperResolutionPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vqvae: Union[VQModel, AutoencoderKL],
        unet: Union[UNet2DModel, UNet2DConditionModel],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            height (`int`, *optional*, defaults to 256):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if isinstance(init_image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(init_image, torch.FloatTensor):
            batch_size = init_image.shape[0]
        else:
            raise ValueError(
                f"`init_image` has to be of type `PIL.Image.Image` or `torch.FloatTensor` but is {type(init_image)}"
            )

        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image)

        height, width = init_image.shape[-2:]

        latents = torch.randn(
            (batch_size, self.unet.in_channels // 2, height, width),
            device="cuda",
            generator=generator,
        )
        latents = latents.to(self.device)
        init_image = init_image.to(device=self.device, dtype=latents.dtype)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature.
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            # concat latents and low resolution image
            latents_input = torch.cat([latents, init_image], dim=1)
            # predict the noise residual
            noise_pred = self.unet(latents_input, t).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample

        # decode the image latents with the VQVAE
        image = self.vqvae.decode(latents).sample
        image = torch.clamp(image, -1.0, 1.0)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)