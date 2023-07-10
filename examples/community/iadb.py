from typing import List, Optional, Tuple, Union

import torch

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class IADBScheduler(SchedulerMixin, ConfigMixin):
    """
    IADBScheduler is a scheduler for the Iterative α-(de)Blending denoising method. It is simple and minimalist.

    For more details, see the original paper: https://arxiv.org/abs/2305.03486 and the blog post: https://ggx-research.github.io/publication/2023/05/10/publication-iadb.html
    """

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x_alpha: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Predict the sample at the previous timestep by reversing the ODE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model. It is the direction from x0 to x1.
            timestep (`float`): current timestep in the diffusion chain.
            x_alpha (`torch.FloatTensor`): x_alpha sample for the current timestep

        Returns:
            `torch.FloatTensor`: the sample at the previous timestep

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        alpha = timestep / self.num_inference_steps
        alpha_next = (timestep + 1) / self.num_inference_steps

        d = model_output

        x_alpha = x_alpha + (alpha_next - alpha) * d

        return x_alpha

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        alpha: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return original_samples * alpha + noise * (1 - alpha)

    def __len__(self):
        return self.config.num_train_timesteps


class IADBPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = torch.randn(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        x_alpha = image.clone()
        for t in self.progress_bar(range(num_inference_steps)):
            alpha = t / num_inference_steps

            # 1. predict noise model_output
            model_output = self.unet(x_alpha, torch.tensor(alpha, device=x_alpha.device)).sample

            # 2. step
            x_alpha = self.scheduler.step(model_output, t, x_alpha)

        image = (x_alpha * 0.5 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
