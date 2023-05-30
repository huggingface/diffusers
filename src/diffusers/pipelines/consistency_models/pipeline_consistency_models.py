import inspect
from typing import Callable, List, Optional, Union

import torch

from ...models import UNet2DModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


class ConsistencyModelPipeline(DiffusionPipeline):
    r"""
    Sampling pipeline for consistency models.
    """

    def __init__(self, unet: UNet2DModel, scheduler: KarrasDiffusionSchedulers, distillation: bool = False) -> None:
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

        self.distillation = distillation
        self.num_classes = unet.config.num_class_embeds

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    # Additionally prepare sigma_min, sigma_max kwargs for CM multistep scheduler
    def prepare_extra_step_kwargs(self, generator, eta, sigma_min, sigma_max):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_sigma_min = "sigma_min" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_sigma_min:
            # Assume accepting sigma_min always means scheduler also accepts sigma_max
            extra_step_kwargs["sigma_min"] = sigma_min
            extra_step_kwargs["sigma_max"] = sigma_max

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    # Unlike stable diffusion, no VAE so no vae_scale_factor, num_channels_latent => num_channels
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_scalings(self, sigma, sigma_data: float = 0.5):
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma, sigma_min: float = 0.002, sigma_data: float = 0.5):
        # sigma_min should be in original sigma space, not in karras sigma space
        # (e.g. not exponentiated by 1 / rho)
        c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
        c_out = (sigma - sigma_min) * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(
        self,
        x_t,
        sigma,
        class_labels=None,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        clip_denoised=True,
    ):
        """
        Run the consistency model forward...?
        """
        # sigma_min should be in original sigma space, not in karras sigma space
        # (e.g. not exponentiated by 1 / rho)
        if self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigma, sigma_min=sigma_min, sigma_data=sigma_data)
            ]
        else:
            c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_scalings(sigma, sigma_data=sigma_data)]
        rescaled_t = 1000 * 0.25 * torch.log(sigma + 1e-44)
        model_output = self.unet(c_in * x_t, rescaled_t, class_labels=class_labels).sample
        denoised = c_out * model_output + c_skip * x_t
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return model_output, denoised

    def to_d(x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / append_dims(sigma, x.ndim)
    
    def check_inputs(self, latents, batch_size, img_size, callback_steps):
        if latents is not None:
            expected_shape = (batch_size, 3, img_size, img_size)
            if latents.shape != expected_shape:
                raise ValueError(
                    f"The shape of latents is {latents.shape} but is expected to be {expected_shape}."
                )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        class_labels: Optional[Union[torch.Tensor, List[int], int]] = None,
        num_inference_steps: int = 40,
        clip_denoised: bool = True,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            class_labels (`torch.Tensor` or `List[int]` or `int`, *optional*):
                Optional class labels for conditioning class-conditional consistency models. Will not be used if the
                model is not class-conditional.
            num_inference_steps (`int`, *optional*, defaults to 40):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            clip_denoised (`bool`, *optional*, defaults to `True`):
                Whether to clip the consistency model denoising output to `(0, 1)`.
            sigma_min (`float`, *optional*, defaults to 0.002):
                The minimum (and last) value in the sigma noise schedule.
            sigma_max (`float`, *optional*, defaults to 80.0):
                The maximum (and first) value in the sigma noise schedule.
            sigma_data (`float`, *optional*, defaults to 0.5):
                TODO
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Prepare call parameters
        img_size = img_size = self.unet.config.sample_size
        device = self.device

        # 1. Check inputs
        self.check_inputs(latents, batch_size, img_size, callback_steps)

        # 2. Prepare image latents
        # Sample image latents x_0 ~ N(0, sigma_0^2 * I)
        sample = self.prepare_latents(
            batch_size=batch_size,
            num_channels=3,
            height=img_size,
            width=img_size,
            dtype=self.unet.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 3. Handle class_labels for class-conditional models
        if self.num_classes is not None:
            if isinstance(class_labels, list):
                class_labels = torch.tensor(class_labels, dtype=torch.int)
            elif isinstance(class_labels, int):
                assert batch_size == 1, "Batch size must be 1 if classes is an int"
                class_labels = torch.tensor([class_labels], dtype=torch.int)
            elif class_labels is None:
                # Randomly generate batch_size class labels
                class_labels = torch.randint(0, self.num_classes, size=batch_size)
            class_labels.to(device)

        # 4. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Now get Karras sigma schedule (which I think the original implementation always uses)
        # See https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L376
        # TODO: how do we ensure that this in Karras sigma space rather than in "original" sigma space?
        # 5. Get sigma schedule
        assert hasattr(self.scheduler, "sigmas"), "Scheduler needs to operate in sigma space"
        sigmas = self.scheduler.sigmas

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta, sigma_min, sigma_max)

        # 7. Denoising loop
        if num_inference_steps == 1:
            # Onestep sampling: simply evaluate the consistency model at the first sigma
            # See https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L643
            sigma = sigma_max
            sigma_in = sample.new_ones([sample.shape[0]]) * sigma
            _, sample = self.denoise(
                sample,
                sigma_in,
                class_labels=class_labels,
                sigma_min=sigma_min,
                sigma_data=sigma_data,
                clip_denoised=clip_denoised,
            )
        else:
            # Multistep sampling or Karras sampler
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                # Don't loop over last timestep
                for i, t in enumerate(timesteps[:-1]):
                    sigma = sigmas[i]
                    sigma_in = sample.new_ones([sample.shape[0]]) * sigma
                    # TODO: check shapes, might need equivalent of s_in in original code
                    # See e.g. https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L510
                    model_output, denoised = self.denoise(
                        sample,
                        sigma_in,
                        class_labels=class_labels,
                        sigma_min=sigma_min,
                        sigma_data=sigma_data,
                        clip_denoised=clip_denoised,
                    )

                    # Works for both Karras-style schedulers (e.g. Euler, Heun) and the CM multistep scheduler
                    sample = self.scheduler.step(denoised, t, sample, **extra_step_kwargs).prev_sample

                    # Note: differs from callback support in original code
                    # See e.g. https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L459
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, sample)

        # 8. Post-process image sample
        sample = (sample / 2 + 0.5).clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        # TODO: Offload to cpu?

        return ImagePipelineOutput(images=sample)
