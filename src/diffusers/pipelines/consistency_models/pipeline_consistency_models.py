import inspect
from typing import List, Optional, Tuple, Union

import torch

from ...models import UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

class ConsistencyModelPipeline(DiffusionPipeline):
    r"""
    TODO
    """
    def __init__(self, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers) -> None:
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

        # Need to handle boundary conditions (e.g. c_skip, c_out, etc.) somewhere.
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
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
    
    def add_noise_to_input(
            self,
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            step: int = 0
        ):
            """
            Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
            higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.
            TODO Args:
            """
            pass

    
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        img_size = img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)
        device = self.device

        # 1. Sample image latents x_0 ~ N(0, sigma_0^2 * I)
        sample = randn_tensor(shape, generator=generator, device=device) * self.scheduler.init_noise_sigma

        # 2. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # TODO: should schedulers always have sigmas? I think the original code always uses sigmas
        # self.scheduler.set_sigmas(num_inference_steps)
        
        # 3. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 4. Denoising loop
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                # TODO: handle class labels?
                model_output = self.unet(sample, t)

                sample = self.scheduler.step(model_output, t, sample, **extra_step_kwargs).prev_sample

                # TODO: need to handle karras sigma stuff here?

                # TODO: need to support callbacks?
        
        # 5. Post-process image sample
        sample = sample.clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            sample = self.numpy_to_pil(sample)
        
        if not return_dict:
            return (sample,)
        
        # TODO: Offload to cpu?

        return ImagePipelineOutput(images=sample)
                



