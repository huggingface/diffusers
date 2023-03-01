from typing import Optional, Tuple, Union

import torch
from einops import rearrange, reduce

from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline, ImagePipelineOutput, UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput


BITS = 8


# convert to bit representations and back taken from https://github.com/lucidrains/bit-diffusion/blob/main/bit_diffusion/bit_diffusion.py
def decimal_to_bits(x, bits=BITS):
    """expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1"""
    device = x.device

    x = (x * 255).int().clamp(0, 255)

    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device)
    mask = rearrange(mask, "d -> d 1 1")
    x = rearrange(x, "b c h w -> b c 1 h w")

    bits = ((x & mask) != 0).float()
    bits = rearrange(bits, "b c d h w -> b (c d) h w")
    bits = bits * 2 - 1
    return bits


def bits_to_decimal(x, bits=BITS):
    """expects bits from -1 to 1, outputs image tensor from 0 to 1"""
    device = x.device

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int32)

    mask = rearrange(mask, "d -> d 1 1")
    x = rearrange(x, "b (c d) h w -> b c d h w", d=8)
    dec = reduce(x * mask, "b c d h w -> b c h w", "sum")
    return (dec / 255).clamp(0.0, 1.0)


# modified scheduler step functions for clamping the predicted x_0 between -bit_scale and +bit_scale
def ddim_bit_scheduler_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = True,
    generator=None,
    return_dict: bool = True,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): TODO
        generator: random number generator.
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class
    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 4. Clip "predicted x_0"
    scale = self.bit_scale
    if self.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -scale, scale)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the model_output is always re-derived from the clipped x_0 in Glide
        model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if eta > 0:
        # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
        device = model_output.device if torch.is_tensor(model_output) else "cpu"
        noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator).to(device)
        variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * noise

        prev_sample = prev_sample + variance

    if not return_dict:
        return (prev_sample,)

    return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


def ddpm_bit_scheduler_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    prediction_type="epsilon",
    generator=None,
    return_dict: bool = True,
) -> Union[DDPMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        prediction_type (`str`, default `epsilon`):
            indicates whether the model predicts the noise (epsilon), or the samples (`sample`).
        generator: random number generator.
        return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class
    Returns:
        [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.
    """
    t = timestep

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif prediction_type == "sample":
        pred_original_sample = model_output
    else:
        raise ValueError(f"Unsupported prediction_type {prediction_type}.")

    # 3. Clip "predicted x_0"
    scale = self.bit_scale
    if self.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -scale, scale)

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[t]) / beta_prod_t
    current_sample_coeff = self.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # 6. Add noise
    variance = 0
    if t > 0:
        noise = torch.randn(
            model_output.size(), dtype=model_output.dtype, layout=model_output.layout, generator=generator
        ).to(model_output.device)
        variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * noise

    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (pred_prev_sample,)

    return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


class BitDiffusion(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        bit_scale: Optional[float] = 1.0,
    ):
        super().__init__()
        self.bit_scale = bit_scale
        self.scheduler.step = (
            ddim_bit_scheduler_step if isinstance(scheduler, DDIMScheduler) else ddpm_bit_scheduler_step
        )

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        batch_size: Optional[int] = 1,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height, width),
            generator=generator,
        )
        latents = decimal_to_bits(latents) * self.bit_scale
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # predict the noise residual
            noise_pred = self.unet(latents, t).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        image = bits_to_decimal(latents)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
