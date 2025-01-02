import math
from typing import List, Optional, Union

import numpy as np
import torch

from ..sigmas.beta_sigmas import BetaSigmas
from ..sigmas.exponential_sigmas import ExponentialSigmas
from ..sigmas.karras_sigmas import KarrasSigmas


def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class BetaSchedule:
    scale_model_input = True

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        rescale_betas_zero_snr: bool = False,
        interpolation_type: str = "linear",
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",  # can be "discrete" or "continuous"
        steps_offset: int = 0,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        final_sigmas_type: str = "zero",  # can be "zero" or "sigma_min"
        **kwargs,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2**-24

        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.interpolation_type = interpolation_type
        self.timestep_spacing = timestep_spacing
        self.timestep_type = timestep_type
        self.steps_offset = steps_offset
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.final_sigmas_type = final_sigmas_type

    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    def __call__(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        sigma_schedule: Optional[Union[KarrasSigmas, ExponentialSigmas, BetaSigmas]] = None,
        **kwargs,
    ):
        if sigmas is not None:
            log_sigmas = np.log(np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5))
            # NOTE: current usage is **with** `sigma_last` - different than FlowMatch.
            sigmas = np.array(sigmas).astype(np.float32)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas[:-1]])

        else:
            if timesteps is not None:
                timesteps = np.array(timesteps).astype(np.float32)
            else:
                # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
                if self.timestep_spacing == "linspace":
                    timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[
                        ::-1
                    ].copy()
                elif self.timestep_spacing == "leading":
                    step_ratio = self.num_train_timesteps // num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (
                        (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
                    )
                    timesteps += self.steps_offset
                elif self.timestep_spacing == "trailing":
                    step_ratio = self.num_train_timesteps / num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (np.arange(self.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
                    timesteps -= 1
                else:
                    raise ValueError(
                        f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                    )

            sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = np.log(sigmas)
            if self.interpolation_type == "linear":
                sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            elif self.interpolation_type == "log_linear":
                sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp().numpy()
            else:
                raise ValueError(
                    f"{self.interpolation_type} is not implemented. Please specify interpolation_type to either"
                    " 'linear' or 'log_linear'"
                )

            if sigma_schedule is not None:
                sigmas = sigma_schedule(sigmas)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            if self.final_sigmas_type == "sigma_min":
                sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            elif self.final_sigmas_type == "zero":
                sigma_last = 0
            else:
                raise ValueError(
                    f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.final_sigmas_type}"
                )

            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        # TODO: Support the full EDM scalings for all prediction types and timestep types
        if self.timestep_type == "continuous" and self.prediction_type == "v_prediction":
            timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas[:-1]]).to(device=device)
        else:
            timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

        return sigmas, timesteps
