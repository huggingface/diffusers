# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


def log_snr(t: torch.Tensor, beta_schedule: str) -> torch.Tensor:
    """
    Calculates the logarithm of the signal-to-noise ratio (SNR) for given time steps `t` under a specified beta
    schedule.

    See appendix K of the [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) paper for more details.

    Args:
        t (torch.Tensor): Tensor of time steps, normalized between [0, 1].
        beta_schedule (str):
            The beta schedule type. Supported types include 'linear', 'squaredcos_cap_v2', and 'sigmoid'.

    Returns:
        torch.Tensor: The log SNR values corresponding to the input time steps under the given beta schedule.

    Raises:
        ValueError: If `t` is outside the range [0, 1] or if the beta_schedule is unsupported.
    """
    if t.min() < 0 or t.max() > 1:
        raise ValueError("`t` must be in range [0, 1].")

    # From https://github.com/Zhengxinyang/LAS-Diffusion/blob/main/network/model_utils.py#L345
    if beta_schedule == "linear":
        return -torch.log(torch.special.expm1(1e-4 + 10 * t**2))
    elif beta_schedule == "squaredcos_cap_v2":
        return -torch.log(torch.clamp((torch.cos((t + 0.008) / (1 + 0.008) * math.pi * 0.5) ** -2) - 1, min=1e-5))
    elif beta_schedule == "sigmoid":
        # From https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb
        gamma_min = -6  # -13.3 in VDM CIFAR10 experiments
        gamma_max = 6  # 5.0 in VDM CIFAR10 experiments
        return gamma_max + (gamma_min - gamma_max) * t

    raise NotImplementedError(f"{beta_schedule} does is not implemented for {VDMScheduler.__class__}")


@dataclass
class VDMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class VDMScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements the discrete and continuous scheduler as presented in `Variational Diffusion Models` [1].

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to None, *optional*):
            The number of diffusion steps to train the model. If not provided, assumes continuous formulation.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `squaredcos_cap_v2` or `sigmoid`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            or `sample` (directly predicts the noisy sample`).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.

    References:
        [1] "Variational Diffusion Models" by Diederik P. Kingma, Tim Salimans, Ben Poole and Jonathan Ho, ArXiv, 2021.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: Optional[int] = None,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: Union[int, float] = 0,
    ):
        # Hardcoded as continuous schedules in `log_snr` are fitted to these values
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.init_noise_sigma = 1.0

        # For linear beta schedule, equivalent to torch.exp(-1e-4 - 10 * t ** 2)
        self.alphas_cumprod = lambda t: torch.sigmoid(self.log_snr(t))  # Equivalent to 1 - self.sigmas
        self.sigmas = lambda t: torch.sigmoid(-self.log_snr(t))  # Equivalent to 1 - self.alphas_cumprod

        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(self.get_timesteps(len(self)))
        if num_train_timesteps:
            alphas_cumprod = self.alphas_cumprod(torch.flip(self.timesteps, dims=(0,)))
            alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]  # TODO: Might not be exact
            self.alphas = torch.cat([alphas_cumprod[:1], alphas])
            self.betas = 1 - self.alphas

    def log_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Computes the logarithm of the signal-to-noise ratio for given timesteps using the configured beta schedule.

        Args:
            timesteps (torch.Tensor): Tensor of timesteps, which can be either normalized to [0, 1] range or discrete.

        Returns:
            torch.Tensor: The computed log SNR values for the given timesteps.

        Raises:
            TypeError: If discrete timesteps are used without setting `num_train_timesteps` in the configuration.
        """
        if not timesteps.is_floating_point():
            if not self.config.num_train_timesteps:
                raise TypeError("Discrete timesteps require `self.config.num_train_timesteps` to be set.")
            timesteps = timesteps / self.config.num_train_timesteps  # Normalize to [0, 1]

        return log_snr(timesteps, beta_schedule=self.config.beta_schedule)

    def get_timesteps(self, num_steps: Optional[int] = None) -> np.ndarray:
        """
        Generates timesteps in the range [0, 1] for the continuous formulation.

        Args:
            num_steps (int, optional): The number of timesteps to generate. Defaults to `num_train_timesteps`.

        Returns:
            np.ndarray: An array of timesteps, distributed according to the `timestep_spacing` configuration.

        Raises:
            ValueError: If an unsupported `timestep_spacing` configuration is provided.
        """
        if num_steps is None:
            num_steps = len(self)
        if self.config.timestep_spacing in ["linspace", "leading"]:
            timesteps = np.linspace(0, 1, num_steps, endpoint=self.config.timestep_spacing == "linspace")[::-1]
        elif self.config.timestep_spacing == "trailing":
            timesteps = np.arange(1, 0, -1 / num_steps) - 1 / num_steps
        else:
            raise ValueError(
                f"`{self.config.timestep_spacing}` timestep spacing is not supported."
                "Choose one of 'linspace', 'leading' or 'trailing'."
            )
        return timesteps.astype(np.float32).copy()

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None):
        """
        Sets the discrete or continuous timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

        Raises:
            ValueError: If an unsupported `timestep_spacing` configuration is provided.
        """
        if not self.config.num_train_timesteps:
            timesteps = self.get_timesteps(num_inference_steps)
        else:
            if self.config.timestep_spacing in ["linspace", "leading"]:
                start = 0
                stop = self.config.num_train_timesteps
                timesteps = np.linspace(
                    start,
                    stop - 1 if self.config.timestep_spacing == "linspace" else stop,
                    num_inference_steps,
                    endpoint=self.config.timestep_spacing == "linspace",
                )[::-1]
            elif self.config.timestep_spacing == "trailing":
                timesteps = (
                    np.arange(
                        self.config.num_train_timesteps, 0, -self.config.num_train_timesteps / num_inference_steps
                    )
                    - 1
                )
            else:
                raise ValueError(
                    f"`{self.config.timestep_spacing}` timestep spacing is not supported."
                    "Choose one of 'linspace', 'leading' or 'trailing'."
                )
            timesteps = timesteps.round().astype(np.int64).copy()

        self.num_inference_steps = num_inference_steps
        timesteps += self.config.steps_offset
        self.timesteps = torch.from_numpy(timesteps).to(device)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.scale_model_input
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[VDMSchedulerOutput, Tuple]:
        """
        Performs a single step of the diffusion process, computing the previous sample and optionally the predicted
        original sample based on the model output and current timestep.

        Args:
            model_output (torch.Tensor): The output from the diffusion model, typically noise predictions.
            timestep (int, float, torch.Tensor): Current timestep in the diffusion process.
            sample (torch.Tensor): The current sample at timestep `t`.
            generator (torch.Generator, *optional*): Generator for random numbers, used for adding noise.
            return_dict (bool): If True, returns a `VDMSchedulerOutput` object; otherwise, returns a tuple.

        Returns:
            VDMSchedulerOutput or Tuple: Depending on `return_dict`, returns either a data class containing the
            previous sample and predicted original sample, or just the previous sample as a tuple.
        """
        # Based on https://github.com/addtt/variational-diffusion-models/blob/main/vdm.py#L29

        if isinstance(timestep, (int, float)):
            timestep = torch.tensor(
                timestep, dtype=torch.float32 if isinstance(timestep, float) else torch.int64, device=sample.device
            )

        if not timestep.is_floating_point():
            if not self.config.num_train_timesteps:
                raise TypeError("Discrete timesteps require `self.config.num_train_timesteps` to be set.")
            timestep = timestep / self.config.num_train_timesteps  # Normalize to [0, 1]
        prev_timestep = (timestep - 1 / len(self)).clamp(0, 1)

        # 1. Compute current and previous alpha and sigma values
        log_snr = self.log_snr(timestep)
        prev_log_snr = self.log_snr(prev_timestep)

        # Allow for batched inputs
        if timestep.ndim > 0:
            log_snr = log_snr.view(timestep.size(0), *((1,) * (sample.ndim - 1)))
            prev_log_snr = prev_log_snr.view(timestep.size(0), *((1,) * (sample.ndim - 1)))

        alpha, sigma = torch.sigmoid(log_snr), torch.sigmoid(-log_snr)
        prev_alpha, prev_sigma = torch.sigmoid(prev_log_snr), torch.sigmoid(-prev_log_snr)

        # 2. Compute predicted original sample x_0
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(sigma) * model_output) / torch.sqrt(alpha)  # Sec. 3.4, eq. 10
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = torch.sqrt(alpha) * sample - torch.sqrt(sigma) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                f" `v_prediction`  for the {self.__class__.__name__}."
            )
            

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Computed predicted previous sample x_{t-1}
        c = -torch.expm1(log_snr - prev_log_snr)
        if self.config.thresholding or self.config.clip_sample or self.config.prediction_type != "epsilon":
            pred_prev_sample = torch.sqrt(prev_alpha) * (
                sample * (1 - c) / torch.sqrt(alpha) + c * pred_original_sample
            )
        else:
            pred_prev_sample = torch.sqrt(prev_alpha / alpha) * (sample - c * torch.sqrt(sigma) * model_output)

        # 5. (Maybe) add noise
        noise_scale = torch.sqrt(prev_sigma * c)  # Becomes 0 for prev_timestep = 0
        if torch.any(noise_scale > 0):
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            pred_prev_sample += noise_scale * noise

        if not return_dict:
            return (pred_prev_sample,)

        return VDMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the original samples according to the noise schedule and the specified timesteps.

        This method calculates the noisy samples by combining the original samples with Gaussian noise scaled according
        to the time-dependent noise levels dictated by the signal-to-noise ratio.

        Args:
            original_samples (torch.Tensor): The original samples from the data distribution before noise is added.
            noise (torch.Tensor): Gaussian noise to be added to the samples.
            timesteps (torch.Tensor): Timesteps at which the samples are processed.

        Returns:
            torch.Tensor: The noisy samples after adding scaled Gaussian noise according to the SNR.
        """
        gamma = self.log_snr(timesteps).to(original_samples.device)
        gamma = gamma.view(timesteps.size(0), *((1,) * (original_samples.ndim - 1)))

        sqrt_alpha_prod = torch.sqrt(torch.sigmoid(gamma))
        sqrt_one_minus_alpha_prod = torch.sqrt(torch.sigmoid(-gamma))  # sqrt(sigma)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        gamma = self.log_snr(timesteps).to(sample.device)
        gamma = gamma.view(timesteps.size(0), *((1,) * (sample.ndim - 1)))

        sqrt_alpha_prod = torch.sqrt(torch.sigmoid(gamma))
        sqrt_one_minus_alpha_prod = torch.sqrt(torch.sigmoid(-gamma))  # sqrt(sigma)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self) -> int:
        """Returns the number of inference steps or the number of training timesteps or 1000, whichever is set."""
        return self.num_inference_steps or self.config.num_train_timesteps or 1000
