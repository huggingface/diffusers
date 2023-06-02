# Copyright 2023 NVIDIA and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging, randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


@dataclass
class CMStochasticIterativeSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        derivative (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Derivative of predicted original image sample (x_0).
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    # derivative: torch.FloatTensor
    # pred_original_sample: Optional[torch.FloatTensor] = None


class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya. "Consistency Models"
    https://arxiv.org/pdf/2303.01469

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details on the parameters, see the original paper's Appendix E.: "Elucidating the Design Space of
    Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364. The grid search values used to find the
    optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        prediction_type (`str`, default `"epsilon"`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        interpolation_type (`str`, default `"linear"`, optional):
            interpolation type to compute intermediate sigmas for the scheduler denoising steps. Should be one of
            [`"linear"`, `"log_linear"`].
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
             This parameter controls whether to use Karras sigmas (Karras et al. (2022) scheme) for step sizes in the
             noise schedule during the sampling process. If True, the sigmas will be determined according to a sequence
             of noise levels {σi} as defined in Equation (5) of the paper https://arxiv.org/pdf/2206.00364.pdf.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 40,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        timesteps: Optional[np.ndarray] = None,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.num_inference_steps = None
        self.timesteps = timesteps
        self.is_scale_input_called = False

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        return indices.item()

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        # Note: timesteps are expected to be increasing rather than decreasing, following original implementation
        if self.timesteps is None:
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)
        else:
            timesteps = self.timesteps.astype(np.float32)

        # Map timesteps to Karras sigmas directly for multistep sampling
        # See https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L675
        num_train_timesteps = self.config.num_train_timesteps
        # Append num_train_timesteps - 1 so sigmas[-1] == sigma_min
        ramp = np.append(timesteps, [num_train_timesteps - 1]) / (num_train_timesteps - 1)
        sigmas = self._convert_to_karras(ramp)

        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        if str(device).startswith("mps"):
            # mps does not support float64
            self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
        else:
            self.timesteps = torch.from_numpy(timesteps).to(device=device)

    # Modified from diffusers.schedulers.scheduling_euler_discrete._convert_to_karras
    # Use self.rho instead of hardcoded 7.0 for rho, sigma_min/max from config, configurable ramp function
    def _convert_to_karras(self, ramp):
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = self.config.sigma_min
        sigma_max: float = self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # TODO: add_noise meant to be called during training (forward diffusion process)
    # TODO: need to check if this corresponds to noise added during training
    # TODO: may want multiple add_noise-type methods for CD, CT
    # Copied from diffusers.schedulers.scheduling_euler_discrete.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def add_noise_to_input(
        self,
        sample: torch.FloatTensor,
        sigma: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.FloatTensor, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i. TODO Args:
        """
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max
        step_idx = (self.sigmas == sigma).nonzero().item()
        sigma_hat = self.sigmas[step_idx + 1].clamp(min=sigma_min, max=sigma_max)

        # sample z ~ N(0, s_noise^2 * I)
        z = self.config.s_noise * randn_tensor(sample.shape, generator=generator, device=sample.device)

        # tau = sigma_hat, eps = sigma_min
        sample_hat = sample + ((sigma_hat**2 - sigma_min**2) ** 0.5 * z)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[CMStochasticIterativeSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            s_churn (`float`)
            s_tmin  (`float`)
            s_tmax  (`float`)
            s_noise (`float`)
            generator (`torch.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.CMStochasticIterativeSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.CMStochasticIterativeSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    f" `{self.__class__}.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        step_index = self.index_for_timestep(timestep)

        # sigma_next corresponds to next_t in original implementation
        sigma_next = self.sigmas[step_index + 1]

        # 1. Sample z ~ N(0, s_noise^2 * I)
        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )
        z = noise * s_noise

        sigma_hat = sigma_next.clamp(min=sigma_min, max=sigma_max)

        # 2. Return noisy sample
        # tau = sigma_hat, eps = sigma_min
        prev_sample = model_output + z * (sigma_hat**2 - sigma_min**2) ** 0.5

        if not return_dict:
            return (prev_sample,)

        return CMStochasticIterativeSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
