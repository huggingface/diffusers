# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CMStochasticIterativeSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):
    """
    Multistep and onestep sampling for consistency models.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 40):
            The number of diffusion steps to train the model.
        sigma_min (`float`, defaults to 0.002):
            Minimum noise magnitude in the sigma schedule. Defaults to 0.002 from the original implementation.
        sigma_max (`float`, defaults to 80.0):
            Maximum noise magnitude in the sigma schedule. Defaults to 80.0 from the original implementation.
        sigma_data (`float`, defaults to 0.5):
            The standard deviation of the data distribution from the EDM
            [paper](https://huggingface.co/papers/2206.00364). Defaults to 0.5 from the original implementation.
        s_noise (`float`, defaults to 1.0):
            The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
            1.011]. Defaults to 1.0 from the original implementation.
        rho (`float`, defaults to 7.0):
            The parameter for calculating the Karras sigma schedule from the EDM
            [paper](https://huggingface.co/papers/2206.00364). Defaults to 7.0 from the original implementation.
        clip_denoised (`bool`, defaults to `True`):
            Whether to clip the denoised outputs to `(-1, 1)`.
        timesteps (`List` or `np.ndarray` or `torch.Tensor`, *optional*):
            An explicit timestep schedule that can be optionally specified. The timesteps are expected to be in
            increasing order.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 40,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        s_noise: float = 1.0,
        rho: float = 7.0,
        clip_denoised: bool = True,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        ramp = np.linspace(0, 1, num_train_timesteps)
        sigmas = self._convert_to_karras(ramp)
        timesteps = self.sigma_to_t(sigmas)

        # setable values
        self.num_inference_steps = None
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps)
        self.custom_timesteps = False
        self.is_scale_input_called = False
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        return indices.item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increae 1 after each scheduler step.
        """
        return self._step_index

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Scales the consistency model input by `(sigma**2 + sigma_data**2) ** 0.5`.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`float` or `torch.FloatTensor`):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Get sigma corresponding to timestep
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        sample = sample / ((sigma**2 + self.config.sigma_data**2) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def sigma_to_t(self, sigmas: Union[float, np.ndarray]):
        """
        Gets scaled timesteps from the Karras sigmas for input to the consistency model.

        Args:
            sigmas (`float` or `np.ndarray`):
                A single Karras sigma or an array of Karras sigmas.

        Returns:
            `float` or `np.ndarray`:
                A scaled input timestep or scaled input timestep array.
        """
        if not isinstance(sigmas, np.ndarray):
            sigmas = np.array(sigmas, dtype=np.float64)

        timesteps = 1000 * 0.25 * np.log(sigmas + 1e-44)

        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        # Follow DDPMScheduler custom timesteps logic
        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            self.custom_timesteps = False

        # Map timesteps to Karras sigmas directly for multistep sampling
        # See https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L675
        num_train_timesteps = self.config.num_train_timesteps
        ramp = timesteps[::-1].copy()
        ramp = ramp / (num_train_timesteps - 1)
        sigmas = self._convert_to_karras(ramp)
        timesteps = self.sigma_to_t(sigmas)

        sigmas = np.concatenate([sigmas, [self.sigma_min]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)

        if str(device).startswith("mps"):
            # mps does not support float64
            self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
        else:
            self.timesteps = torch.from_numpy(timesteps).to(device=device)

        self._step_index = None

    # Modified _convert_to_karras implementation that takes in ramp as argument
    def _convert_to_karras(self, ramp):
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = self.config.sigma_min
        sigma_max: float = self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def get_scalings(self, sigma):
        sigma_data = self.config.sigma_data

        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def get_scalings_for_boundary_condition(self, sigma):
        """
        Gets the scalings used in the consistency model parameterization (from Appendix C of the
        [paper](https://huggingface.co/papers/2303.01469)) to enforce boundary condition.

        <Tip>

        `epsilon` in the equations for `c_skip` and `c_out` is set to `sigma_min`.

        </Tip>

        Args:
            sigma (`torch.FloatTensor`):
                The current sigma in the Karras sigma schedule.

        Returns:
            `tuple`:
                A two-element tuple where `c_skip` (which weights the current sample) is the first element and `c_out`
                (which weights the consistency model output) is the second element.
        """
        sigma_min = self.config.sigma_min
        sigma_data = self.config.sigma_data

        c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
        c_out = (sigma - sigma_min) * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[CMStochasticIterativeSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            timestep (`float`):
                The current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
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

        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        if self.step_index is None:
            self._init_step_index(timestep)

        # sigma_next corresponds to next_t in original implementation
        sigma = self.sigmas[self.step_index]
        if self.step_index + 1 < self.config.num_train_timesteps:
            sigma_next = self.sigmas[self.step_index + 1]
        else:
            # Set sigma_next to sigma_min
            sigma_next = self.sigmas[-1]

        # Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition(sigma)

        # 1. Denoise model output using boundary conditions
        denoised = c_out * model_output + c_skip * sample
        if self.config.clip_denoised:
            denoised = denoised.clamp(-1, 1)

        # 2. Sample z ~ N(0, s_noise^2 * I)
        # Noise is not used for onestep sampling.
        if len(self.timesteps) > 1:
            noise = randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
            )
        else:
            noise = torch.zeros_like(model_output)
        z = noise * self.config.s_noise

        sigma_hat = sigma_next.clamp(min=sigma_min, max=sigma_max)

        # 3. Return noisy sample
        # tau = sigma_hat, eps = sigma_min
        prev_sample = denoised + z * (sigma_hat**2 - sigma_min**2) ** 0.5

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return CMStochasticIterativeSchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
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

    def __len__(self):
        return self.config.num_train_timesteps
