# Copyright 2025 Katherine Crowson and The HuggingFace Team. All rights reserved.
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
from typing import List, Literal, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerDiscrete
class EDMEulerSchedulerOutput(BaseOutput):
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


class EDMEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements the Euler scheduler in EDM formulation as presented in Karras et al. 2022 [1].

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://huggingface.co/papers/2206.00364

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        sigma_min (`float`, *optional*, defaults to `0.002`):
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the EDM paper [1]; a reasonable
            range is [0, 10].
        sigma_max (`float`, *optional*, defaults to `80.0`):
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the EDM paper [1]; a reasonable
            range is [0.2, 80.0].
        sigma_data (`float`, *optional*, defaults to `0.5`):
            The standard deviation of the data distribution. This is set to 0.5 in the EDM paper [1].
        sigma_schedule (`Literal["karras", "exponential"]`, *optional*, defaults to `"karras"`):
            Sigma schedule to compute the `sigmas`. By default, we use the schedule introduced in the EDM paper
            (https://huggingface.co/papers/2206.00364). The `"exponential"` schedule was incorporated in this model:
            https://huggingface.co/stabilityai/cosxl.
        num_train_timesteps (`int`, *optional*, defaults to `1000`):
            The number of diffusion steps to train the model.
        prediction_type (`Literal["epsilon", "v_prediction"]`, *optional*, defaults to `"epsilon"`):
            Prediction type of the scheduler function. `"epsilon"` predicts the noise of the diffusion process, and
            `"v_prediction"` (see section 2.4 of [Imagen Video](https://huggingface.co/papers/2210.02303) paper).
        rho (`float`, *optional*, defaults to `7.0`):
            The rho parameter used for calculating the Karras sigma schedule, which is set to 7.0 in the EDM paper [1].
        final_sigmas_type (`Literal["zero", "sigma_min"]`, *optional*, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `"zero"`, the final sigma is set to 0.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        sigma_schedule: Literal["karras", "exponential"] = "karras",
        num_train_timesteps: int = 1000,
        prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
        rho: float = 7.0,
        final_sigmas_type: Literal["zero", "sigma_min"] = "zero",
    ) -> None:
        if sigma_schedule not in ["karras", "exponential"]:
            raise ValueError(f"Wrong value for provided for `{sigma_schedule=}`.`")

        # setable values
        self.num_inference_steps = None

        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.arange(num_train_timesteps + 1, dtype=sigmas_dtype) / num_train_timesteps
        if sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(sigmas)
        elif sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(sigmas)
        sigmas = sigmas.to(torch.float32)

        self.timesteps = self.precondition_noise(sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = sigmas[-1]
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        self.sigmas = torch.cat([sigmas, torch.full((1,), fill_value=sigma_last, device=sigmas.device)])

        self.is_scale_input_called = False

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self) -> float:
        """
        Return the standard deviation of the initial noise distribution.

        Returns:
            `float`:
                The initial noise sigma value computed as `(sigma_max**2 + 1) ** 0.5`.
        """
        return (self.config.sigma_max**2 + 1) ** 0.5

    @property
    def step_index(self) -> Optional[int]:
        """
        Return the index counter for the current timestep. The index will increase by 1 after each scheduler step.

        Returns:
            `int` or `None`:
                The current step index, or `None` if not yet initialized.
        """
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        """
        Return the index for the first timestep. This should be set from the pipeline with the `set_begin_index`
        method.

        Returns:
            `int` or `None`:
                The begin index, or `None` if not yet set.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def precondition_inputs(self, sample: torch.Tensor, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Precondition the input sample by scaling it according to the EDM formulation.

        Args:
            sample (`torch.Tensor`):
                The input sample tensor to precondition.
            sigma (`float` or `torch.Tensor`):
                The current sigma (noise level) value.

        Returns:
            `torch.Tensor`:
                The scaled input sample.
        """
        c_in = self._get_conditioning_c_in(sigma)
        scaled_sample = sample * c_in
        return scaled_sample

    def precondition_noise(self, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Precondition the noise level by applying a logarithmic transformation.

        Args:
            sigma (`float` or `torch.Tensor`):
                The sigma (noise level) value to precondition.

        Returns:
            `torch.Tensor`:
                The preconditioned noise value computed as `0.25 * log(sigma)`.
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma])

        c_noise = 0.25 * torch.log(sigma)

        return c_noise

    def precondition_outputs(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Precondition the model outputs according to the EDM formulation.

        Args:
            sample (`torch.Tensor`):
                The input sample tensor.
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sigma (`float` or `torch.Tensor`):
                The current sigma (noise level) value.

        Returns:
            `torch.Tensor`:
                The denoised sample computed by combining the skip connection and output scaling.
        """
        sigma_data = self.config.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

        if self.config.prediction_type == "epsilon":
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        else:
            raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")

        denoised = c_skip * sample + c_out * model_output

        return denoised

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Scale the denoising model input to match the Euler algorithm. Ensures interchangeability with schedulers that
        need to scale the denoising model input depending on the current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample tensor.
            timestep (`float` or `torch.Tensor`):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = self.precondition_inputs(sample, sigma)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[torch.Tensor, List[float]]] = None,
    ) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`torch.Tensor` or `List[float]`, *optional*):
                Custom sigmas to use for the denoising process. If not defined, the default behavior when
                `num_inference_steps` is passed will be used.
        """
        self.num_inference_steps = num_inference_steps

        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        if sigmas is None:
            sigmas = torch.linspace(0, 1, self.num_inference_steps, dtype=sigmas_dtype)
        elif isinstance(sigmas, float):
            sigmas = torch.tensor(sigmas, dtype=sigmas_dtype)
        else:
            sigmas = sigmas.to(sigmas_dtype)
        if self.config.sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(sigmas)
        elif self.config.sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(sigmas)
        sigmas = sigmas.to(dtype=torch.float32, device=device)

        self.timesteps = self.precondition_noise(sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = sigmas[-1]
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        self.sigmas = torch.cat([sigmas, torch.full((1,), fill_value=sigma_last, device=sigmas.device)])
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    # Taken from https://github.com/crowsonkb/k-diffusion/blob/686dbad0f39640ea25c8a8c6a6e56bb40eacefa2/k_diffusion/sampling.py#L17
    def _compute_karras_sigmas(
        self,
        ramp: torch.Tensor,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Construct the noise schedule of [Karras et al. (2022)](https://huggingface.co/papers/2206.00364).

        Args:
            ramp (`torch.Tensor`):
                A tensor of values in [0, 1] representing the interpolation positions.
            sigma_min (`float`, *optional*):
                Minimum sigma value. If `None`, uses `self.config.sigma_min`.
            sigma_max (`float`, *optional*):
                Maximum sigma value. If `None`, uses `self.config.sigma_max`.

        Returns:
            `torch.Tensor`:
                The computed Karras sigma schedule.
        """
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _compute_exponential_sigmas(
        self,
        ramp: torch.Tensor,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute the exponential sigma schedule. Implementation closely follows k-diffusion:
        https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/sampling.py#L26

        Args:
            ramp (`torch.Tensor`):
                A tensor of values representing the interpolation positions.
            sigma_min (`float`, *optional*):
                Minimum sigma value. If `None`, uses `self.config.sigma_min`.
            sigma_max (`float`, *optional*):
                Maximum sigma value. If `None`, uses `self.config.sigma_max`.

        Returns:
            `torch.Tensor`:
                The computed exponential sigma schedule.
        """
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp)).exp().flip(0)
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep
    def index_for_timestep(
        self, timestep: Union[float, torch.Tensor], schedule_timesteps: Optional[torch.Tensor] = None
    ) -> int:
        """
        Find the index of a given timestep in the timestep schedule.

        Args:
            timestep (`float` or `torch.Tensor`):
                The timestep value to find in the schedule.
            schedule_timesteps (`torch.Tensor`, *optional*):
                The timestep schedule to search in. If `None`, uses `self.timesteps`.

        Returns:
            `int`:
                The index of the timestep in the schedule. For the very first step, returns the second index if
                multiple matches exist to avoid skipping a sigma when starting mid-schedule (e.g., for image-to-image).
        """
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep: Union[float, torch.Tensor]) -> None:
        """
        Initialize the step index for the scheduler based on the given timestep.

        Args:
            timestep (`float` or `torch.Tensor`):
                The current timestep to initialize the step index from.
        """
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        pred_original_sample: Optional[torch.Tensor] = None,
    ) -> Union[EDMEulerSchedulerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`float` or `torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`, *optional*, defaults to `0.0`):
                The amount of stochasticity to add at each step. Higher values add more noise.
            s_tmin (`float`, *optional*, defaults to `0.0`):
                The minimum sigma threshold below which no noise is added.
            s_tmax (`float`, *optional*, defaults to `float("inf")`):
                The maximum sigma threshold above which no noise is added.
            s_noise (`float`, *optional*, defaults to `1.0`):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator for reproducibility.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return an [`~schedulers.scheduling_edm_euler.EDMEulerSchedulerOutput`] or tuple.
            pred_original_sample (`torch.Tensor`, *optional*):
                The predicted denoised sample from a previous step. If provided, skips recomputation.

        Returns:
            [`~schedulers.scheduling_edm_euler.EDMEulerSchedulerOutput`] or `tuple`:
                If `return_dict` is `True`, an [`~schedulers.scheduling_edm_euler.EDMEulerSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the previous sample tensor and the
                second element is the predicted original sample tensor.
        """

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EDMEulerScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                device=model_output.device,
                generator=generator,
            )
            eps = noise * s_noise
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if pred_original_sample is None:
            pred_original_sample = self.precondition_outputs(sample, model_output, sigma_hat)

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return EDMEulerSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples according to the noise schedule at the specified timesteps.

        Args:
            original_samples (`torch.Tensor`):
                The original samples to which noise will be added.
            noise (`torch.Tensor`):
                The noise tensor to add to the original samples.
            timesteps (`torch.Tensor`):
                The timesteps at which to add noise, determining the noise level from the schedule.

        Returns:
            `torch.Tensor`:
                The noisy samples with added noise scaled according to the timestep schedule.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def _get_conditioning_c_in(self, sigma: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Compute the input conditioning factor for the EDM formulation.

        Args:
            sigma (`float` or `torch.Tensor`):
                The current sigma (noise level) value.

        Returns:
            `float` or `torch.Tensor`:
                The input conditioning factor `c_in`.
        """
        c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        return c_in

    def __len__(self) -> int:
        return self.config.num_train_timesteps
