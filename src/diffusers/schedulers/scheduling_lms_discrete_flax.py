# Copyright 2022 Katherine Crowson and The HuggingFace Team. All rights reserved.
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

import flax
import jax.numpy as jnp
from scipy import integrate

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    _FLAX_COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    broadcast_to_shape_from_left,
)


@flax.struct.dataclass
class LMSDiscreteSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None
    sigmas: Optional[jnp.ndarray] = None
    derivatives: jnp.ndarray = jnp.array([])

    @classmethod
    def create(cls, num_train_timesteps: int, sigmas: jnp.ndarray):
        return cls(timesteps=jnp.arange(0, num_train_timesteps)[::-1], sigmas=sigmas)


@dataclass
class FlaxLMSSchedulerOutput(FlaxSchedulerOutput):
    state: LMSDiscreteSchedulerState


class FlaxLMSDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Linear Multistep Scheduler for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`jnp.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
    """

    _compatibles = _FLAX_COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS.copy()

    @property
    def has_state(self):
        return True

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[jnp.ndarray] = None,
    ):
        if trained_betas is not None:
            self.betas = jnp.asarray(trained_betas)
        elif beta_schedule == "linear":
            self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps, dtype=jnp.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=jnp.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)

    def create_state(self):
        self.state = LMSDiscreteSchedulerState.create(
            num_train_timesteps=self.config.num_train_timesteps,
            sigmas=((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5,
        )

    def scale_model_input(self, state: LMSDiscreteSchedulerState, sample: jnp.ndarray, timestep: int) -> jnp.ndarray:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.

        Args:
            state (`LMSDiscreteSchedulerState`):
                the `FlaxLMSDiscreteScheduler` state data class instance.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            timestep (`int`):
                current discrete timestep in the diffusion chain.

        Returns:
            `jnp.ndarray`: scaled input sample
        """
        (step_index,) = jnp.where(scheduler_state.timesteps == timestep, size=1)
        sigma = scheduler_state.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def get_lms_coefficient(self, state, order, t, current_order):
        """
        Compute a linear multistep coefficient.

        Args:
            order (TODO):
            t (TODO):
            current_order (TODO):
        """

        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - state.sigmas[t - k]) / (state.sigmas[t - current_order] - state.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(lms_derivative, state.sigmas[t], state.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def set_timesteps(
        self, state: LMSDiscreteSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> LMSDiscreteSchedulerState:
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`LMSDiscreteSchedulerState`):
                the `FlaxLMSDiscreteScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        timesteps = jnp.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=jnp.float32)

        low_idx = jnp.floor(timesteps).astype(int)
        high_idx = jnp.ceil(timesteps).astype(int)
        frac = jnp.mod(timesteps, 1.0)
        sigmas = jnp.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0])]).astype(jnp.float32)

        return state.replace(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps.astype(int),
            derivatives=jnp.array([]),
            sigmas=sigmas,
        )

    def step(
        self,
        state: LMSDiscreteSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        order: int = 4,
        return_dict: bool = True,
    ) -> Union[FlaxLMSSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`LMSDiscreteSchedulerState`): the `FlaxLMSDiscreteScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            order: coefficient for multi-step inference.
            return_dict (`bool`): option for returning tuple rather than FlaxLMSSchedulerOutput class

        Returns:
            [`FlaxLMSSchedulerOutput`] or `tuple`: [`FlaxLMSSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        sigma = state.sigmas[timestep]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma
        state = state.replace(derivatives=jnp.append(state.derivatives, derivative))
        if len(state.derivatives) > order:
            state = state.replace(derivatives=jnp.delete(state.derivatives, 0))

        # 3. Compute linear multistep coefficients
        order = min(timestep + 1, order)
        lms_coeffs = [self.get_lms_coefficient(state, order, timestep, curr_order) for curr_order in range(order)]

        # 4. Compute previous sample based on the derivatives path
        prev_sample = sample + sum(
            coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(state.derivatives))
        )

        if not return_dict:
            return (prev_sample, state)

        return FlaxLMSSchedulerOutput(prev_sample=prev_sample, state=state)

    def add_noise(
        self,
        state: LMSDiscreteSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        sigma = state.sigmas[timesteps].flatten()
        sigma = broadcast_to_shape_from_left(sigma, noise.shape)

        noisy_samples = original_samples + noise * sigma

        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
