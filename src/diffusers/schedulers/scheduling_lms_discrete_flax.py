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

from dataclasses import dataclass

import flax
import jax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    broadcast_to_shape_from_left,
)


logger = logging.get_logger(__name__)


@flax.struct.dataclass
class LMSDiscreteSchedulerState:
    common: CommonSchedulerState

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    sigmas: jnp.ndarray
    num_inference_steps: int = None

    # running values
    derivatives: jnp.ndarray | None = None

    @classmethod
    def create(
        cls,
        common: CommonSchedulerState,
        init_noise_sigma: jnp.ndarray,
        timesteps: jnp.ndarray,
        sigmas: jnp.ndarray,
    ):
        return cls(
            common=common,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            sigmas=sigmas,
        )


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
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://huggingface.co/papers/2210.02303)
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            the `dtype` used for params and computation.
    """

    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    dtype: jnp.dtype

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
        trained_betas: jnp.ndarray | None = None,
        prediction_type: str = "epsilon",
        dtype: jnp.dtype = jnp.float32,
    ):
        logger.warning(
            "Flax classes are deprecated and will be removed in Diffusers v0.40.0. We "
            "recommend migrating to PyTorch classes or pinning your version of Diffusers."
        )
        self.dtype = dtype

    def create_state(self, common: CommonSchedulerState | None = None) -> LMSDiscreteSchedulerState:
        if common is None:
            common = CommonSchedulerState.create(self)

        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
        sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5

        # standard deviation of the initial noise distribution
        init_noise_sigma = sigmas.max()

        return LMSDiscreteSchedulerState.create(
            common=common,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            sigmas=sigmas,
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
        step_index = jnp.where(state.timesteps == timestep, jnp.arange(state.timesteps.shape[0]), 0).sum()
        sigma = state.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def get_lms_coefficient(self, state: LMSDiscreteSchedulerState, order, t, current_order):
        """
        Compute a linear multistep coefficient.

        Args:
            order (`int`):
                The order of the linear multistep method.
            t (`int`):
                The current step index in the inference schedule.
            current_order (`int`):
                The current order for which to compute the coefficient.
        """
        num_sigmas = state.sigmas.shape[0]
        num_integration_steps = 10

        def lms_derivative(tau):
            num_tau = tau.shape[0]
            mask_indices = jnp.broadcast_to(
                jnp.arange(num_sigmas).reshape(1, -1),
                (num_tau, num_sigmas),
            )
            greater_than = t - order + 1 <= mask_indices
            lower_than = mask_indices < t + 1
            not_same_value = mask_indices != t - current_order
            mask = greater_than & lower_than & not_same_value

            correct_coeffs = (tau.reshape(-1, 1) - state.sigmas.reshape(1, -1)) / (
                state.sigmas[t - current_order] - state.sigmas.reshape(1, -1) + 1e-5
            )
            coeffs = jnp.where(mask, correct_coeffs, jnp.ones_like(mask))
            return jnp.prod(coeffs, axis=1)

        x = jnp.linspace(state.sigmas[t], state.sigmas[t + 1], num_integration_steps)
        return jnp.trapezoid(lms_derivative(x), x=x, axis=0)

    def set_timesteps(
        self,
        state: LMSDiscreteSchedulerState,
        num_inference_steps: int,
        shape: tuple = (),
        max_order: int = 4,
    ) -> LMSDiscreteSchedulerState:
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`LMSDiscreteSchedulerState`):
                the `FlaxLMSDiscreteScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            max_order (`int`, defaults to `4`):
                The maximum multistep order. Used to pre-allocate the derivatives buffer for jittable inference.
        """

        timesteps = jnp.linspace(
            self.config.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=self.dtype,
        )

        low_idx = jnp.floor(timesteps).astype(jnp.int32)
        high_idx = jnp.ceil(timesteps).astype(jnp.int32)

        frac = jnp.mod(timesteps, 1.0)

        sigmas = ((1 - state.common.alphas_cumprod) / state.common.alphas_cumprod) ** 0.5
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])

        timesteps = timesteps.astype(jnp.int32)

        # initial running values
        derivatives = jnp.zeros((max_order,) + shape, dtype=self.dtype)

        return state.replace(
            timesteps=timesteps,
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
            derivatives=derivatives,
        )

    def step(
        self,
        state: LMSDiscreteSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        order: int = 4,
        return_dict: bool = True,
    ) -> FlaxLMSSchedulerOutput | tuple:
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
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        step_index = jnp.where(state.timesteps == timestep, jnp.arange(state.timesteps.shape[0]), 0).sum()
        sigma = state.sigmas[step_index] + 1e-5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative and maintain a fixed-size rolling buffer
        derivative = (sample - pred_original_sample) / sigma
        derivative = derivative.reshape(1, *derivative.shape).astype(self.dtype)
        state = state.replace(derivatives=jnp.concatenate([state.derivatives[1:], derivative], axis=0))

        # 3. Compute linear multistep coefficients and the previous sample based on the derivatives path
        effective_order = jnp.minimum(step_index + 1, order)
        prev_sample = jax.lax.fori_loop(
            0,
            order,
            lambda i, val: jnp.where(
                i < effective_order,
                val + self.get_lms_coefficient(state, effective_order, step_index, i) * state.derivatives[-(i + 1)],
                val,
            ),
            sample,
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
