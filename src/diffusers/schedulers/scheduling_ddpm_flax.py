# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    add_noise_common,
    get_velocity_common,
)


@flax.struct.dataclass
class DDPMSchedulerState:
    common: CommonSchedulerState

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    num_inference_steps: Optional[int] = None

    @classmethod
    def create(cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray):
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps)


@dataclass
class FlaxDDPMSchedulerOutput(FlaxSchedulerOutput):
    state: DDPMSchedulerState


class FlaxDDPMScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://huggingface.co/papers/2006.11239

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        variance_type (`str`):
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between -1 and 1 for numerical stability.
        prediction_type (`str`, default `epsilon`):
            indicates whether the model predicts the noise (epsilon), or the samples. One of `epsilon`, `sample`.
            `v-prediction` is not supported for this scheduler.
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
        trained_betas: Optional[jnp.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype

    def create_state(self, common: Optional[CommonSchedulerState] = None) -> DDPMSchedulerState:
        if common is None:
            common = CommonSchedulerState.create(self)

        # standard deviation of the initial noise distribution
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]

        return DDPMSchedulerState.create(
            common=common,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )

    def scale_model_input(
        self, state: DDPMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
        return sample

    def set_timesteps(
        self, state: DDPMSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> DDPMSchedulerState:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`DDIMSchedulerState`):
                the `FlaxDDPMScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        step_ratio = self.config.num_train_timesteps // num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # rounding to avoid issues when num_inference_step is power of 3
        timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round()[::-1]

        return state.replace(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
        )

    def _get_variance(self, state: DDPMSchedulerState, t, predicted_variance=None, variance_type=None):
        alpha_prod_t = state.common.alphas_cumprod[t]
        alpha_prod_t_prev = jnp.where(t > 0, state.common.alphas_cumprod[t - 1], jnp.array(1.0, dtype=self.dtype))

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://huggingface.co/papers/2006.11239)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * state.common.betas[t]

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = jnp.clip(variance, a_min=1e-20)
        # for rl-diffuser https://huggingface.co/papers/2205.09991
        elif variance_type == "fixed_small_log":
            variance = jnp.log(jnp.clip(variance, a_min=1e-20))
        elif variance_type == "fixed_large":
            variance = state.common.betas[t]
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = jnp.log(state.common.betas[t])
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = variance
            max_log = state.common.betas[t]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        state: DDPMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        key: Optional[jax.Array] = None,
        return_dict: bool = True,
    ) -> Union[FlaxDDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`DDPMSchedulerState`): the `FlaxDDPMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            key (`jax.Array`): a PRNG key.
            return_dict (`bool`): option for returning tuple rather than FlaxDDPMSchedulerOutput class

        Returns:
            [`FlaxDDPMSchedulerOutput`] or `tuple`: [`FlaxDDPMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        t = timestep

        if key is None:
            key = jax.random.key(0)

        if (
            len(model_output.shape) > 1
            and model_output.shape[1] == sample.shape[1] * 2
            and self.config.variance_type in ["learned", "learned_range"]
        ):
            model_output, predicted_variance = jnp.split(model_output, sample.shape[1], axis=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = state.common.alphas_cumprod[t]
        alpha_prod_t_prev = jnp.where(t > 0, state.common.alphas_cumprod[t - 1], jnp.array(1.0, dtype=self.dtype))
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://huggingface.co/papers/2006.11239
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` "
                " for the FlaxDDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = jnp.clip(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://huggingface.co/papers/2006.11239
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * state.common.betas[t]) / beta_prod_t
        current_sample_coeff = state.common.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://huggingface.co/papers/2006.11239
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        def random_variance():
            split_key = jax.random.split(key, num=1)[0]
            noise = jax.random.normal(split_key, shape=model_output.shape, dtype=self.dtype)
            return (self._get_variance(state, t, predicted_variance=predicted_variance) ** 0.5) * noise

        variance = jnp.where(t > 0, random_variance(), jnp.zeros(model_output.shape, dtype=self.dtype))

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample, state)

        return FlaxDDPMSchedulerOutput(prev_sample=pred_prev_sample, state=state)

    def add_noise(
        self,
        state: DDPMSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        return add_noise_common(state.common, original_samples, noise, timesteps)

    def get_velocity(
        self,
        state: DDPMSchedulerState,
        sample: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        return get_velocity_common(state.common, sample, noise, timesteps)

    def __len__(self):
        return self.config.num_train_timesteps
