# Copyright 2024 Google Brain and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from jax import random

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import FlaxSchedulerMixin, FlaxSchedulerOutput, broadcast_to_shape_from_left


@flax.struct.dataclass
class ScoreSdeVeSchedulerState:
    # setable values
    timesteps: Optional[jnp.ndarray] = None
    discrete_sigmas: Optional[jnp.ndarray] = None
    sigmas: Optional[jnp.ndarray] = None

    @classmethod
    def create(cls):
        return cls()


@dataclass
class FlaxSdeVeOutput(FlaxSchedulerOutput):
    """
    Output class for the ScoreSdeVeScheduler's step function output.

    Args:
        state (`ScoreSdeVeSchedulerState`):
        prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Mean averaged `prev_sample`. Same as `prev_sample`, only mean-averaged over previous timesteps.
    """

    state: ScoreSdeVeSchedulerState
    prev_sample: jnp.ndarray
    prev_sample_mean: Optional[jnp.ndarray] = None


class FlaxScoreSdeVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    The variance exploding stochastic differential equation (SDE) scheduler.

    For more information, see the original paper: https://huggingface.co/papers/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        snr (`float`):
            coefficient weighting the step from the model_output sample (from the network) to the random noise.
        sigma_min (`float`):
                initial noise scale for sigma sequence in sampling procedure. The minimum sigma should mirror the
                distribution of the data.
        sigma_max (`float`): maximum value used for the range of continuous timesteps passed into the model.
        sampling_eps (`float`): the end value of sampling, where timesteps decrease progressively from 1 to
        epsilon.
        correct_steps (`int`): number of correction steps performed on a produced sample.
    """

    @property
    def has_state(self):
        return True

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
    ):
        pass

    def create_state(self):
        state = ScoreSdeVeSchedulerState.create()
        return self.set_sigmas(
            state,
            self.config.num_train_timesteps,
            self.config.sigma_min,
            self.config.sigma_max,
            self.config.sampling_eps,
        )

    def set_timesteps(
        self, state: ScoreSdeVeSchedulerState, num_inference_steps: int, shape: Tuple = (), sampling_eps: float = None
    ) -> ScoreSdeVeSchedulerState:
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sampling_eps (`float`, optional):
                final timestep value (overrides value given at Scheduler instantiation).

        """
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps

        timesteps = jnp.linspace(1, sampling_eps, num_inference_steps)
        return state.replace(timesteps=timesteps)

    def set_sigmas(
        self,
        state: ScoreSdeVeSchedulerState,
        num_inference_steps: int,
        sigma_min: float = None,
        sigma_max: float = None,
        sampling_eps: float = None,
    ) -> ScoreSdeVeSchedulerState:
        """
        Sets the noise scales used for the diffusion chain. Supporting function to be run before inference.

        The sigmas control the weight of the `drift` and `diffusion` components of sample update.

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                initial noise scale value (overrides value given at Scheduler instantiation).
            sigma_max (`float`, optional):
                final noise scale value (overrides value given at Scheduler instantiation).
            sampling_eps (`float`, optional):
                final timestep value (overrides value given at Scheduler instantiation).
        """
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if state.timesteps is None:
            state = self.set_timesteps(state, num_inference_steps, sampling_eps)

        discrete_sigmas = jnp.exp(jnp.linspace(jnp.log(sigma_min), jnp.log(sigma_max), num_inference_steps))
        sigmas = jnp.array([sigma_min * (sigma_max / sigma_min) ** t for t in state.timesteps])

        return state.replace(discrete_sigmas=discrete_sigmas, sigmas=sigmas)

    def get_adjacent_sigma(self, state, timesteps, t):
        return jnp.where(timesteps == 0, jnp.zeros_like(t), state.discrete_sigmas[timesteps - 1])

    def step_pred(
        self,
        state: ScoreSdeVeSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        key: jax.Array,
        return_dict: bool = True,
    ) -> Union[FlaxSdeVeOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than FlaxSdeVeOutput class

        Returns:
            [`FlaxSdeVeOutput`] or `tuple`: [`FlaxSdeVeOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if state.timesteps is None:
            raise ValueError(
                "`state.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * jnp.ones(
            sample.shape[0],
        )
        timesteps = (timestep * (len(state.timesteps) - 1)).long()

        sigma = state.discrete_sigmas[timesteps]
        adjacent_sigma = self.get_adjacent_sigma(state, timesteps, timestep)
        drift = jnp.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        diffusion = diffusion.flatten()
        diffusion = broadcast_to_shape_from_left(diffusion, sample.shape)
        drift = drift - diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        key = random.split(key, num=1)
        noise = random.normal(key=key, shape=sample.shape)
        prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = prev_sample_mean + diffusion * noise  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean, state)

        return FlaxSdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean, state=state)

    def step_correct(
        self,
        state: ScoreSdeVeSchedulerState,
        model_output: jnp.ndarray,
        sample: jnp.ndarray,
        key: jax.Array,
        return_dict: bool = True,
    ) -> Union[FlaxSdeVeOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. This is often run repeatedly
        after making the prediction for the previous timestep.

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than FlaxSdeVeOutput class

        Returns:
            [`FlaxSdeVeOutput`] or `tuple`: [`FlaxSdeVeOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if state.timesteps is None:
            raise ValueError(
                "`state.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        key = random.split(key, num=1)
        noise = random.normal(key=key, shape=sample.shape)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = jnp.linalg.norm(model_output)
        noise_norm = jnp.linalg.norm(noise)
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * jnp.ones(sample.shape[0])

        # compute corrected sample: model_output term and noise term
        step_size = step_size.flatten()
        step_size = broadcast_to_shape_from_left(step_size, sample.shape)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample, state)

        return FlaxSdeVeOutput(prev_sample=prev_sample, state=state)

    def __len__(self):
        return self.config.num_train_timesteps
