# Copyright 2025 NVIDIA and The HuggingFace Team. All rights reserved.
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
import jax
import jax.numpy as jnp
from jax import random

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils_flax import FlaxSchedulerMixin


@flax.struct.dataclass
class KarrasVeSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None
    schedule: Optional[jnp.ndarray] = None  # sigma(t_i)

    @classmethod
    def create(cls):
        return cls()


@dataclass
class FlaxKarrasVeOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        derivative (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Derivative of predicted original image sample (x_0).
        state (`KarrasVeSchedulerState`): the `FlaxKarrasVeScheduler` state data class.
    """

    prev_sample: jnp.ndarray
    derivative: jnp.ndarray
    state: KarrasVeSchedulerState


class FlaxKarrasVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://huggingface.co/papers/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://huggingface.co/papers/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details on the parameters, see the original paper's Appendix E.: "Elucidating the Design Space of
    Diffusion-Based Generative Models." https://huggingface.co/papers/2206.00364. The grid search values used to find
    the optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.

    Args:
        sigma_min (`float`): minimum noise magnitude
        sigma_max (`float`): maximum noise magnitude
        s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
            A reasonable range is [1.000, 1.011].
        s_churn (`float`): the parameter controlling the overall amount of stochasticity.
            A reasonable range is [0, 100].
        s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
            A reasonable range is [0, 10].
        s_max (`float`): the end value of the sigma range where we add noise.
            A reasonable range is [0.2, 80].
    """

    @property
    def has_state(self):
        return True

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        s_noise: float = 1.007,
        s_churn: float = 80,
        s_min: float = 0.05,
        s_max: float = 50,
    ):
        pass

    def create_state(self):
        return KarrasVeSchedulerState.create()

    def set_timesteps(
        self, state: KarrasVeSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> KarrasVeSchedulerState:
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`KarrasVeSchedulerState`):
                the `FlaxKarrasVeScheduler` state data class.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

        """
        timesteps = jnp.arange(0, num_inference_steps)[::-1].copy()
        schedule = [
            (
                self.config.sigma_max**2
                * (self.config.sigma_min**2 / self.config.sigma_max**2) ** (i / (num_inference_steps - 1))
            )
            for i in timesteps
        ]

        return state.replace(
            num_inference_steps=num_inference_steps,
            schedule=jnp.array(schedule, dtype=jnp.float32),
            timesteps=timesteps,
        )

    def add_noise_to_input(
        self,
        state: KarrasVeSchedulerState,
        sample: jnp.ndarray,
        sigma: float,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.

        TODO Args:
        """
        if self.config.s_min <= sigma <= self.config.s_max:
            gamma = min(self.config.s_churn / state.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        key = random.split(key, num=1)
        eps = self.config.s_noise * random.normal(key=key, shape=sample.shape)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        state: KarrasVeSchedulerState,
        model_output: jnp.ndarray,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxKarrasVeOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`KarrasVeSchedulerState`): the `FlaxKarrasVeScheduler` state data class.
            model_output (`torch.Tensor` or `np.ndarray`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.Tensor` or `np.ndarray`): TODO
            return_dict (`bool`): option for returning tuple rather than FlaxKarrasVeOutput class

        Returns:
            [`~schedulers.scheduling_karras_ve_flax.FlaxKarrasVeOutput`] or `tuple`: Updated sample in the diffusion
            chain and derivative. [`~schedulers.scheduling_karras_ve_flax.FlaxKarrasVeOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """

        pred_original_sample = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - pred_original_sample) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        if not return_dict:
            return (sample_prev, derivative, state)

        return FlaxKarrasVeOutput(prev_sample=sample_prev, derivative=derivative, state=state)

    def step_correct(
        self,
        state: KarrasVeSchedulerState,
        model_output: jnp.ndarray,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: jnp.ndarray,
        sample_prev: jnp.ndarray,
        derivative: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxKarrasVeOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. TODO complete description

        Args:
            state (`KarrasVeSchedulerState`): the `FlaxKarrasVeScheduler` state data class.
            model_output (`torch.Tensor` or `np.ndarray`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.Tensor` or `np.ndarray`): TODO
            sample_prev (`torch.Tensor` or `np.ndarray`): TODO
            derivative (`torch.Tensor` or `np.ndarray`): TODO
            return_dict (`bool`): option for returning tuple rather than FlaxKarrasVeOutput class

        Returns:
            prev_sample (TODO): updated sample in the diffusion chain. derivative (TODO): TODO

        """
        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (sample_prev, derivative, state)

        return FlaxKarrasVeOutput(prev_sample=sample_prev, derivative=derivative, state=state)

    def add_noise(self, state: KarrasVeSchedulerState, original_samples, noise, timesteps):
        raise NotImplementedError()
