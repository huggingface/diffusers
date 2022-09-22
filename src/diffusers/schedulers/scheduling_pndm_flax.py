# Copyright 2022 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta=0.999) -> jnp.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`jnp.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas, dtype=jnp.float32)


@flax.struct.dataclass
class PNDMSchedulerState:
    # setable values
    _timesteps: jnp.ndarray
    num_inference_steps: Optional[int] = None
    prk_timesteps: Optional[jnp.ndarray] = None
    plms_timesteps: Optional[jnp.ndarray] = None
    timesteps: Optional[jnp.ndarray] = None

    # running values
    cur_model_output: Optional[jnp.ndarray] = None
    counter: int = 0
    cur_sample: Optional[jnp.ndarray] = None
    ets: jnp.ndarray = jnp.array([])

    @classmethod
    def create(cls, num_train_timesteps: int):
        return cls(_timesteps=jnp.arange(0, num_train_timesteps)[::-1])


@dataclass
class FlaxSchedulerOutput(SchedulerOutput):
    state: PNDMSchedulerState


class FlaxPNDMScheduler(SchedulerMixin, ConfigMixin):
    """
    Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
    namely Runge-Kutta method and a linear multi-step method.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2202.09778

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`jnp.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        skip_prk_steps (`bool`):
            allows the scheduler to skip the Runge-Kutta steps that are defined in the original paper as being required
            before plms steps; defaults to `False`.
        set_alpha_to_one (`bool`, default `False`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.
    """

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
        skip_prk_steps: bool = False,
        set_alpha_to_one: bool = False,
        steps_offset: int = 0,
    ):
        if trained_betas is not None:
            self.betas = jnp.asarray(trained_betas)
        if beta_schedule == "linear":
            self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps, dtype=jnp.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=jnp.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)

        self.final_alpha_cumprod = jnp.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

    def create_state(self):
        return PNDMSchedulerState.create(num_train_timesteps=self.config.num_train_timesteps)

    def set_timesteps(self, state: PNDMSchedulerState, num_inference_steps: int) -> PNDMSchedulerState:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`PNDMSchedulerState`):
                the `FlaxPNDMScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        offset = self.config.steps_offset

        step_ratio = self.config.num_train_timesteps // num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # rounding to avoid issues when num_inference_step is power of 3
        _timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round() + offset

        state = state.replace(num_inference_steps=num_inference_steps, _timesteps=_timesteps)

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            state = state.replace(
                prk_timesteps=jnp.array([]),
                plms_timesteps=jnp.concatenate(
                    [state._timesteps[:-1], state._timesteps[-2:-1], state._timesteps[-1:]]
                )[::-1],
            )
        else:
            prk_timesteps = jnp.array(state._timesteps[-self.pndm_order :]).repeat(2) + jnp.tile(
                jnp.array([0, self.config.num_train_timesteps // num_inference_steps // 2]), self.pndm_order
            )

            state = state.replace(
                prk_timesteps=(prk_timesteps[:-1].repeat(2)[1:-1])[::-1],
                plms_timesteps=state._timesteps[:-3][::-1],
            )

        return state.replace(
            timesteps=jnp.concatenate([state.prk_timesteps, state.plms_timesteps]).astype(jnp.int64),
            ets=jnp.array([]),
            counter=0,
        )

    def step(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This function calls `step_prk()` or `step_plms()` depending on the internal variable `counter`.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`FlaxSchedulerOutput`] or `tuple`: [`FlaxSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.

        """
        if state.counter < len(state.prk_timesteps) and not self.config.skip_prk_steps:
            return self.step_prk(
                state=state, model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict
            )
        else:
            return self.step_plms(
                state=state, model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict
            )

    def step_prk(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxSchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`FlaxSchedulerOutput`] or `tuple`: [`FlaxSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.

        """
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        diff_to_prev = 0 if state.counter % 2 else self.config.num_train_timesteps // state.num_inference_steps // 2
        prev_timestep = timestep - diff_to_prev
        timestep = state.prk_timesteps[state.counter // 4 * 4]

        if state.counter % 4 == 0:
            state = state.replace(
                cur_model_output=state.cur_model_output + 1 / 6 * model_output,
                ets=state.ets.append(model_output),
                cur_sample=sample,
            )
        elif (self.counter - 1) % 4 == 0:
            state = state.replace(cur_model_output=state.cur_model_output + 1 / 3 * model_output)
        elif (self.counter - 2) % 4 == 0:
            state = state.replace(cur_model_output=state.cur_model_output + 1 / 3 * model_output)
        elif (self.counter - 3) % 4 == 0:
            model_output = state.cur_model_output + 1 / 6 * model_output
            state = state.replace(cur_model_output=0)

        # cur_sample should not be `None`
        cur_sample = state.cur_sample if state.cur_sample is not None else sample

        prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
        state = state.replace(counter=state.counter + 1)

        if not return_dict:
            return (prev_sample, state)

        return FlaxSchedulerOutput(prev_sample=prev_sample, state=state)

    def step_plms(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxSchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`FlaxSchedulerOutput`] or `tuple`: [`FlaxSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.

        """
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if not self.config.skip_prk_steps and len(state.ets) < 3:
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )

        prev_timestep = timestep - self.config.num_train_timesteps // state.num_inference_steps

        if state.counter != 1:
            state = state.replace(ets=state.ets.append(model_output))
        else:
            prev_timestep = timestep
            timestep = timestep + self.config.num_train_timesteps // state.num_inference_steps

        if len(state.ets) == 1 and state.counter == 0:
            model_output = model_output
            state = state.replace(cur_sample=sample)
        elif len(state.ets) == 1 and state.counter == 1:
            model_output = (model_output + state.ets[-1]) / 2
            sample = state.cur_sample
            state = state.replace(cur_sample=None)
        elif len(state.ets) == 2:
            model_output = (3 * state.ets[-1] - state.ets[-2]) / 2
        elif len(state.ets) == 3:
            model_output = (23 * state.ets[-1] - 16 * state.ets[-2] + 5 * state.ets[-3]) / 12
        else:
            model_output = (1 / 24) * (
                55 * state.ets[-1] - 59 * state.ets[-2] + 37 * state.ets[-3] - 9 * state.ets[-4]
            )

        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
        state = state.replace(counter=state.counter + 1)

        if not return_dict:
            return (prev_sample, state)

        return FlaxSchedulerOutput(prev_sample=prev_sample, state=state)

    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )

        return prev_sample

    def add_noise(
        self,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[..., None]

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., None]

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
