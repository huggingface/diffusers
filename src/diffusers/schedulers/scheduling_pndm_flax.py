# Copyright 2024 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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
)


@flax.struct.dataclass
class PNDMSchedulerState:
    common: CommonSchedulerState
    final_alpha_cumprod: jnp.ndarray

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    num_inference_steps: Optional[int] = None
    prk_timesteps: Optional[jnp.ndarray] = None
    plms_timesteps: Optional[jnp.ndarray] = None

    # running values
    cur_model_output: Optional[jnp.ndarray] = None
    counter: Optional[jnp.int32] = None
    cur_sample: Optional[jnp.ndarray] = None
    ets: Optional[jnp.ndarray] = None

    @classmethod
    def create(
        cls,
        common: CommonSchedulerState,
        final_alpha_cumprod: jnp.ndarray,
        init_noise_sigma: jnp.ndarray,
        timesteps: jnp.ndarray,
    ):
        return cls(
            common=common,
            final_alpha_cumprod=final_alpha_cumprod,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )


@dataclass
class FlaxPNDMSchedulerOutput(FlaxSchedulerOutput):
    state: PNDMSchedulerState


class FlaxPNDMScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
    namely Runge-Kutta method and a linear multi-step method.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

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
            An offset added to the inference steps, as required by some model families.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            the `dtype` used for params and computation.
    """

    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    dtype: jnp.dtype
    pndm_order: int

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
        prediction_type: str = "epsilon",
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

    def create_state(self, common: Optional[CommonSchedulerState] = None) -> PNDMSchedulerState:
        if common is None:
            common = CommonSchedulerState.create(self)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        final_alpha_cumprod = (
            jnp.array(1.0, dtype=self.dtype) if self.config.set_alpha_to_one else common.alphas_cumprod[0]
        )

        # standard deviation of the initial noise distribution
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]

        return PNDMSchedulerState.create(
            common=common,
            final_alpha_cumprod=final_alpha_cumprod,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )

    def set_timesteps(self, state: PNDMSchedulerState, num_inference_steps: int, shape: Tuple) -> PNDMSchedulerState:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`PNDMSchedulerState`):
                the `FlaxPNDMScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            shape (`Tuple`):
                the shape of the samples to be generated.
        """

        step_ratio = self.config.num_train_timesteps // num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # rounding to avoid issues when num_inference_step is power of 3
        _timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round() + self.config.steps_offset

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51

            prk_timesteps = jnp.array([], dtype=jnp.int32)
            plms_timesteps = jnp.concatenate([_timesteps[:-1], _timesteps[-2:-1], _timesteps[-1:]])[::-1]

        else:
            prk_timesteps = _timesteps[-self.pndm_order :].repeat(2) + jnp.tile(
                jnp.array([0, self.config.num_train_timesteps // num_inference_steps // 2], dtype=jnp.int32),
                self.pndm_order,
            )

            prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1]
            plms_timesteps = _timesteps[:-3][::-1]

        timesteps = jnp.concatenate([prk_timesteps, plms_timesteps])

        # initial running values

        cur_model_output = jnp.zeros(shape, dtype=self.dtype)
        counter = jnp.int32(0)
        cur_sample = jnp.zeros(shape, dtype=self.dtype)
        ets = jnp.zeros((4,) + shape, dtype=self.dtype)

        return state.replace(
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prk_timesteps=prk_timesteps,
            plms_timesteps=plms_timesteps,
            cur_model_output=cur_model_output,
            counter=counter,
            cur_sample=cur_sample,
            ets=ets,
        )

    def scale_model_input(
        self, state: PNDMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
        return sample

    def step(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxPNDMSchedulerOutput, Tuple]:
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
            return_dict (`bool`): option for returning tuple rather than FlaxPNDMSchedulerOutput class

        Returns:
            [`FlaxPNDMSchedulerOutput`] or `tuple`: [`FlaxPNDMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """

        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.config.skip_prk_steps:
            prev_sample, state = self.step_plms(state, model_output, timestep, sample)
        else:
            prk_prev_sample, prk_state = self.step_prk(state, model_output, timestep, sample)
            plms_prev_sample, plms_state = self.step_plms(state, model_output, timestep, sample)

            cond = state.counter < len(state.prk_timesteps)

            prev_sample = jax.lax.select(cond, prk_prev_sample, plms_prev_sample)

            state = state.replace(
                cur_model_output=jax.lax.select(cond, prk_state.cur_model_output, plms_state.cur_model_output),
                ets=jax.lax.select(cond, prk_state.ets, plms_state.ets),
                cur_sample=jax.lax.select(cond, prk_state.cur_sample, plms_state.cur_sample),
                counter=jax.lax.select(cond, prk_state.counter, plms_state.counter),
            )

        if not return_dict:
            return (prev_sample, state)

        return FlaxPNDMSchedulerOutput(prev_sample=prev_sample, state=state)

    def step_prk(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    ) -> Union[FlaxPNDMSchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxPNDMSchedulerOutput class

        Returns:
            [`FlaxPNDMSchedulerOutput`] or `tuple`: [`FlaxPNDMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """

        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        diff_to_prev = jnp.where(
            state.counter % 2, 0, self.config.num_train_timesteps // state.num_inference_steps // 2
        )
        prev_timestep = timestep - diff_to_prev
        timestep = state.prk_timesteps[state.counter // 4 * 4]

        model_output = jax.lax.select(
            (state.counter % 4) != 3,
            model_output,  # remainder 0, 1, 2
            state.cur_model_output + 1 / 6 * model_output,  # remainder 3
        )

        state = state.replace(
            cur_model_output=jax.lax.select_n(
                state.counter % 4,
                state.cur_model_output + 1 / 6 * model_output,  # remainder 0
                state.cur_model_output + 1 / 3 * model_output,  # remainder 1
                state.cur_model_output + 1 / 3 * model_output,  # remainder 2
                jnp.zeros_like(state.cur_model_output),  # remainder 3
            ),
            ets=jax.lax.select(
                (state.counter % 4) == 0,
                state.ets.at[0:3].set(state.ets[1:4]).at[3].set(model_output),  # remainder 0
                state.ets,  # remainder 1, 2, 3
            ),
            cur_sample=jax.lax.select(
                (state.counter % 4) == 0,
                sample,  # remainder 0
                state.cur_sample,  # remainder 1, 2, 3
            ),
        )

        cur_sample = state.cur_sample
        prev_sample = self._get_prev_sample(state, cur_sample, timestep, prev_timestep, model_output)
        state = state.replace(counter=state.counter + 1)

        return (prev_sample, state)

    def step_plms(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    ) -> Union[FlaxPNDMSchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxPNDMSchedulerOutput class

        Returns:
            [`FlaxPNDMSchedulerOutput`] or `tuple`: [`FlaxPNDMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """

        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # NOTE: There is no way to check in the jitted runtime if the prk mode was ran before

        prev_timestep = timestep - self.config.num_train_timesteps // state.num_inference_steps
        prev_timestep = jnp.where(prev_timestep > 0, prev_timestep, 0)

        # Reference:
        # if state.counter != 1:
        #     state.ets.append(model_output)
        # else:
        #     prev_timestep = timestep
        #     timestep = timestep + self.config.num_train_timesteps // state.num_inference_steps

        prev_timestep = jnp.where(state.counter == 1, timestep, prev_timestep)
        timestep = jnp.where(
            state.counter == 1, timestep + self.config.num_train_timesteps // state.num_inference_steps, timestep
        )

        # Reference:
        # if len(state.ets) == 1 and state.counter == 0:
        #     model_output = model_output
        #     state.cur_sample = sample
        # elif len(state.ets) == 1 and state.counter == 1:
        #     model_output = (model_output + state.ets[-1]) / 2
        #     sample = state.cur_sample
        #     state.cur_sample = None
        # elif len(state.ets) == 2:
        #     model_output = (3 * state.ets[-1] - state.ets[-2]) / 2
        # elif len(state.ets) == 3:
        #     model_output = (23 * state.ets[-1] - 16 * state.ets[-2] + 5 * state.ets[-3]) / 12
        # else:
        #     model_output = (1 / 24) * (55 * state.ets[-1] - 59 * state.ets[-2] + 37 * state.ets[-3] - 9 * state.ets[-4])

        state = state.replace(
            ets=jax.lax.select(
                state.counter != 1,
                state.ets.at[0:3].set(state.ets[1:4]).at[3].set(model_output),  # counter != 1
                state.ets,  # counter 1
            ),
            cur_sample=jax.lax.select(
                state.counter != 1,
                sample,  # counter != 1
                state.cur_sample,  # counter 1
            ),
        )

        state = state.replace(
            cur_model_output=jax.lax.select_n(
                jnp.clip(state.counter, 0, 4),
                model_output,  # counter 0
                (model_output + state.ets[-1]) / 2,  # counter 1
                (3 * state.ets[-1] - state.ets[-2]) / 2,  # counter 2
                (23 * state.ets[-1] - 16 * state.ets[-2] + 5 * state.ets[-3]) / 12,  # counter 3
                (1 / 24)
                * (55 * state.ets[-1] - 59 * state.ets[-2] + 37 * state.ets[-3] - 9 * state.ets[-4]),  # counter >= 4
            ),
        )

        sample = state.cur_sample
        model_output = state.cur_model_output
        prev_sample = self._get_prev_sample(state, sample, timestep, prev_timestep, model_output)
        state = state.replace(counter=state.counter + 1)

        return (prev_sample, state)

    def _get_prev_sample(self, state: PNDMSchedulerState, sample, timestep, prev_timestep, model_output):
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
        alpha_prod_t = state.common.alphas_cumprod[timestep]
        alpha_prod_t_prev = jnp.where(
            prev_timestep >= 0, state.common.alphas_cumprod[prev_timestep], state.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.config.prediction_type == "v_prediction":
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        elif self.config.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`"
            )

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
        state: PNDMSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        return add_noise_common(state.common, original_samples, noise, timesteps)

    def __len__(self):
        return self.config.num_train_timesteps
