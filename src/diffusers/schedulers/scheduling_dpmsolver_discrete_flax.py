# Copyright 2022 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import flax
import jax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import FlaxSchedulerMixin, FlaxSchedulerOutput, broadcast_to_shape_from_left


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
class DPMSolverDiscreteSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None

    # running values
    model_outputs: Optional[jnp.ndarray] = None
    lower_order_nums: Optional[int] = None
    step_index: Optional[int] = None
    prev_timestep: Optional[int] = None
    cur_sample: Optional[jnp.ndarray] = None

    @classmethod
    def create(cls, num_train_timesteps: int):
        return cls(timesteps=jnp.arange(0, num_train_timesteps)[::-1])


@dataclass
class FlaxDPMSolverDiscreteSchedulerOutput(FlaxSchedulerOutput):
    state: DPMSolverDiscreteSchedulerState


class FlaxDPMSolverDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    DPM-Solver.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
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
        solver_order: int = 2,
        predict_x0: bool = True,
        thresholding: bool = False,
        sample_max_value: float = 1.0,
        solver_type: str = "dpm_solver",
        denoise_final: bool = False,
    ):
        if trained_betas is not None:
            self.betas = jnp.asarray(trained_betas)
        elif beta_schedule == "linear":
            self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps, dtype=jnp.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=jnp.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = jnp.sqrt(self.alphas_cumprod)
        self.sigma_t = jnp.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = jnp.log(self.alpha_t) - jnp.log(self.sigma_t)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DPM-Solver
        self.solver_order = solver_order
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.sample_max_value = sample_max_value
        self.denoise_final = denoise_final
        if solver_type in ["dpm_solver", "taylor"]:
            self.solver_type = solver_type
        else:
            raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

    def create_state(self):
        return DPMSolverDiscreteSchedulerState.create(num_train_timesteps=self.config.num_train_timesteps)

    def set_timesteps(self, state: DPMSolverDiscreteSchedulerState, num_inference_steps: int, shape: Tuple) -> DPMSolverDiscreteSchedulerState:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`DPMSolverDiscreteSchedulerState`):
                the `FlaxDPMSolverDiscreteScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            shape (`Tuple`):
                the shape of the samples to be generated.
        """
        timesteps = jnp.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1).round()[::-1][:-1].astype(jnp.int32)

        return state.replace(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            model_outputs=jnp.zeros((self.solver_order,) + shape),
            lower_order_nums=0,
            step_index=0,
            prev_timestep=-1,
            cur_sample=jnp.zeros(shape),
        )

    def convert_model_output(
        self,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        TODO
        """
        if self.predict_x0:
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = (sample - sigma_t * model_output) / alpha_t
            if self.thresholding:
                # A hyperparameter in the paper of Imagen (https://arxiv.org/abs/2205.11487).
                p = 0.995
                s = jnp.percentile(jnp.abs(x0_pred), p, axis=tuple(range(1, x0_pred.ndim)))
                s = jnp.max(s, self.max_val)
                x0_pred = jnp.clip(x0_pred, -s, s) / s
            return x0_pred
        else:
            return model_output

    def dpm_solver_first_order_update(
        self,
        model_output: jnp.ndarray,
        timestep: int,
        prev_timestep: int,
        sample: jnp.ndarray
    ) -> jnp.ndarray:
        """
        TODO
        """
        t, s0 = prev_timestep, timestep
        m0 = model_output
        lambda_t, lambda_s = self.lambda_t[t], self.lambda_t[s0]
        alpha_t, alpha_s = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s = self.sigma_t[t], self.sigma_t[s0]
        h = lambda_t - lambda_s
        if self.predict_x0:
            x_t = (
                (sigma_t / sigma_s) * sample
                - (alpha_t * (jnp.exp(-h) - 1.)) * m0
            )
        else:
            x_t = (
                (alpha_t / alpha_s) * sample
                - (sigma_t * (jnp.exp(h) - 1.)) * m0
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: jnp.ndarray,
        timestep_list: List[int],
        prev_timestep: int,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        TODO
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1. / r0) * (m0 - m1)
        if self.predict_x0:
            if self.solver_type == 'dpm_solver':
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (jnp.exp(-h) - 1.)) * D0
                    - 0.5 * (alpha_t * (jnp.exp(-h) - 1.)) * D1
                )
            elif self.solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (jnp.exp(-h) - 1.)) * D0
                    + (alpha_t * ((jnp.exp(-h) - 1.) / h + 1.)) * D1
                )
        else:
            if self.solver_type == 'dpm_solver':
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (jnp.exp(h) - 1.)) * D0
                    - 0.5 * (sigma_t * (jnp.exp(h) - 1.)) * D1
                )
            elif self.solver_type == 'taylor':
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (jnp.exp(h) - 1.)) * D0
                    - (sigma_t * ((jnp.exp(h) - 1.) / h - 1.)) * D1
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: jnp.ndarray,
        timestep_list: List[int],
        prev_timestep: int,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        TODO
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1], self.lambda_t[s2]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1. / r0) * (m0 - m1), (1. / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        if self.predict_x0:
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (jnp.exp(-h) - 1.)) * D0
                + (alpha_t * ((jnp.exp(-h) - 1.) / h + 1.)) * D1
                - (alpha_t * ((jnp.exp(-h) - 1. + h) / h**2 - 0.5)) * D2
            )
        else:
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (jnp.exp(h) - 1.)) * D0
                - (sigma_t * ((jnp.exp(h) - 1.) / h - 1.)) * D1
                - (sigma_t * ((jnp.exp(h) - 1. - h) / h**2 - 0.5)) * D2
            )
        return x_t

    def step(
        self,
        state: DPMSolverDiscreteSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxDPMSolverDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by DPM-Solver. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`DPMSolverDiscreteSchedulerState`): the `FlaxDPMSolverDiscreteScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxDPMSolverDiscreteSchedulerOutput class

        Returns:
            [`FlaxDPMSolverDiscreteSchedulerOutput`] or `tuple`: [`FlaxDPMSolverDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        prev_timestep = jax.lax.cond(
            state.step_index == len(state.timesteps) - 1,
            lambda _: 0,
            lambda _: state.timesteps[state.step_index + 1],
            (),
        )

        model_output = self.convert_model_output(model_output, timestep, sample) 

        model_outputs_new = jnp.roll(state.model_outputs, -1, axis=0)
        model_outputs_new = model_outputs_new.at[-1].set(model_output)
        state = state.replace(
            model_outputs=model_outputs_new,
            prev_timestep=prev_timestep,
            cur_sample=sample,
        )

        def step_1(
            state: DPMSolverDiscreteSchedulerState
        ) -> jnp.ndarray:
            return self.dpm_solver_first_order_update(
                state.model_outputs[-1],
                state.timesteps[state.step_index],
                state.prev_timestep,
                state.cur_sample,
            )

        def step_23(
            state: DPMSolverDiscreteSchedulerState
        ) -> jnp.ndarray:

            def step_2(
                state: DPMSolverDiscreteSchedulerState
            ) -> jnp.ndarray:
                timestep_list = jnp.array([
                    state.timesteps[state.step_index - 1],
                    state.timesteps[state.step_index]
                ])
                return self.multistep_dpm_solver_second_order_update(
                    state.model_outputs,
                    timestep_list,
                    state.prev_timestep,
                    state.cur_sample,
                )

            def step_3(
                state: DPMSolverDiscreteSchedulerState
            ) -> jnp.ndarray:
                timestep_list = jnp.array([
                    state.timesteps[state.step_index - 2],
                    state.timesteps[state.step_index - 1],
                    state.timesteps[state.step_index]
                ])
                return self.multistep_dpm_solver_third_order_update(
                    state.model_outputs,
                    timestep_list,
                    state.prev_timestep,
                    state.cur_sample,
                )

            if self.solver_order == 2:
                return step_2(state)
            elif self.denoise_final:
                return jax.lax.cond(
                    state.lower_order_nums < 2,
                    step_2,
                    lambda state: jax.lax.cond(
                        state.step_index == len(state.timesteps) - 2,
                        step_2,
                        step_3,
                        state,
                    ),
                    state,
                )
            else:
                return jax.lax.cond(
                    state.lower_order_nums < 2,
                    step_2,
                    step_3,
                    state,
                )

        if self.solver_order == 1:
            prev_sample = step_1(state)
        elif self.denoise_final:
            prev_sample = jax.lax.cond(
                state.lower_order_nums < 1,
                step_1,
                lambda state: jax.lax.cond(
                    state.step_index == len(state.timesteps) - 1,
                    step_1,
                    step_23,
                    state,
                ),
                state,
            )
        else:
            prev_sample = jax.lax.cond(
                state.lower_order_nums < 1,
                step_1,
                step_23,
                state,
            ) 

        state = state.replace(
            lower_order_nums=jnp.minimum(state.lower_order_nums + 1, self.solver_order),
            step_index=(state.step_index + 1),
        )

        if not return_dict:
            return (prev_sample, state)

        return FlaxDPMSolverDiscreteSchedulerOutput(prev_sample=prev_sample, state=state)

    def scale_model_input(
        self, state: DPMSolverDiscreteSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            state (`DPMSolverDiscreteSchedulerState`): the `FlaxDPMSolverDiscreteScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
        return sample

    def __len__(self):
        return self.config.num_train_timesteps
