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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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
class DPMSolverMultistepSchedulerState:
    common: CommonSchedulerState
    alpha_t: jnp.ndarray
    sigma_t: jnp.ndarray
    lambda_t: jnp.ndarray

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    num_inference_steps: Optional[int] = None

    # running values
    model_outputs: Optional[jnp.ndarray] = None
    lower_order_nums: Optional[jnp.int32] = None
    prev_timestep: Optional[jnp.int32] = None
    cur_sample: Optional[jnp.ndarray] = None

    @classmethod
    def create(
        cls,
        common: CommonSchedulerState,
        alpha_t: jnp.ndarray,
        sigma_t: jnp.ndarray,
        lambda_t: jnp.ndarray,
        init_noise_sigma: jnp.ndarray,
        timesteps: jnp.ndarray,
    ):
        return cls(
            common=common,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            lambda_t=lambda_t,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )


@dataclass
class FlaxDPMSolverMultistepSchedulerOutput(FlaxSchedulerOutput):
    state: DPMSolverMultistepSchedulerState


class FlaxDPMSolverMultistepScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    DPM-Solver (and the improved version DPM-Solver++) is a fast dedicated high-order solver for diffusion ODEs with
    the convergence order guarantee. Empirically, sampling by DPM-Solver with only 20 steps can generate high-quality
    samples, and it can generate quite good samples even in only 10 steps.

    For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095

    Currently, we support the multistep DPM-Solver for both noise prediction models and data prediction models. We
    recommend to use `solver_order=2` for guided sampling, and `solver_order=3` for unconditional sampling.

    We also support the "dynamic thresholding" method in Imagen (https://arxiv.org/abs/2205.11487). For pixel-space
    diffusion models, you can set both `algorithm_type="dpmsolver++"` and `thresholding=True` to use the dynamic
    thresholding. Note that the thresholding method is unsuitable for latent-space diffusion models (such as
    stable-diffusion).

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

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
        solver_order (`int`, default `2`):
            the order of DPM-Solver; can be `1` or `2` or `3`. We recommend to use `solver_order=2` for guided
            sampling, and `solver_order=3` for unconditional sampling.
        prediction_type (`str`, default `epsilon`):
            indicates whether the model predicts the noise (epsilon), or the data / `x0`. One of `epsilon`, `sample`,
            or `v-prediction`.
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            For pixel-space diffusion models, you can set both `algorithm_type=dpmsolver++` and `thresholding=True` to
            use the dynamic thresholding. Note that the thresholding method is unsuitable for latent-space diffusion
            models (such as stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487).
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid only when `thresholding=True` and
            `algorithm_type="dpmsolver++`.
        algorithm_type (`str`, default `dpmsolver++`):
            the algorithm type for the solver. Either `dpmsolver` or `dpmsolver++`. The `dpmsolver` type implements the
            algorithms in https://arxiv.org/abs/2206.00927, and the `dpmsolver++` type implements the algorithms in
            https://arxiv.org/abs/2211.01095. We recommend to use `dpmsolver++` with `solver_order=2` for guided
            sampling (e.g. stable-diffusion).
        solver_type (`str`, default `midpoint`):
            the solver type for the second-order solver. Either `midpoint` or `heun`. The solver type slightly affects
            the sample quality, especially for small number of steps. We empirically find that `midpoint` solvers are
            slightly better, so we recommend to use the `midpoint` type.
        lower_order_final (`bool`, default `True`):
            whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
            find this trick can stabilize the sampling of DPM-Solver for steps < 15, especially for steps <= 10.
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
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype

    def create_state(self, common: Optional[CommonSchedulerState] = None) -> DPMSolverMultistepSchedulerState:
        if common is None:
            common = CommonSchedulerState.create(self)

        # Currently we only support VP-type noise schedule
        alpha_t = jnp.sqrt(common.alphas_cumprod)
        sigma_t = jnp.sqrt(1 - common.alphas_cumprod)
        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)

        # settings for DPM-Solver
        if self.config.algorithm_type not in ["dpmsolver", "dpmsolver++"]:
            raise NotImplementedError(f"{self.config.algorithm_type} does is not implemented for {self.__class__}")
        if self.config.solver_type not in ["midpoint", "heun"]:
            raise NotImplementedError(f"{self.config.solver_type} does is not implemented for {self.__class__}")

        # standard deviation of the initial noise distribution
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]

        return DPMSolverMultistepSchedulerState.create(
            common=common,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            lambda_t=lambda_t,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )

    def set_timesteps(
        self, state: DPMSolverMultistepSchedulerState, num_inference_steps: int, shape: Tuple
    ) -> DPMSolverMultistepSchedulerState:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`DPMSolverMultistepSchedulerState`):
                the `FlaxDPMSolverMultistepScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            shape (`Tuple`):
                the shape of the samples to be generated.
        """

        timesteps = (
            jnp.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1)
            .round()[::-1][:-1]
            .astype(jnp.int32)
        )

        # initial running values

        model_outputs = jnp.zeros((self.config.solver_order,) + shape, dtype=self.dtype)
        lower_order_nums = jnp.int32(0)
        prev_timestep = jnp.int32(-1)
        cur_sample = jnp.zeros(shape, dtype=self.dtype)

        return state.replace(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            model_outputs=model_outputs,
            lower_order_nums=lower_order_nums,
            prev_timestep=prev_timestep,
            cur_sample=cur_sample,
        )

    def convert_model_output(
        self,
        state: DPMSolverMultistepSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.

        DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
        discretize an integral of the data prediction model. So we need to first convert the model output to the
        corresponding type to match the algorithm.

        Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
        DPM-Solver++ for both noise prediction model and data prediction model.

        Args:
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `jnp.ndarray`: the converted model output.
        """
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type == "dpmsolver++":
            if self.config.prediction_type == "epsilon":
                alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
                    " or `v_prediction` for the FlaxDPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                # Dynamic thresholding in https://arxiv.org/abs/2205.11487
                dynamic_max_val = jnp.percentile(
                    jnp.abs(x0_pred), self.config.dynamic_thresholding_ratio, axis=tuple(range(1, x0_pred.ndim))
                )
                dynamic_max_val = jnp.maximum(
                    dynamic_max_val, self.config.sample_max_value * jnp.ones_like(dynamic_max_val)
                )
                x0_pred = jnp.clip(x0_pred, -dynamic_max_val, dynamic_max_val) / dynamic_max_val
            return x0_pred
        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type == "dpmsolver":
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
                    " or `v_prediction` for the FlaxDPMSolverMultistepScheduler."
                )

    def dpm_solver_first_order_update(
        self,
        state: DPMSolverMultistepSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        prev_timestep: int,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `jnp.ndarray`: the sample tensor at the previous timestep.
        """
        t, s0 = prev_timestep, timestep
        m0 = model_output
        lambda_t, lambda_s = state.lambda_t[t], state.lambda_t[s0]
        alpha_t, alpha_s = state.alpha_t[t], state.alpha_t[s0]
        sigma_t, sigma_s = state.sigma_t[t], state.sigma_t[s0]
        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (jnp.exp(-h) - 1.0)) * m0
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (jnp.exp(h) - 1.0)) * m0
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        state: DPMSolverMultistepSchedulerState,
        model_output_list: jnp.ndarray,
        timestep_list: List[int],
        prev_timestep: int,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        One step for the second-order multistep DPM-Solver.

        Args:
            model_output_list (`List[jnp.ndarray]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `jnp.ndarray`: the sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = state.lambda_t[t], state.lambda_t[s0], state.lambda_t[s1]
        alpha_t, alpha_s0 = state.alpha_t[t], state.alpha_t[s0]
        sigma_t, sigma_s0 = state.sigma_t[t], state.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (jnp.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (jnp.exp(-h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (jnp.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((jnp.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (jnp.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (jnp.exp(h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (jnp.exp(h) - 1.0)) * D0
                    - (sigma_t * ((jnp.exp(h) - 1.0) / h - 1.0)) * D1
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        state: DPMSolverMultistepSchedulerState,
        model_output_list: jnp.ndarray,
        timestep_list: List[int],
        prev_timestep: int,
        sample: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        One step for the third-order multistep DPM-Solver.

        Args:
            model_output_list (`List[jnp.ndarray]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `jnp.ndarray`: the sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            state.lambda_t[t],
            state.lambda_t[s0],
            state.lambda_t[s1],
            state.lambda_t[s2],
        )
        alpha_t, alpha_s0 = state.alpha_t[t], state.alpha_t[s0]
        sigma_t, sigma_s0 = state.sigma_t[t], state.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (jnp.exp(-h) - 1.0)) * D0
                + (alpha_t * ((jnp.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((jnp.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (jnp.exp(h) - 1.0)) * D0
                - (sigma_t * ((jnp.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((jnp.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        return x_t

    def step(
        self,
        state: DPMSolverMultistepSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxDPMSolverMultistepSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by DPM-Solver. Core function to propagate the diffusion process
        from the learned model outputs (most often the predicted noise).

        Args:
            state (`DPMSolverMultistepSchedulerState`):
                the `FlaxDPMSolverMultistepScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxDPMSolverMultistepSchedulerOutput class

        Returns:
            [`FlaxDPMSolverMultistepSchedulerOutput`] or `tuple`: [`FlaxDPMSolverMultistepSchedulerOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        (step_index,) = jnp.where(state.timesteps == timestep, size=1)
        step_index = step_index[0]

        prev_timestep = jax.lax.select(step_index == len(state.timesteps) - 1, 0, state.timesteps[step_index + 1])

        model_output = self.convert_model_output(state, model_output, timestep, sample)

        model_outputs_new = jnp.roll(state.model_outputs, -1, axis=0)
        model_outputs_new = model_outputs_new.at[-1].set(model_output)
        state = state.replace(
            model_outputs=model_outputs_new,
            prev_timestep=prev_timestep,
            cur_sample=sample,
        )

        def step_1(state: DPMSolverMultistepSchedulerState) -> jnp.ndarray:
            return self.dpm_solver_first_order_update(
                state,
                state.model_outputs[-1],
                state.timesteps[step_index],
                state.prev_timestep,
                state.cur_sample,
            )

        def step_23(state: DPMSolverMultistepSchedulerState) -> jnp.ndarray:
            def step_2(state: DPMSolverMultistepSchedulerState) -> jnp.ndarray:
                timestep_list = jnp.array([state.timesteps[step_index - 1], state.timesteps[step_index]])
                return self.multistep_dpm_solver_second_order_update(
                    state,
                    state.model_outputs,
                    timestep_list,
                    state.prev_timestep,
                    state.cur_sample,
                )

            def step_3(state: DPMSolverMultistepSchedulerState) -> jnp.ndarray:
                timestep_list = jnp.array(
                    [
                        state.timesteps[step_index - 2],
                        state.timesteps[step_index - 1],
                        state.timesteps[step_index],
                    ]
                )
                return self.multistep_dpm_solver_third_order_update(
                    state,
                    state.model_outputs,
                    timestep_list,
                    state.prev_timestep,
                    state.cur_sample,
                )

            step_2_output = step_2(state)
            step_3_output = step_3(state)

            if self.config.solver_order == 2:
                return step_2_output
            elif self.config.lower_order_final and len(state.timesteps) < 15:
                return jax.lax.select(
                    state.lower_order_nums < 2,
                    step_2_output,
                    jax.lax.select(
                        step_index == len(state.timesteps) - 2,
                        step_2_output,
                        step_3_output,
                    ),
                )
            else:
                return jax.lax.select(
                    state.lower_order_nums < 2,
                    step_2_output,
                    step_3_output,
                )

        step_1_output = step_1(state)
        step_23_output = step_23(state)

        if self.config.solver_order == 1:
            prev_sample = step_1_output

        elif self.config.lower_order_final and len(state.timesteps) < 15:
            prev_sample = jax.lax.select(
                state.lower_order_nums < 1,
                step_1_output,
                jax.lax.select(
                    step_index == len(state.timesteps) - 1,
                    step_1_output,
                    step_23_output,
                ),
            )

        else:
            prev_sample = jax.lax.select(
                state.lower_order_nums < 1,
                step_1_output,
                step_23_output,
            )

        state = state.replace(
            lower_order_nums=jnp.minimum(state.lower_order_nums + 1, self.config.solver_order),
        )

        if not return_dict:
            return (prev_sample, state)

        return FlaxDPMSolverMultistepSchedulerOutput(prev_sample=prev_sample, state=state)

    def scale_model_input(
        self, state: DPMSolverMultistepSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            state (`DPMSolverMultistepSchedulerState`):
                the `FlaxDPMSolverMultistepScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
        return sample

    def add_noise(
        self,
        state: DPMSolverMultistepSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        return add_noise_common(state.common, original_samples, noise, timesteps)

    def __len__(self):
        return self.config.num_train_timesteps
