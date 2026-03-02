# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
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
from typing import Literal

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..schedulers.scheduling_utils import SchedulerMixin
from ..utils import BaseOutput, deprecate


@dataclass
class HeliosSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    model_outputs: torch.FloatTensor | None = None
    last_sample: torch.FloatTensor | None = None
    this_order: int | None = None


class HeliosScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,  # Following Stable diffusion 3,
        stages: int = 3,
        stage_range: list = [0, 1 / 3, 2 / 3, 1],
        gamma: float = 1 / 3,
        # For UniPC
        thresholding: bool = False,
        prediction_type: str = "flow_prediction",
        solver_order: int = 2,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list[int] = [],
        solver_p: SchedulerMixin = None,
        use_flow_sigmas: bool = True,
        scheduler_type: str = "unipc",  # ["euler", "unipc"]
        use_dynamic_shifting: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "exponential",
    ):
        self.timestep_ratios = {}  # The timestep ratio for each stage
        self.timesteps_per_stage = {}  # The detailed timesteps per stage (fix max and min per stage)
        self.sigmas_per_stage = {}  # always uniform [1000, 0]
        self.start_sigmas = {}  # for start point / upsample renoise
        self.end_sigmas = {}  # for end point
        self.ori_start_sigmas = {}

        # self.init_sigmas()
        self.init_sigmas_for_each_stage()
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.gamma = gamma

        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

    def init_sigmas(self):
        """
        initialize the global timesteps and sigmas
        """
        num_train_timesteps = self.config.num_train_timesteps
        shift = self.config.shift

        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps + 1)
        sigmas = 1.0 - alphas
        sigmas = np.flip(shift * sigmas / (1 + (shift - 1) * sigmas))[:-1].copy()
        sigmas = torch.from_numpy(sigmas)
        timesteps = (sigmas * num_train_timesteps).clone()

        self._step_index = None
        self._begin_index = None
        self.timesteps = timesteps
        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def init_sigmas_for_each_stage(self):
        """
        Init the timesteps for each stage
        """
        self.init_sigmas()

        stage_distance = []
        stages = self.config.stages
        training_steps = self.config.num_train_timesteps
        stage_range = self.config.stage_range

        # Init the start and end point of each stage
        for i_s in range(stages):
            # To decide the start and ends point
            start_indice = int(stage_range[i_s] * training_steps)
            start_indice = max(start_indice, 0)
            end_indice = int(stage_range[i_s + 1] * training_steps)
            end_indice = min(end_indice, training_steps)
            start_sigma = self.sigmas[start_indice].item()
            end_sigma = self.sigmas[end_indice].item() if end_indice < training_steps else 0.0
            self.ori_start_sigmas[i_s] = start_sigma

            if i_s != 0:
                ori_sigma = 1 - start_sigma
                gamma = self.config.gamma
                corrected_sigma = (1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)) * ori_sigma
                # corrected_sigma = 1 / (2 - ori_sigma) * ori_sigma
                start_sigma = 1 - corrected_sigma

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[i_s] = start_sigma
            self.end_sigmas[i_s] = end_sigma

        # Determine the ratio of each stage according to flow length
        tot_distance = sum(stage_distance)
        for i_s in range(stages):
            if i_s == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:i_s]) / tot_distance
            if i_s == stages - 1:
                end_ratio = 0.9999999999999999
            else:
                end_ratio = sum(stage_distance[: i_s + 1]) / tot_distance

            self.timestep_ratios[i_s] = (start_ratio, end_ratio)

        # Determine the timesteps and sigmas for each stage
        for i_s in range(stages):
            timestep_ratio = self.timestep_ratios[i_s]
            # timestep_max = self.timesteps[int(timestep_ratio[0] * training_steps)]
            timestep_max = min(self.timesteps[int(timestep_ratio[0] * training_steps)], 999)
            timestep_min = self.timesteps[min(int(timestep_ratio[1] * training_steps), training_steps - 1)]
            timesteps = np.linspace(timestep_max, timestep_min, training_steps + 1)
            self.timesteps_per_stage[i_s] = (
                timesteps[:-1] if isinstance(timesteps, torch.Tensor) else torch.from_numpy(timesteps[:-1])
            )
            stage_sigmas = np.linspace(0.999, 0, training_steps + 1)
            self.sigmas_per_stage[i_s] = torch.from_numpy(stage_sigmas[:-1])

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        stage_index: int | None = None,
        device: str | torch.device = None,
        sigmas: bool | None = None,
        mu: bool | None = None,
        is_amplify_first_chunk: bool = False,
    ):
        """
        Setting the timesteps and sigmas for each stage
        """
        if self.config.scheduler_type == "dmd":
            if is_amplify_first_chunk:
                num_inference_steps = num_inference_steps * 2 + 1
            else:
                num_inference_steps = num_inference_steps + 1

        self.num_inference_steps = num_inference_steps
        self.init_sigmas()

        if self.config.stages == 1:
            if sigmas is None:
                sigmas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)[:-1].astype(
                    np.float32
                )
                if self.config.shift != 1.0:
                    assert not self.config.use_dynamic_shifting
                    sigmas = self.time_shift(self.config.shift, 1.0, sigmas)
            timesteps = (sigmas * self.config.num_train_timesteps).copy()
            sigmas = torch.from_numpy(sigmas)
        else:
            stage_timesteps = self.timesteps_per_stage[stage_index]
            timesteps = np.linspace(
                stage_timesteps[0].item(),
                stage_timesteps[-1].item(),
                num_inference_steps,
            )

            stage_sigmas = self.sigmas_per_stage[stage_index]
            ratios = np.linspace(stage_sigmas[0].item(), stage_sigmas[-1].item(), num_inference_steps)
            sigmas = torch.from_numpy(ratios)

        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1)]).to(device=device)

        self._step_index = None
        self.reset_scheduler_history()

        if self.config.scheduler_type == "dmd":
            self.timesteps = self.timesteps[:-1]
            self.sigmas = torch.cat([self.sigmas[:-2], self.sigmas[-1:]])

        if self.config.use_dynamic_shifting:
            assert self.config.shift == 1.0
            self.sigmas = self.time_shift(mu, 1.0, self.sigmas)
            if self.config.stages == 1:
                self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
            else:
                self.timesteps = self.timesteps_per_stage[stage_index].min() + self.sigmas[:-1] * (
                    self.timesteps_per_stage[stage_index].max() - self.timesteps_per_stage[stage_index].min()
                )

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.time_shift
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        """
        Apply time shifting to the sigmas.

        Args:
            mu (`float`):
                The mu parameter for the time shift.
            sigma (`float`):
                The sigma parameter for the time shift.
            t (`torch.Tensor`):
                The input timesteps.

        Returns:
            `torch.Tensor`:
                The time-shifted timesteps.
        """
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._time_shift_exponential
    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._time_shift_linear
    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    # ---------------------------------- Euler ----------------------------------
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step_euler(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor = None,
        sample: torch.FloatTensor = None,
        generator: torch.Generator | None = None,
        sigma: torch.FloatTensor | None = None,
        sigma_next: torch.FloatTensor | None = None,
        return_dict: bool = True,
    ) -> HeliosSchedulerOutput | tuple:
        assert (sigma is None) == (sigma_next is None), "sigma and sigma_next must both be None or both be not None"

        if sigma is None and sigma_next is None:
            if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
            ):
                raise ValueError(
                    (
                        "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                        " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                        " one of the `scheduler.timesteps` as a timestep."
                    ),
                )

        if self.step_index is None:
            self._step_index = 0

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if sigma is None and sigma_next is None:
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return HeliosSchedulerOutput(prev_sample=prev_sample)

    # ---------------------------------- UniPC ----------------------------------
    def _sigma_to_alpha_sigma_t(self, sigma):
        if self.config.use_flow_sigmas:
            alpha_t = 1 - sigma
            sigma_t = torch.clamp(sigma, min=1e-8)
        else:
            alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
            sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        sigma: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Convert the model output to the corresponding type the UniPC algorithm needs.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        flag = False
        if sigma is None:
            flag = True
            sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.config.prediction_type == "flow_prediction":
                if flag:
                    sigma_t = self.sigmas[self.step_index]
                else:
                    sigma_t = sigma
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
                    "`v_prediction`, or `flow_prediction` for the UniPCMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        order: int = None,
        sigma: torch.Tensor = None,
        sigma_next: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of UniP at this timestep (corresponds to the *p* in UniPC-p).

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError("missing `order` as a required keyword argument")
        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        if sigma_next is None and sigma is None:
            sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        else:
            sigma_t, sigma_s0 = sigma_next, sigma
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int = None,
        sigma_before: torch.Tensor = None,
        sigma: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniC (B(h) version).

        Args:
            this_model_output (`torch.Tensor`):
                The model outputs at `x_t`.
            this_timestep (`int`):
                The current timestep `t`.
            last_sample (`torch.Tensor`):
                The generated sample before the last predictor `x_{t-1}`.
            this_sample (`torch.Tensor`):
                The generated sample after the last predictor `x_{t}`.
            order (`int`):
                The `p` of UniC-p at this step. The effective order of accuracy should be `order + 1`.

        Returns:
            `torch.Tensor`:
                The corrected sample tensor at the current timestep.
        """
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError("missing `last_sample` as a required keyword argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError("missing `this_sample` as a required keyword argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError("missing `order` as a required keyword argument")
        if this_timestep is not None:
            deprecate(
                "this_timestep",
                "1.0.0",
                "Passing `this_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        if sigma_before is None and sigma is None:
            sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        else:
            sigma_t, sigma_s0 = sigma, sigma_before
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def step_unipc(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor = None,
        sample: torch.Tensor = None,
        return_dict: bool = True,
        model_outputs: list = None,
        timestep_list: list = None,
        sigma_before: torch.Tensor = None,
        sigma: torch.Tensor = None,
        sigma_next: torch.Tensor = None,
        cus_step_index: int = None,
        cus_lower_order_num: int = None,
        cus_this_order: int = None,
        cus_last_sample: torch.Tensor = None,
    ) -> HeliosSchedulerOutput | tuple:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if cus_step_index is None:
            if self.step_index is None:
                self._step_index = 0
        else:
            self._step_index = cus_step_index

        if cus_lower_order_num is not None:
            self.lower_order_nums = cus_lower_order_num

        if cus_this_order is not None:
            self.this_order = cus_this_order

        if cus_last_sample is not None:
            self.last_sample = cus_last_sample

        use_corrector = (
            self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )

        # Convert model output using the proper conversion method
        model_output_convert = self.convert_model_output(model_output, sample=sample, sigma=sigma)

        if model_outputs is not None and timestep_list is not None:
            self.model_outputs = model_outputs[:-1]
            self.timestep_list = timestep_list[:-1]

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                sigma_before=sigma_before,
                sigma=sigma,
            )

        if model_outputs is not None and timestep_list is not None:
            model_outputs[-1] = model_output_convert
            self.model_outputs = model_outputs[1:]
            self.timestep_list = timestep_list[1:]
        else:
            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
                self.timestep_list[i] = self.timestep_list[i + 1]
            self.model_outputs[-1] = model_output_convert
            self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            sample=sample,
            order=self.this_order,
            sigma=sigma,
            sigma_next=sigma_next,
        )

        if cus_lower_order_num is None:
            if self.lower_order_nums < self.config.solver_order:
                self.lower_order_nums += 1

        # upon completion increase step index by one
        if cus_step_index is None:
            self._step_index += 1

        if not return_dict:
            return (prev_sample, model_outputs, self.last_sample, self.this_order)

        return HeliosSchedulerOutput(
            prev_sample=prev_sample,
            model_outputs=model_outputs,
            last_sample=self.last_sample,
            this_order=self.this_order,
        )

    # ---------------------------------- Merge ----------------------------------
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor = None,
        sample: torch.FloatTensor = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> HeliosSchedulerOutput | tuple:
        if self.config.scheduler_type == "euler":
            return self.step_euler(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                generator=generator,
                return_dict=return_dict,
            )
        elif self.config.scheduler_type == "unipc":
            return self.step_unipc(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict,
            )
        else:
            raise NotImplementedError

    def reset_scheduler_history(self):
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.disable_corrector = self.config.disable_corrector
        self.solver_p = self.config.solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

    def __len__(self):
        return self.config.num_train_timesteps
