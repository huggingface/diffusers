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
from ..utils import BaseOutput


@dataclass
class HeliosDMDSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    model_outputs: torch.FloatTensor | None = None
    last_sample: torch.FloatTensor | None = None
    this_order: int | None = None


class HeliosDMDScheduler(SchedulerMixin, ConfigMixin):
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
        prediction_type: str = "flow_prediction",
        use_flow_sigmas: bool = True,
        use_dynamic_shifting: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "linear",
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

    # ---------------------------------- For DMD ----------------------------------
    def add_noise(self, original_samples, noise, timestep, sigmas, timesteps):
        sigmas = sigmas.to(noise.device)
        timesteps = timesteps.to(noise.device)
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def convert_flow_pred_to_x0(self, flow_pred, xt, timestep, sigmas, timesteps):
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        device = flow_pred.device
        flow_pred, xt, sigmas, timesteps = (x.double().to(device) for x in (flow_pred, xt, sigmas, timesteps))

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor = None,
        sample: torch.FloatTensor = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        cur_sampling_step: int = 0,
        dmd_noisy_tensor: torch.FloatTensor | None = None,
        dmd_sigmas: torch.FloatTensor | None = None,
        dmd_timesteps: torch.FloatTensor | None = None,
        all_timesteps: torch.FloatTensor | None = None,
    ) -> HeliosDMDSchedulerOutput | tuple:
        pred_image_or_video = self.convert_flow_pred_to_x0(
            flow_pred=model_output,
            xt=sample,
            timestep=torch.full((model_output.shape[0],), timestep, dtype=torch.long, device=model_output.device),
            sigmas=dmd_sigmas,
            timesteps=dmd_timesteps,
        )
        if cur_sampling_step < len(all_timesteps) - 1:
            prev_sample = self.add_noise(
                pred_image_or_video,
                dmd_noisy_tensor,
                torch.full(
                    (model_output.shape[0],),
                    all_timesteps[cur_sampling_step + 1],
                    dtype=torch.long,
                    device=model_output.device,
                ),
                sigmas=dmd_sigmas,
                timesteps=dmd_timesteps,
            )
        else:
            prev_sample = pred_image_or_video

        if not return_dict:
            return (prev_sample,)

        return HeliosDMDSchedulerOutput(prev_sample=prev_sample)

    def reset_scheduler_history(self):
        self._step_index = None
        self._begin_index = None

    def __len__(self):
        return self.config.num_train_timesteps
