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
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
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
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DPMSolverDiscreteScheduler(SchedulerMixin, ConfigMixin):
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

    _compatible_classes = [
        "DDIMScheduler",
        "DDPMScheduler",
        "PNDMScheduler",
        "LMSDiscreteScheduler",
        "EulerDiscreteScheduler",
        "EulerAncestralDiscreteScheduler",
    ]

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        solver_order: int = 2,
        predict_x0: bool = True,
        thresholding: bool = False,
        sample_max_value: float = 1.0,
        solver_type: str = "dpm_solver",
        denoise_final: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

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

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None,] * self.solver_order
        self.lower_order_nums = 0

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.model_outputs = [None,] * self.solver_order
        self.lower_order_nums = 0

    def convert_model_output(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        TODO
        """
        if self.predict_x0:
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = (sample - sigma_t * model_output) / alpha_t
            if self.thresholding:
                # Dynamic thresholding in https://arxiv.org/abs/2205.11487
                p = 0.995   # A hyperparameter in the paper of "Imagen" (https://arxiv.org/abs/2205.11487).
                s = torch.quantile(torch.abs(x0_pred).reshape((x0_pred.shape[0], -1)), p, dim=1)
                s = torch.maximum(s, self.sample_max_value * torch.ones_like(s).to(s.device))[(...,) + (None,)*(x0_pred.ndim - 1)]
                x0_pred = torch.clamp(x0_pred, -s, s) / s
            return x0_pred
        else:
            return model_output

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        TODO
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        if self.predict_x0:
            x_t = (
                (sigma_t / sigma_s) * sample
                - (alpha_t * (torch.exp(-h) - 1.)) * model_output
            )
        else:
            x_t = (
                (alpha_t / alpha_s) * sample
                - (sigma_t * (torch.exp(h) - 1.)) * model_output
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
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
                    - (alpha_t * (torch.exp(-h) - 1.)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.)) * D1
                )
            elif self.solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.) / h + 1.)) * D1
                )
        else:
            if self.solver_type == 'dpm_solver':
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.)) * D1
                )
            elif self.solver_type == 'taylor':
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.) / h - 1.)) * D1
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
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
                - (alpha_t * (torch.exp(-h) - 1.)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.) / h + 1.)) * D1
                - (alpha_t * ((torch.exp(-h) - 1. + h) / h**2 - 0.5)) * D2
            )
        else:
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (torch.exp(h) - 1.)) * D0
                - (sigma_t * ((torch.exp(h) - 1.) / h - 1.)) * D1
                - (sigma_t * ((torch.exp(h) - 1. - h) / h**2 - 0.5)) * D2
            )
        return x_t

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DPM-Solver. This has one forward pass with multiple
        times to approximate the solution.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        denoise_final = (step_index == len(self.timesteps) - 1) and self.denoise_final
        denoise_second = (step_index == len(self.timesteps) - 2) and self.denoise_final

        model_output = self.convert_model_output(model_output, timestep, sample) 
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.solver_order == 1 or self.lower_order_nums < 1 or denoise_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample)
        elif self.solver_order == 2 or self.lower_order_nums < 2 or denoise_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, timestep_list, prev_timestep, sample)
        else:
            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, timestep_list, prev_timestep, sample)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def __len__(self):
        return self.config.num_train_timesteps
