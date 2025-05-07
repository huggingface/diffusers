# Copyright 2024 The SkyReels-V2 Authors and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `FlowUniPCMultistepScheduler` is a training-free framework designed for the fast sampling of diffusion models,
    adapted for flow matching.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        solver_order (`int`, default `2`):
            The UniPC order which can be any positive integer. The effective order of accuracy is `solver_order + 1`
            due to the UniC. It is recommended to use `solver_order=2` for guided sampling, and `solver_order=3` for
            unconditional sampling.
        prediction_type (`str`, defaults to "flow_prediction"):
            Prediction type of the scheduler function; must be `flow_prediction` for this scheduler, which predicts the
            flow of the diffusion process.
        shift (`float`, defaults to 1.0):
            Scaling factor for time shifting in flow matching.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to use dynamic time shifting based on image resolution.
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True` and `predict_x0=True`.
        predict_x0 (`bool`, defaults to `True`):
            Whether to use the updating algorithm on the predicted x0.
        solver_type (`str`, default `bh2`):
            Solver type for UniPC. It is recommended to use `bh1` for unconditional sampling when steps < 10, and `bh2`
            otherwise.
        lower_order_final (`bool`, default `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        disable_corrector (`list`, default `[]`):
            Decides which step to disable the corrector to mitigate the misalignment between `epsilon_theta(x_t, c)`
            and `epsilon_theta(x_t^c, c)` which can influence convergence for a large guidance scale. Corrector is
            usually disabled during the first few steps.
        solver_p (`SchedulerMixin`, default `None`):
            Any other scheduler that if specified, the algorithm becomes `solver_p + UniC`.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting=False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p: SchedulerMixin = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
    ):
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

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

    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for sampling. If `None`, default values are used.
            mu (`float`, *optional*):
                Value for dynamic shifting based on image resolution. Required when `use_dynamic_shifting=True`.
            shift (`float`, *optional*):
                Scaling factor for time shifting. Overrides config value if provided.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]  # pyright: ignore

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)  # pyright: ignore
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # pyright: ignore

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)  # pyright: ignore

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def _sigma_to_alpha_sigma_t(self, sigma):
        # Compute alpha, sigma_t from sigma
        alpha = torch.sigmoid(-sigma)
        sigma_t = torch.sqrt((1 - alpha**2) / alpha**2)
        return alpha, sigma_t

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return mu * t / (mu + (sigma - mu) * t)

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert the model output to the corresponding type the flow matching framework expects.

        Args:
            model_output (`torch.Tensor`): direct output from the model.
            sample (`torch.Tensor`, *optional*): current instance of sample being created.

        Returns:
            `torch.Tensor`: converted model output for the flow matching framework.
        """
        # We dynamically set the scheduler to the correct inference steps
        if self.config.prediction_type == "flow_prediction":
            sigma = self.sigmas[self._step_index]
            t = self.timesteps[self._step_index].to(model_output.device, dtype=model_output.dtype)
            t = t / self.config.num_train_timesteps

            # Compute alpha, sigma_t from sigma
            alpha, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            alpha = alpha.to(model_output.device, dtype=model_output.dtype)
            sigma_t = sigma_t.to(model_output.device, dtype=model_output.dtype)

            if self.predict_x0:
                if self.config.thresholding:
                    model_output = self._threshold_sample(model_output)
                x0_pred = model_output
                derivative = (sample - alpha * x0_pred) / sigma_t
            else:
                derivative = model_output
            return derivative
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be flow_prediction for {self.__class__}"
            )

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        order: int = None,  # pyright: ignore
        **kwargs,
    ) -> torch.Tensor:
        """
        Multistep universal `P` for building the predictor solver.

        Args:
            model_output (`torch.Tensor`):
                Direct output from the model.
            sample (`torch.Tensor`, *optional*):
                Current instance of sample being created.
            order (`int`, *optional*):
                Order of the solver. If `None`, it will be set based on scheduler configuration.

        Returns:
            `torch.Tensor`: The predicted sample for the predictor solver.
        """
        if order is None:
            order = self.config.solver_order

        model_output = self.convert_model_output(model_output, sample=sample)

        # For P(x_{t+1}, x_t, x, x_{t-1} + univeral_coeff * ds_{t-1})
        # DPMSolver only need to save x_{t-1}, x_t, and ds_{t-1} and the higher order.
        # We reuse the concept of UniPC and DPMSolver in the `uni_p*_update` function
        self.model_outputs.append(model_output.to(sample.dtype))
        self.model_outputs.pop(0)

        time_step = self.timesteps[self._step_index].to(sample.device, model_output.dtype)
        prev_time_step = self.timesteps[self._step_index + 1].to(sample.device, model_output.dtype)

        if self._step_index >= len(self.timesteps):
            raise ValueError("Requested prediction step cannot advance any further. You cannot advance further.")

        # current_sigma = self.sigmas[self._step_index].to(sample.device, model_output.dtype)
        dt = prev_time_step - time_step

        # 1. P(x_{t+1}, x_t, ds_t)
        # Define discretized time and compute the time difference
        model_output_dagger = model_output
        # time_step_array = torch.tensor([1.0, time_step, time_step**2, time_step**3])
        # prev_time_step_array = torch.tensor([1.0, prev_time_step, prev_time_step**2, prev_time_step**3])

        if order == 1:  # predictor with euler steps
            if self.config.solver_type == "bh1":
                x_t = sample + dt * model_output_dagger
            elif self.config.solver_type == "bh2":
                x_t = sample + dt * model_output_dagger
        else:
            self.timestep_list.append(time_step)
            self.timestep_list.pop(0)

            # Order matching the UniPC
            if 2 <= order <= 3:
                current_model_output = model_output_dagger
                prev_model_output = self.model_outputs[-2]

                time_coe = dt

                rhos = self.sigmas[self._step_index - 1] / self.sigmas[self._step_index]
                rhos = rhos.to(sample.device, model_output.dtype)

                # t -> t + 1
                if order == 2:
                    # Bh1
                    if self.config.solver_type == "bh1":
                        # Taylor expansion
                        h_tau = time_coe
                        h_phi = time_coe

                        # 2nd order expansion
                        x_t = (
                            sample
                            + h_phi * current_model_output
                            + 0.5 * h_phi**2 * (current_model_output - prev_model_output) / dt
                        )
                    elif self.config.solver_type == "bh2":
                        # Original DPM Solver ++: https://github.com/LuChengTHU/dpm-solver
                        h_t, h_t_1 = time_step, self.timestep_list[-2]
                        # r = rhos

                        # prediction: 2nd order expansion from UniPC paper
                        h = h_t - h_t_1
                        x_t = (
                            sample
                            + h * current_model_output
                            - 0.5 * h**2 * (current_model_output - prev_model_output) / (h_t - h_t_1)
                        )
                elif order == 3:
                    prev_prev_model_output = self.model_outputs[-3]
                    h_t, h_t_1, h_t_2 = time_step, self.timestep_list[-2], self.timestep_list[-3]
                    # r, r_1 = rhos, self.sigmas[self._step_index - 2] / self.sigmas[self._step_index - 1]
                    _, r_1 = rhos, self.sigmas[self._step_index - 2] / self.sigmas[self._step_index - 1]
                    r_1 = r_1.to(sample.device, model_output.dtype)

                    # Original DPM Solver ++: https://github.com/LuChengTHU/dpm-solver
                    if self.config.solver_type == "bh1":
                        # Taylor expansion
                        h_tau = time_coe
                        h_phi = time_coe
                        h = h_t_1 - h_t_2
                        derivative2 = (current_model_output - prev_model_output) / (h_t - h_t_1)
                        derivative3 = (
                            derivative2 - (prev_model_output - prev_prev_model_output) / (h_t_1 - h_t_2)
                        ) / (h_t - h_t_2)
                        x_t = (
                            sample
                            + h_tau * current_model_output
                            + 0.5 * h_tau**2 * derivative2
                            + (1.0 / 6.0) * h_tau**3 * derivative3
                        )
                    elif self.config.solver_type == "bh2":
                        # From UniC paper: https://github.com/wl-zhao/UniPC
                        h1 = h_t - h_t_1
                        h2 = h_t_1 - h_t_2
                        h_left_01 = h_t - h_t_1
                        h_left_12 = h_t_1 - h_t_2
                        h_left_02 = h_t - h_t_2
                        taylor1 = current_model_output
                        taylor2 = (current_model_output - prev_model_output) / h_left_01
                        taylor3 = (taylor2 - (prev_model_output - prev_prev_model_output) / h_left_12) / h_left_02
                        x_t = sample + h1 * taylor1 + h1**2 * taylor2 / 2 + h1**2 * h2 * taylor3 / 6

            else:
                raise NotImplementedError(f"Multistep UniCI predict with order {order} is not implemented yet.")

        # The format of predictor solvers in DPM-Solver.
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int = None,  # pyright: ignore
        **kwargs,
    ) -> torch.Tensor:
        """
        Multistep universal `C` for updating the corrector solver.

        Args:
            this_model_output (`torch.Tensor`):
                Direct output from the model of the current scale.
            last_sample (`torch.Tensor`, *optional*):
                Sample from the previous scale.
            this_sample (`torch.Tensor`, *optional*):
                Current sample.
            order (`int`, *optional*):
                Order of the solver. If `None`, it will be set based on scheduler configuration.

        Returns:
            `torch.Tensor`: The updated sample for the corrector solver.
        """
        if order is None:
            order = self.config.solver_order
        # Similar structure as the universal P
        # Convert to flow matching format
        this_model_output = self.convert_model_output(this_model_output, sample=this_sample)

        if self._step_index > self.num_inference_steps - 1:
            prev_time_step = torch.tensor(0.0)
        else:
            prev_time_step = self.timesteps[self._step_index + 1].to(this_sample.device, this_model_output.dtype)

        time_step = self.timesteps[self._step_index].to(this_sample.device, this_model_output.dtype)
        dt = prev_time_step - time_step

        if order == 1:
            model_output_processor = this_model_output
            # Model output is scaled if we used noise with multiscale
            # Corrector
            if self.config.solver_type == "bh1":
                # Normal euler step to compute corrector (UniC)
                x_t = last_sample + dt * model_output_processor
            elif self.config.solver_type == "bh2":
                # Midpoint method for Heun's 2nd order method
                midpoint_model_output = 0.5 * (model_output_processor + this_model_output)
                # Runge-Kutta 2nd order
                x_t = last_sample + dt * midpoint_model_output
        else:  # order > 1:
            self.timestep_list.append(time_step)
            self.timestep_list.pop(0)
            self.model_outputs.append(this_model_output.to(last_sample.dtype))
            self.model_outputs.pop(0)

            current_model_output = this_model_output
            prev_model_output = self.model_outputs[-2]

            time_coe = dt

            rhos = self.sigmas[self._step_index - 1] / self.sigmas[self._step_index]
            rhos = rhos.to(last_sample.device, last_sample.dtype)

            # t -> t + 1
            if order == 2:
                # Bh1
                if self.config.solver_type == "bh1":
                    # Taylor expansion
                    h_tau = time_coe
                    h_phi = time_coe

                    # 2nd order expansion
                    x_t = (
                        last_sample
                        + h_phi * current_model_output
                        + 0.5 * h_phi**2 * (current_model_output - prev_model_output) / dt
                    )
                elif self.config.solver_type == "bh2":
                    h_t, h_t_1 = time_step, self.timestep_list[-2]
                    # r = rhos
                    h = h_t - h_t_1
                    x_t = (
                        last_sample
                        + h * current_model_output
                        - 0.5 * h**2 * (current_model_output - prev_model_output) / (h_t - h_t_1)
                    )
            elif order == 3:
                prev_prev_model_output = self.model_outputs[-3]
                h_t, h_t_1, h_t_2 = time_step, self.timestep_list[-2], self.timestep_list[-3]
                # r, r_1 = rhos, self.sigmas[self._step_index - 2] / self.sigmas[self._step_index - 1]
                _, r_1 = rhos, self.sigmas[self._step_index - 2] / self.sigmas[self._step_index - 1]
                r_1 = r_1.to(last_sample.device, last_sample.dtype)

                # Original DPM Solver ++: https://github.com/LuChengTHU/dpm-solver
                if self.config.solver_type == "bh1":
                    # Taylor expansion
                    h_tau = time_coe
                    h_phi = time_coe
                    h = h_t_1 - h_t_2
                    derivative2 = (current_model_output - prev_model_output) / (h_t - h_t_1)
                    derivative3 = (derivative2 - (prev_model_output - prev_prev_model_output) / (h_t_1 - h_t_2)) / (
                        h_t - h_t_2
                    )
                    x_t = (
                        last_sample
                        + h_tau * current_model_output
                        + 0.5 * h_tau**2 * derivative2
                        + (1.0 / 6.0) * h_tau**3 * derivative3
                    )
                elif self.config.solver_type == "bh2":
                    # From UniC paper: https://github.com/wl-zhao/UniPC
                    h1 = h_t - h_t_1
                    h2 = h_t_1 - h_t_2
                    h_left_01 = h_t - h_t_1
                    h_left_12 = h_t_1 - h_t_2
                    h_left_02 = h_t - h_t_2
                    taylor1 = current_model_output
                    taylor2 = (current_model_output - prev_model_output) / h_left_01
                    taylor3 = (taylor2 - (prev_model_output - prev_prev_model_output) / h_left_12) / h_left_02
                    x_t = last_sample + h1 * taylor1 + h1**2 * taylor2 / 2 + h1**2 * h2 * taylor3 / 6
            else:
                raise NotImplementedError(f"Multistep UniCI predict with order {order} is not implemented yet.")

        # The format of corrector solvers in DPM-Solver.
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        if self.begin_index is not None:
            indices = indices[self.begin_index :]

        if len(indices) == 0:
            raise ValueError(
                f"could not find timestep {timestep} from self.timesteps, Currently, self.timesteps have shape {self.timesteps.shape}, "
                f"and set scale to {self.config.set_scale}"
            )
        return indices[0].item()

    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = self.index_for_timestep(timestep)
        self._step_index = step_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`torch.Tensor` or `int`):
                The discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # Initialize the step_index if performing the first step
        if self._step_index is None:
            if self.begin_index is None:
                self._step_index = 0
            else:
                self._step_index = self.begin_index
            self._init_step_index(timestep)

        # Upcast sample and model_output to float32 for self.sigmas
        sample = sample.to(self.sigmas.dtype)
        model_output = model_output.to(self.sigmas.dtype)

        # Apply predicctor (P): x_t -> x_t-1
        if self.config.lower_order_final and self._step_index > self.num_inference_steps - 4:
            # For DPM-Solver++(2S), we use lower order solver for the final steps to stabilize the long time inference
            # it is equivalent to use our coefficients but change the order
            target_order = min(int(self.config.solver_order - 1), 2)

            # 3rd order method + 2nd order + 1st order
            if self.config.solver_order > 2 and self._step_index > self.num_inference_steps - 2:
                # set order to 1 for the final step
                target_order = min(int(target_order - 1), 2)

            # Switch to lower order for DPM-Solver++(2S) in the final steps to stabilize the long time inference
            lower_order_predict = self.multistep_uni_p_bh_update(
                model_output=model_output, sample=sample, order=target_order
            )
            next_sample = lower_order_predict
        else:
            this_predict = self.multistep_uni_p_bh_update(
                model_output=model_output, sample=sample, order=self.config.solver_order
            )
            next_sample = this_predict

        # Apply a corrector
        if self._step_index not in self.config.disable_corrector:
            # UniCPC
            # predictor: x_1 -> x_t-1, corrector: x_1 -> x_t-1
            if self.solver_p:
                # solver_p_output = self.solver_p.step(model_output, timestep, sample, return_dict=False)[0]
                _ = self.solver_p.step(model_output, timestep, sample, return_dict=False)[0]
                next_sample = self.multistep_uni_c_bh_update(
                    this_model_output=model_output,
                    last_sample=next_sample,
                    this_sample=sample,
                    order=self.config.solver_order,
                )

        # update step index
        self._step_index += 1
        self.last_sample = sample

        if not return_dict:
            return (next_sample,)

        return SchedulerOutput(prev_sample=next_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if timesteps.device != sigmas.device:
            timesteps = timesteps.to(sigmas.device)
        if timesteps.dtype != torch.int64:
            timesteps = timesteps.to(torch.int64)

        schedule_timesteps = self.timesteps

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma

        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
