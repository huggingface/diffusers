# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class UCGMScheduler(SchedulerMixin, ConfigMixin):
    """
    UCGM-S (Unified Continuous Generative Models - Sampler) scheduler.

    This scheduler implements the sampling strategy proposed in UCGM.
    It supports Kumaraswamy time warping, extrapolation for acceleration,
    and stochastic sampling.

    References:
        - UCGM (Sampler): https://arxiv.org/abs/2505.07447
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        sampling_steps: int = 50,
        stochast_ratio: float = 0.0,
        extrapol_ratio: float = 0.0,
        time_dist_ctrl: List[float] = [1.17, 0.8, 1.1],
        rfba_gap_steps: List[float] = [0.0, 0.0],
        sampling_style: str = "mul",  # "few", "mul", "any"
        integ_st: int = 1,  # Integration start direction (1 or 0)
        **kwargs,
    ):
        """
        Args:
            sampling_steps (`int`, defaults to 50):
                The number of sampling steps.
            stochast_ratio (`float`, defaults to 0.0):
                The ratio of stochastic noise added during sampling.
            extrapol_ratio (`float`, defaults to 0.0):
                The ratio for extrapolation to accelerate convergence based on history.
            time_dist_ctrl (`List[float]`, defaults to [1.0, 1.0, 1.0]):
                Parameters [a, b, c] for the Kumaraswamy transform.
            rfba_gap_steps (`List[float]`, defaults to [0.0, 0.0]):
                Gap steps for the RFBA boundary handling.
            sampling_style (`str`, defaults to "few"):
                Controls the 't' passed to the model. Options: "few", "mul", "any".
            integ_st (`int`, defaults to 1):
                Integration start flag. 1 implies standard Rectified Flow direction.
        """
        self.timesteps = None
        self.sigmas = None
        self.x_hats = []
        self.z_hats = []
        self.buffer_freq = 1
        self.begin_index = None
        self.current_sigma = None
        self.next_sigma = None

    def kumaraswamy_transform(self, t: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
        """
        Applies the Kumaraswamy transform to the time steps.
        """
        return (1 - (1 - t**a) ** b) ** c

    def alpha_in(self, t):
        # If integ_st == 0, we swap alpha_in and gamma_in.
        # So alpha_in becomes base gamma_in (1-t)
        if self.config.integ_st == 0:
            return 1 - t
        return t

    def gamma_in(self, t):
        # If integ_st == 0, we swap.
        # So gamma_in becomes base alpha_in (t)
        if self.config.integ_st == 0:
            return t
        return 1 - t

    def alpha_to(self, t):
        # If integ_st == 0, swap alpha_to and gamma_to.
        # So alpha_to becomes base gamma_to (-1)
        if self.config.integ_st == 0:
            return -1.0
        return 1.0

    def gamma_to(self, t):
        # If integ_st == 0, swap.
        # So gamma_to becomes base alpha_to (1)
        if self.config.integ_st == 0:
            return 1.0
        return -1.0

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        timesteps: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain.
        Strictly aligned with UCGM sampling_loop logic.
        """
        # 1. Handle Custom Inputs (sigmas / timesteps)
        if sigmas is not None or timesteps is not None:
            # Type conversion to Tensor (Float64)
            if sigmas is not None:
                if not isinstance(sigmas, torch.Tensor):
                    sigmas = torch.tensor(sigmas, dtype=torch.float64, device=device)
                else:
                    sigmas = sigmas.to(device=device, dtype=torch.float64)

            if timesteps is not None:
                if not isinstance(timesteps, torch.Tensor):
                    timesteps = torch.tensor(timesteps, dtype=torch.float64, device=device)
                else:
                    timesteps = timesteps.to(device=device, dtype=torch.float64)

            # Consistency checks
            if sigmas is not None and timesteps is not None:
                if len(sigmas) != len(timesteps):
                    raise ValueError("`sigmas` and `timesteps` must have the same length")

            # Derive sigmas from timesteps if missing
            if sigmas is None and timesteps is not None:
                sigmas = timesteps / 1000.0

            # Apply Time Distribution Control (Kumaraswamy) to Custom Inputs
            # UCGM logic operates on t in [0, 1] (ascending), where sigma = 1 - t.
            # We assume custom sigmas are roughly 1 -> 0 (descending noise level).

            # Convert sigma to t (0->1)
            t_input = (1.0 - sigmas).clamp(0.0, 1.0)

            # Apply transform
            t_transformed = self.kumaraswamy_transform(t_input, *self.config.time_dist_ctrl)

            # Convert back to sigma (1->0)
            sigmas = 1.0 - t_transformed

            # [CRITICAL] Ensure tail is 0.0 to enable the final cleanup step
            if sigmas[-1].abs() > 1e-6:
                sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)])

            self.sigmas = sigmas
            # Recalculate timesteps from the potentially modified sigmas (with 0.0 appended)
            self.timesteps = (self.sigmas * 1000.0).to(device)

            # Reset Buffers
            self.x_hats = []
            self.z_hats = []
            self.begin_index = None
            return

        # 2. Generation Logic (Default - Kumaraswamy + RFBA)
        if num_inference_steps is None:
            num_inference_steps = self.config.sampling_steps

        rfba_gap_steps = self.config.rfba_gap_steps
        time_dist_ctrl = self.config.time_dist_ctrl

        # Calculate actual steps (Original Logic)
        actual_steps = num_inference_steps

        if abs(rfba_gap_steps[1] - 0.0) < 1e-9:
            actual_steps = actual_steps + 1

        # Generate Linspace (Float64 for precision)
        t_start = rfba_gap_steps[0]
        t_end = 1.0 - rfba_gap_steps[1]

        t_steps = torch.linspace(t_start, t_end, actual_steps, dtype=torch.float64, device=device)

        if abs(rfba_gap_steps[1] - 0.0) < 1e-9:
            t_steps = t_steps[:-1]

        # Apply Kumaraswamy Transform
        t_steps = self.kumaraswamy_transform(t_steps, *time_dist_ctrl)

        # Generate Sigmas (Noise Levels) and append 0.0
        # sigmas represents 't' in the noise schedule (from 1.0 down to 0.0)
        sigmas = 1.0 - t_steps
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)])

        # Deduplicate final 0.0 if necessary (though linspace logic above handles most cases)
        if len(sigmas) > 1 and torch.abs(sigmas[-1] - sigmas[-2]) < 1e-6:
            sigmas = sigmas[:-1]

        self.sigmas = sigmas

        # Set self.timesteps
        self.timesteps = (self.sigmas * 1000.0).to(device)

        # Reset Buffers
        self.x_hats = []
        self.z_hats = []
        self.begin_index = None

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def set_begin_index(self, begin_index: int = 0):
        self.begin_index = begin_index

    def get_next_timestep(self):
        return self.next_sigma * 1000.0

    def get_current_timestep(self):
        return self.current_sigma * 1000.0

    def get_next_sigma(self):
        return self.next_sigma

    def get_current_sigma(self):
        return self.current_sigma

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep.
        """
        if self.timesteps is None:
            raise ValueError("`self.timesteps` is not set.")

        # --- 1. Robust Step Index Lookup ---
        if torch.is_tensor(timestep) and timestep.ndim > 0:
            timestep = timestep[0]

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.timesteps.device, dtype=torch.float64)
        else:
            timestep = timestep.to(self.timesteps.device, dtype=torch.float64)

        # Find closest step
        step_index = (self.timesteps - timestep).abs().argmin().item()

        # Boundary check: if we are at the very last step (t=0), returns sample
        if step_index >= len(self.timesteps) - 1:
            return SchedulerOutput(prev_sample=sample)

        # --- 2. Prepare Timesteps (Float64) ---
        t_cur = self.sigmas[step_index].abs()
        t_next = self.sigmas[step_index + 1].abs()
        self.current_sigma = t_cur
        self.next_sigma = t_next

        # --- 3. Internal Calculation (Strict Float64) ---
        F_t = ((-1) ** (1 - self.config.integ_st)) * model_output.to(torch.float64)
        sample_f64 = sample.to(torch.float64)

        # Coefficients
        g_to = self.gamma_to(t_cur)
        g_in = self.gamma_in(t_cur)
        a_in = self.alpha_in(t_cur)
        a_to = self.alpha_to(t_cur)

        # Derived dent: alpha_in * gamma_to - gamma_in * alpha_to
        # Standard: t*(-1) - (1-t)*1 = -1.
        # Swapped: (1-t)*1 - t*(-1) = 1-t+t = 1.
        # We calculate it dynamically to be safe.
        dent = a_in * g_to - g_in * a_to

        # Reconstruct x_hat, z_hat
        z_hat = (sample_f64 * g_to - F_t * g_in) / dent
        x_hat = (F_t * a_in - sample_f64 * a_to) / dent

        # --- 4. Extrapolation Logic (Aligned with original index i) ---
        extrapol_ratio = self.config.extrapol_ratio
        if self.buffer_freq > 0 and extrapol_ratio > 0:
            self.z_hats.append(z_hat)
            self.x_hats.append(x_hat)

            if len(self.x_hats) > self.buffer_freq + 1:
                idx_prev = -(self.buffer_freq + 1)
                z_prev = self.z_hats[idx_prev]
                x_prev = self.x_hats[idx_prev]

                z_hat = z_hat + extrapol_ratio * (z_hat - z_prev)
                x_hat = x_hat + extrapol_ratio * (x_hat - x_prev)

                self.z_hats.pop(0)
                self.x_hats.pop(0)

        # --- 5. Stochasticity ---
        stochast_ratio = self.config.stochast_ratio
        if isinstance(stochast_ratio, (float, int)) and stochast_ratio > 0:
            noi = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            noi = noi.to(torch.float64)  # Cast to calculation dtype
        else:
            noi = torch.zeros_like(sample_f64)
            stochast_ratio = 0.0

        # --- 6. Update Step (Euler) ---
        g_in_next = self.gamma_in(t_next)
        a_in_next = self.alpha_in(t_next)

        z_term = z_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
        prev_sample = g_in_next * x_hat + a_in_next * z_term

        # Cast back to input dtype only at the very end
        prev_sample = prev_sample.to(sample.dtype)

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.sampling_steps
