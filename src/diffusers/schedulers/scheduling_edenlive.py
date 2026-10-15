# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils.torch_utils import randn_tensor
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from .scheduling_utils import SchedulerMixin


class EdenLiveFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow Euler scheduler for EdenLive, including its distilled stochastic trajectory.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            Scale used to convert model timesteps to noise levels.
        shift (`float`, defaults to 3.0):
            Flow shift applied to generated schedules. Explicit timesteps and sigmas are already in model space.
        stochastic_sampling (`bool`, defaults to `False`):
            Predict a clean endpoint and re-noise it between model evaluations.
        default_timesteps (`list[float]`, *optional*):
            Model timesteps used by a distilled checkpoint. Their length fixes the default number of inference steps.
    """

    order = 1
    _compatibles = []

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        stochastic_sampling: bool = False,
        default_timesteps: list[float] | None = None,
    ):
        if num_train_timesteps < 1 or shift <= 0:
            raise ValueError("`num_train_timesteps` and `shift` must be positive.")
        self.num_inference_steps = None
        self.init_noise_sigma = 1.0
        self._step_index = None
        self._begin_index = None
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self.sigmas = torch.empty(0, dtype=torch.float32)

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        if not 0 <= begin_index < len(self.timesteps):
            raise ValueError("`begin_index` must select an inference timestep.")
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[float] | None = None,
        sigmas: list[float] | None = None,
    ):
        """Set model evaluation points and reset the step and begin indices for a new video or chunk.

        Args:
            num_inference_steps (`int`, *optional*):
                Number of model evaluations. Required when neither an explicit nor a configured default schedule is
                available. When a schedule is provided, this value must match its number of positive evaluation points.
            device (`str` or `torch.device`, *optional*):
                Device for the FP32 `timesteps` and `sigmas` tensors.
            timesteps (`list[float]`, *optional*):
                Strictly decreasing model timesteps in [0, `num_train_timesteps`]. They are divided by
                `num_train_timesteps` to obtain sigmas and are not shifted again. An optional final zero is removed
                from the model evaluation points.
            sigmas (`list[float]`, *optional*):
                Strictly decreasing noise levels in [0, 1], optionally ending in zero. Mutually exclusive with
                `timesteps`. All values must be finite. When both arguments are omitted, `default_timesteps` takes
                precedence; otherwise a schedule is generated using `shift` and `num_inference_steps`.

        The internal `sigmas` tensor includes a terminal zero after the positive model evaluation points.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError("Pass only one of `timesteps` or `sigmas`.")
        if timesteps is None and sigmas is None and self.config.default_timesteps is not None:
            timesteps = self.config.default_timesteps
        if timesteps is not None or sigmas is not None:
            values = torch.as_tensor(
                timesteps if timesteps is not None else sigmas, dtype=torch.float32, device=device
            )
            if values.ndim != 1 or values.numel() == 0:
                raise ValueError("The schedule must be a non-empty one-dimensional sequence.")
            if timesteps is not None:
                values = values / self.config.num_train_timesteps
            if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > 1):
                raise ValueError("Schedule sigmas must be finite and lie in [0, 1].")
            if torch.any(values[1:] >= values[:-1]):
                raise ValueError("Schedule sigmas must be strictly decreasing.")
            if values[-1] == 0:
                values = values[:-1]
            if values.numel() == 0 or torch.any(values == 0):
                raise ValueError("The schedule must contain positive model evaluation sigmas.")
            if num_inference_steps is not None and num_inference_steps != len(values):
                raise ValueError("`num_inference_steps` must match the number of model evaluation points.")
        else:
            if num_inference_steps is None or num_inference_steps < 1:
                raise ValueError("Pass a positive `num_inference_steps` or an explicit schedule.")
            values = torch.linspace(
                1.0, 1.0 / self.config.num_train_timesteps, num_inference_steps + 1, dtype=torch.float32, device=device
            )[:-1]
            values = self.config.shift * values / (1.0 + (self.config.shift - 1.0) * values)
            values[0] -= 1.0e-6
        self.num_inference_steps = len(values)
        self.timesteps = values * self.config.num_train_timesteps
        self.sigmas = torch.cat([values, values.new_zeros(1)])
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        schedule_timesteps = self.timesteps if schedule_timesteps is None else schedule_timesteps
        timestep = torch.as_tensor(timestep, device=schedule_timesteps.device, dtype=schedule_timesteps.dtype)
        indices = (schedule_timesteps == timestep).nonzero().flatten()
        if len(indices) == 0:
            raise ValueError("The timestep is not in this scheduler's current schedule.")
        return indices[1 if len(indices) > 1 else 0].item()

    def scale_model_input(self, sample: torch.Tensor, timestep=None):
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        variance_noise: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        """Advance one velocity prediction, preserving the sample dtype.

        Call `set_timesteps` before stepping through a video or causal chunk. Stochastic sampling predicts a clean
        endpoint and adds fresh noise between model evaluations; the terminal step does not draw noise.

        Args:
            model_output (`torch.Tensor`):
                Predicted flow velocity, with the same shape as `sample`.
            timestep (`float` or `torch.Tensor`):
                Current model evaluation point from `scheduler.timesteps`.
            sample (`torch.Tensor`):
                Current noisy latent sample.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator, or one per batch item, for stochastic noise draws. Mutually exclusive with
                `variance_noise`. Deterministic sampling does not draw noise.
            variance_noise (`torch.Tensor`, *optional*):
                Precomputed noise for a nonterminal stochastic step, for example noise shared across distributed ranks.
                Must have the same shape and dtype as `sample` and be on its device. Ignored when no noise is needed.
            return_dict (`bool`, defaults to `True`):
                Return the scheduler output when true; otherwise return a one-element tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                The next sample in `prev_sample`, or the first tuple entry, with the same dtype as `sample`.
        """
        if self.num_inference_steps is None:
            raise ValueError("Call `set_timesteps` before `step`.")
        if variance_noise is not None and generator is not None:
            raise ValueError("Pass only one of `generator` or `variance_noise`.")
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index
        if self._step_index >= self.num_inference_steps:
            raise ValueError("The schedule is exhausted. Call `set_timesteps` for the next chunk.")
        sigma = self.sigmas[self._step_index].to(sample.device)
        sigma_next = self.sigmas[self._step_index + 1].to(sample.device)
        if self.config.stochastic_sampling:
            endpoint = sample - sigma.to(sample.dtype) * model_output
            if self._step_index + 1 == self.num_inference_steps:
                prev_sample = endpoint
            else:
                if variance_noise is None:
                    variance_noise = randn_tensor(
                        sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
                    )
                elif variance_noise.shape != sample.shape or variance_noise.dtype != sample.dtype:
                    raise ValueError("`variance_noise` must have the same shape and dtype as `sample`.")
                sigma_next = sigma_next.to(sample.dtype)
                prev_sample = (1.0 - sigma_next) * endpoint + sigma_next * variance_noise
        else:
            delta = (sigma_next - sigma).view(*([1] * sample.ndim))
            prev_sample = sample.to(model_output.dtype) + model_output * delta
        prev_sample = prev_sample.to(sample.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
