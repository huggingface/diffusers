# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMapEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor`):
            Computed sample :math:`z_r` at the target flow-map timestep `r_timestep`. Should be used as the
            next denoising input.
    """

    prev_sample: torch.Tensor


class FlowMapEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler-style sampler for flow-map-distilled diffusion models.

    Flow-map models learn arbitrary-interval transitions :math:`z_t \\to z_r` rather than the fixed
    :math:`z_t \\to z_0` mapping of consistency models, so a single distilled checkpoint can be evaluated at
    1, 2, 4, 8, ... NFE without retraining. The `step` method advances the sample from `timestep` to
    `r_timestep` along the predicted velocity.

    Introduced in
    [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://huggingface.co/papers/2605.13724).

    This scheduler inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the
    generic methods implemented for all schedulers (loading, saving, etc.).

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps used to train the underlying flow-map model.
        shift (`float`, defaults to 1.0):
            Multiplicative timestep shift applied to the inference schedule. ``shift=1.0`` is the identity; values
            greater than 1.0 push the schedule toward more denoising at later steps (e.g., ``shift=5`` matches the
            Wan2.1 default).
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        self.set_timesteps(num_train_timesteps, device="cpu")

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """No-op identity scaling. Provided for API compatibility with other Diffusers schedulers."""
        return sample

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Linearly interpolate ``sample`` toward ``noise`` according to the normalized ``timestep``."""
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)

        timestep = timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        sample = timestep * noise + (1.0 - timestep) * sample
        return sample

    def apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Apply the configured shift transformation to a sigma tensor."""
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
    ) -> None:
        """Build the inference timestep schedule on ``device`` and store it on ``self.timesteps``."""
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device)
        timesteps = self.apply_shift(timesteps)

        self.timesteps = timesteps * self.config.num_train_timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        r_timestep: Optional[Union[float, torch.FloatTensor]] = None,
        return_dict: bool = True,
    ) -> Union[FlowMapEulerDiscreteSchedulerOutput, Tuple[torch.Tensor]]:
        """
        Advance ``sample`` from ``timestep`` to ``r_timestep`` using the model-predicted velocity.

        Unlike a standard Euler scheduler, both endpoints of the interval are caller-provided so that any-step
        sampling is possible: a single model call can step from `t` to any chosen target `r` (including `r=0`
        for a one-shot generation).

        Args:
            model_output (`torch.Tensor`):
                Direct output from the flow-map model (predicted mean velocity).
            timestep (`float` or `torch.Tensor`):
                Source timestep ``t`` in the same units as ``self.timesteps``.
            sample (`torch.Tensor`):
                Current sample :math:`z_t`.
            r_timestep (`float` or `torch.Tensor`):
                Target timestep ``r``. Must be provided; passing ``r_timestep=timestep`` is a no-op.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`FlowMapEulerDiscreteSchedulerOutput`] (the default) or a plain tuple.

        Returns:
            [`FlowMapEulerDiscreteSchedulerOutput`] or `tuple`:
                When ``return_dict=True``, returns a [`FlowMapEulerDiscreteSchedulerOutput`] whose
                ``prev_sample`` is :math:`z_r`. Otherwise returns a 1-tuple ``(prev_sample,)``.
        """
        if r_timestep is None:
            raise ValueError(
                "`FlowMapEulerDiscreteScheduler.step` requires an explicit `r_timestep`; this scheduler does "
                "not infer the target timestep from internal state."
            )
        timestep = timestep / self.config.num_train_timesteps
        r_timestep = r_timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        r_timestep = r_timestep.view(*r_timestep.shape, *([1] * (model_output.ndim - r_timestep.ndim)))
        prev_sample = sample - (timestep - r_timestep) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMapEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
