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

from typing import Optional, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FlowMapEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler-style sampler for flow-map-distilled diffusion models.

    Flow-map models learn arbitrary-interval transitions :math:`z_t \\to z_r` rather than the fixed
    :math:`z_t \\to z_0` mapping of consistency models, so a single distilled checkpoint can be evaluated at
    1, 2, 4, 8, ... NFE without retraining. The `step` method advances the sample from `timestep` to
    `r_timestep` along the predicted velocity.

    Introduced in
    [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://huggingface.co/papers/<arxiv-id>).

    This scheduler inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the
    generic methods implemented for all schedulers (loading, saving, etc.).

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps used to train the underlying flow-map model.
        shift (`float`, defaults to 1.0):
            Multiplicative timestep shift applied to the inference schedule. ``shift=1.0`` is the identity; values
            greater than 1.0 push the schedule toward more denoising at later steps (e.g., ``shift=5`` matches the
            Wan2.1 default).
        weight_type (`str`, defaults to `"gaussian"`):
            Loss-weighting scheme for training. ``"gaussian"`` uses logit-normal weighting centered at
            ``num_train_timesteps / 2``. ``"beta08"`` uses a beta(1.0, 0.5)-shaped weighting biased toward small
            timesteps.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        weight_type: str = "gaussian",
    ):
        self.set_timesteps(num_train_timesteps, device="cpu")
        self.set_train_weight(weight_type)

    def adaptive_weighting(self, loss, p=1.0, eps=1e-3):
        """Inverse-loss reweighting used during distillation training."""
        weight = 1.0 / torch.pow(loss.detach() + eps, p)
        return weight * loss

    def set_train_weight(self, weight_type):
        """Precompute per-timestep training loss weights."""
        if self.config.weight_type == "gaussian":
            x = self.timesteps
            y = torch.exp(-2 * ((x - self.config.num_train_timesteps / 2) / self.config.num_train_timesteps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (self.config.num_train_timesteps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
        elif self.config.weight_type == "beta08":
            t = self.timesteps / self.config.num_train_timesteps
            y = (t**1.0) * ((1 - t) ** 0.5)
            self.linear_timesteps_weights = y * (self.config.num_train_timesteps / y.sum())
        else:
            raise ValueError(f"Invalid weight type: {weight_type}")

    @torch.no_grad()
    def get_train_weight(self, timesteps):
        """Return the precomputed loss weight for each entry in ``timesteps``."""
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timesteps.flatten().unsqueeze(0).to(self.timesteps.device)).abs(),
            dim=0,
        ).reshape(timesteps.shape)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights.to(timesteps.device)

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

    def apply_shift(self, sigmas):
        """Apply the configured shift transformation to a sigma tensor."""
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
    ):
        """Build the inference timestep schedule on ``device`` and store it on ``self.timesteps``."""
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device)
        timesteps = self.apply_shift(timesteps)

        self.timesteps = timesteps * self.config.num_train_timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        timestep: Optional[Union[float, torch.FloatTensor]] = None,
        r_timestep: Optional[Union[float, torch.FloatTensor]] = None,
    ):
        """
        Advance ``sample`` from ``timestep`` to ``r_timestep`` using the model-predicted velocity.

        Unlike a standard Euler scheduler, both endpoints of the interval are caller-provided so that any-step
        sampling is possible: a single model call can step from `t` to any chosen target `r` (including `r=0` for
        a one-shot generation).
        """
        timestep = timestep / self.config.num_train_timesteps
        r_timestep = r_timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        r_timestep = r_timestep.view(*r_timestep.shape, *([1] * (model_output.ndim - r_timestep.ndim)))
        prev_sample = sample - (timestep - r_timestep) * model_output
        return prev_sample.to(model_output.dtype)
