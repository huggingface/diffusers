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

import math

import torch

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, GuiderOutput


class MagiClassifierFreeGuidance(BaseGuidance):
    """
    Combine text-and-prefix, prefix-only, and independent-chunk velocities for MAGI base models.

    Args:
        timestep_thresholds (`tuple[float]`, defaults to `(0.0, 0.0217, 0.1, 0.3, 0.999)`):
            Increasing interval boundaries in normalized noise-to-clean time, starting at zero.
        prefix_scales (`tuple[float]`, defaults to `(1.5, 1.5, 1.5, 1.0, 1.0)`):
            Prefix guidance scales, one per timestep interval.
        text_scales (`tuple[float]`, defaults to `(7.5, 7.5, 7.5, 0.0, 0.0)`):
            Text guidance scales, one per timestep interval.

    Call `set_state` with times shaped `(batch, chunks)` before applying guidance. All three branches remain available
    for clean-cache management, even when a guidance coefficient is zero.
    """

    _input_predictions = ["pred_cond", "pred_prefix", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        timestep_thresholds=(0.0, 0.0217, 0.1, 0.3, 0.999),
        prefix_scales=(1.5, 1.5, 1.5, 1.0, 1.0),
        text_scales=(7.5, 7.5, 7.5, 0.0, 0.0),
    ):
        super().__init__()
        if (
            not timestep_thresholds
            or len(prefix_scales) != len(timestep_thresholds)
            or len(text_scales) != len(timestep_thresholds)
        ):
            raise ValueError("Thresholds and guidance scales must have the same nonzero length.")
        if not all(
            math.isfinite(value) for values in (timestep_thresholds, prefix_scales, text_scales) for value in values
        ):
            raise ValueError("Thresholds and guidance scales must be finite.")
        if timestep_thresholds[0] != 0 or any(a >= b for a, b in zip(timestep_thresholds, timestep_thresholds[1:])):
            raise ValueError("Timestep thresholds must start at zero and increase strictly.")

    @property
    def num_conditions(self):
        return 3

    @property
    def is_conditional(self):
        return self._count_prepared == 1

    def prepare_inputs(self, data):
        return [self._prepare_batch(data, i, name) for i, name in enumerate(self._input_predictions)]

    def prepare_inputs_from_block_state(self, data, input_fields):
        return [
            self._prepare_batch_from_block_state(input_fields, data, i, name)
            for i, name in enumerate(self._input_predictions)
        ]

    def forward(self, pred_cond, pred_prefix, pred_uncond):
        if self._timestep is None:
            raise ValueError("Set the current chunk timesteps with set_state before applying guidance.")
        times = self._timestep.to(device=pred_cond.device, dtype=torch.float32)
        if times.ndim == 1:
            times = times[None].expand(pred_cond.shape[0], -1)
        if (
            times.ndim != 2
            or times.shape[0] != pred_cond.shape[0]
            or times.shape[1] == 0
            or pred_cond.shape[2] % times.shape[1]
        ):
            raise ValueError("Guidance timesteps must match the batch and divide the latent frames into chunks.")
        thresholds = torch.tensor(self.config.timestep_thresholds, device=times.device, dtype=torch.float32)
        indices = torch.searchsorted(thresholds - 1e-7, times.contiguous()) - 1
        indices = indices.clamp(0, len(thresholds) - 1)
        frames_per_chunk = pred_cond.shape[2] // times.shape[1]
        prefix_scale = torch.tensor(self.config.prefix_scales, device=times.device, dtype=torch.float32)[indices]
        text_scale = torch.tensor(self.config.text_scales, device=times.device, dtype=torch.float32)[indices]
        prefix_scale = prefix_scale.repeat_interleave(frames_per_chunk, dim=1)[:, None, :, None, None]
        text_scale = text_scale.repeat_interleave(frames_per_chunk, dim=1)[:, None, :, None, None]
        pred = (1 - prefix_scale) * pred_uncond.float()
        pred = pred + (prefix_scale - text_scale) * pred_prefix.float() + text_scale * pred_cond.float()
        if not self._enabled:
            pred = pred_cond
        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)
