# Copyright (C) 2026 Boogu Team.
#
# This file is adapted by Boogu Team from prior open-source scheduler work.
# Boogu uses the standard flow-matching Euler scheduler; the only Boogu-specific
# piece is the time convention (sigma runs 0 -> 1 and is fed to the transformer
# directly), so this is a thin subclass of the built-in
# `FlowMatchEulerDiscreteScheduler` that reuses its `step` and time-shift math.
#
# Original work:
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team.
# All rights reserved.
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

import numpy as np
import torch

from ..configuration_utils import register_to_config
from ..utils import logging
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class BooguFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """Flow-matching Euler scheduler with Boogu's training-time convention.

    Boogu trains with a sigma schedule that runs ``0 -> 1`` and feeds that sigma
    to the transformer as the timestep directly (unlike the built-in scheduler,
    whose timesteps run ``1000 -> 0``). The denoising step and the time-shift
    formula are identical to the built-in scheduler, so this subclass only
    overrides ``set_timesteps`` to produce the Boogu-convention schedule and
    reuses the parent ``step`` (and step-index bookkeeping) unchanged.

    The released checkpoints use a static ``v1`` time shift only. ``v1`` is the
    parent's ``exponential`` shift applied with the time axis reversed
    (``t -> 1 - t``); the parent's ``_time_shift_exponential`` is reused for it.
    Dynamic / ``v2`` configurations are not supported and raise at construction.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        do_shift: bool = True,
        dynamic_time_shift: bool = False,
        time_shift_version: str = "v1",
        seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        time_shift_v2_half_scaling_factor: float = 60.0,
    ):
        # use_dynamic_shifting=True keeps the parent from applying its own static
        # `shift`; Boogu applies the shift itself inside `set_timesteps`.
        super().__init__(num_train_timesteps=num_train_timesteps, use_dynamic_shifting=True)
        if dynamic_time_shift or time_shift_version != "v1":
            raise ValueError(
                "BooguFlowMatchEulerDiscreteScheduler only supports static v1 time-shifting "
                "(do_shift=True, dynamic_time_shift=False, time_shift_version='v1'); "
                f"got dynamic_time_shift={dynamic_time_shift}, time_shift_version={time_shift_version!r}."
            )

    @staticmethod
    def _get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[list] = None,
        num_tokens: Optional[int] = None,
    ):
        """Set the discrete timesteps (Boogu convention: sigma runs 0 -> 1)."""
        self.num_inference_steps = num_inference_steps
        t = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]

        if self.config.do_shift:
            mu = self._get_lin_function(y1=self.config.base_shift, y2=self.config.max_shift)(self.config.seq_len)
            # Boogu v1 == 1 - exponential_shift(1 - t); reuse the parent's formula.
            t = (1.0 - self._time_shift_exponential(mu, 1.0, 1.0 - torch.from_numpy(t))).numpy()

        sigmas = torch.from_numpy(t).to(dtype=torch.float32, device=device)
        self.timesteps = sigmas  # 0-1 sigma, fed to the transformer as the timestep
        self.sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        self._step_index = None
        self._begin_index = None
