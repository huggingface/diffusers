# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: Copyright 2024 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def set_flow_match_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: int,
    device: str | torch.device | None = None,
    seq_len: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Set Boogu's training-aligned timesteps on the official flow-match scheduler.

    Boogu trains with a static ``v1`` time shift and a sigma schedule that runs
    ``0 -> 1``, feeding that sigma to the transformer as the timestep directly
    (unlike the built-in scheduler, whose timesteps run ``1000 -> 0``). The shift
    amount ``mu`` is a fixed function of ``seq_len`` (resolution-independent), and
    the shift itself reuses the parent's exponential formula. This overwrites the
    scheduler's ``timesteps`` / ``sigmas`` to that convention; ``step`` is the
    official one and works unchanged on the resulting schedule.

    Args:
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            The official scheduler whose schedule is overwritten in place.
        num_inference_steps (`int`):
            The number of denoising steps.
        device (`str` or `torch.device`, *optional*):
            The device the schedule is placed on.
        seq_len (`int`, *optional*):
            Image sequence length used to compute the static shift. Defaults to
            ``scheduler.config.seq_len``.

    Returns:
        `tuple[torch.Tensor, int]`: the timestep schedule and the number of steps.
    """
    if seq_len is None:
        seq_len = scheduler.config.seq_len

    # Static v1 shift: mu is a linear function of seq_len between (base_image_seq_len,
    # base_shift) and (max_image_seq_len, max_shift).
    slope = (scheduler.config.max_shift - scheduler.config.base_shift) / (
        scheduler.config.max_image_seq_len - scheduler.config.base_image_seq_len
    )
    mu = scheduler.config.base_shift + slope * (seq_len - scheduler.config.base_image_seq_len)

    t = np.linspace(0.0, 1.0, num_inference_steps + 1, dtype=np.float32)[:-1]
    # Boogu v1 == 1 - exponential_shift(mu, 1, 1 - t); reuse the parent's formula.
    t = (1.0 - scheduler._time_shift_exponential(mu, 1.0, 1.0 - torch.from_numpy(t))).numpy()

    timesteps = torch.from_numpy(t).to(dtype=torch.float32, device=device)
    scheduler.timesteps = timesteps  # 0-1 sigma, fed to the transformer as the timestep
    scheduler.sigmas = torch.cat([timesteps, torch.ones(1, device=timesteps.device)])
    scheduler.num_inference_steps = num_inference_steps
    scheduler._step_index = None
    scheduler._begin_index = None

    return scheduler.timesteps, num_inference_steps
