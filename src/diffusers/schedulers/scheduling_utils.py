# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import abc
from dataclasses import dataclass

import torch

from ..utils import BaseOutput


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class BaseScheduler(abc.ABC):
    config_name = SCHEDULER_CONFIG_NAME

    def scale_initial_noise(self, noise: torch.FloatTensor):
        """
        Scales the initial noise to the correct range for the scheduler.
        """
        return noise

    def scale_model_input(self, sample: torch.FloatTensor, step: int):
        """
        Scales the model input (`sample`) to the correct range for the scheduler.
        """
        return sample

    @abc.abstractmethod
    def get_noise_condition(self, step: int):
        """
        Returns the input noise condition for the model (e.g. `timestep` or `sigma`).
        """
        raise NotImplementedError("Scheduler must implement the `get_noise_condition` function.")


class SchedulerMixin:
    """
    Mixin containing common functions for the schedulers.
    """

    config_name = SCHEDULER_CONFIG_NAME
