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
from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp

from ..utils import BaseOutput


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


@dataclass
class FlaxSchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.

    Args:
        prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: jnp.ndarray


class FlaxSchedulerMixin:
    """
    Mixin containing common functions for the schedulers.
    """

    config_name = SCHEDULER_CONFIG_NAME


def broadcast_to_shape_from_left(x: jnp.ndarray, shape: Tuple[int]) -> jnp.ndarray:
    assert len(shape) >= x.ndim
    return jnp.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)
