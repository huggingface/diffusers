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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..models.hooks import ModelHook, add_hook_to_module
from ..utils import logging
from .pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class TeaCacheConfig:
    pass


class TeaCacheDenoiserState:
    def __init__(self):
        self.iteration = 0
        self.accumulated_l1_difference = 0.0
        self.timestep_modulated_cache = None
    
    def reset(self):
        self.iteration = 0
        self.accumulated_l1_difference = 0.0
        self.timestep_modulated_cache = None


def apply_teacache(pipeline: DiffusionPipeline, config: TeaCacheConfig, denoiser: Optional[nn.Module]) -> None:
    r"""Applies [TeaCache]() to a given pipeline or denoiser module.
    
    Args:
        TODO
    """
