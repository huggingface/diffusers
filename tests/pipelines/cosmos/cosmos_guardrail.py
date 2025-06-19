# Copyright 2025 The HuggingFace Team.
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

# ===== This file is an implementation of a dummy guardrail for the fast tests =====

from typing import Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


class DummyCosmosSafetyChecker(ModelMixin, ConfigMixin):
    def __init__(self) -> None:
        super().__init__()

        self._dtype = torch.float32

    def check_text_safety(self, prompt: str) -> bool:
        return True

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def to(self, device: Union[str, torch.device] = None, dtype: torch.dtype = None) -> None:
        self._dtype = dtype

    @property
    def device(self) -> torch.device:
        return None

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
