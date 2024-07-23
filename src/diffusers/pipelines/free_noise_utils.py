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

from typing import Optional


class FreeNoiseMixin:
    r"""Mixin class for [FreeNoise](https://arxiv.org/abs/2310.15169)."""

    def enable_free_noise(self, context_length: Optional[int] = 16, context_stride: int = 4, shuffle: bool = True) -> None:
        self._free_noise_context_length = context_length or self.motion_adapter.config.motion_max_seq_length
        self._free_noise_context_stride = context_stride
        self._free_noise_shuffle = shuffle

    def disable_free_noise(self) -> None:
        self._free_noise_context_length = None

    @property
    def free_noise_enabled(self):
        return hasattr(self, "_free_noise_context_length") and self._free_noise_context_length is not None
