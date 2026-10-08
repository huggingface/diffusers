# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import numpy as np
import torch


class DummyPAN2SafetyChecker(torch.nn.Module):
    r"""
    Stands in for `pan2_guardrail.PAN2SafetyChecker`, whose models are too large for the fast tests, with the same
    interface. It records what it checks and returns the given verdicts.
    """

    def __init__(self, text_safe: bool = True, video_safe: bool = True) -> None:
        super().__init__()
        self.text_safe = text_safe
        self.video_safe = video_safe
        self.checked_prompts = []
        self.checked_videos = []
        self.register_buffer("_device_tracker", torch.zeros(1, dtype=torch.float32), persistent=False)

    def check_text_safety(self, prompt: str) -> bool:
        self.checked_prompts.append(prompt)
        return self.text_safe

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray | None:
        self.checked_videos.append(frames)
        return frames if self.video_safe else None

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device
