# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from .scheduling_token_diffusion import TokenDiffusionScheduler


# Backwards-compatible alias. `TokenDiffusionScheduler` now natively supports
# the `block_mask` parameter in `add_noise()` and `step()`, so a separate
# subclass is no longer needed.
BlockTokenDiffusionScheduler = TokenDiffusionScheduler

__all__ = ["BlockTokenDiffusionScheduler"]
