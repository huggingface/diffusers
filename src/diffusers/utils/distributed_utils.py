# Copyright 2025 The HuggingFace Inc. team.
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


try:
    import torch
except ImportError:
    torch = None


def is_torch_dist_rank_zero() -> bool:
    if torch is None:
        return True

    dist_module = getattr(torch, "distributed", None)
    if dist_module is None or not dist_module.is_available():
        return True

    if not dist_module.is_initialized():
        return True

    try:
        return dist_module.get_rank() == 0
    except (RuntimeError, ValueError):
        return True
