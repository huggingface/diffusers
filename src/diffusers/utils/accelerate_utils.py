# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""
Accelerate utilities: Utilities related to accelerate
"""

from packaging import version

from .import_utils import is_accelerate_available


if is_accelerate_available():
    import accelerate


def apply_forward_hook(method):
    """
    Decorator that applies a registered CpuOffload hook to an arbitrary function rather than `forward`. This is useful
    for cases where a PyTorch module provides functions other than `forward` that should trigger a move to the
    appropriate acceleration device. This is the case for `encode` and `decode` in [`AutoencoderKL`].

    This decorator looks inside the internal `_hf_hook` property to find a registered offload hook.

    :param method: The method to decorate. This method should be a method of a PyTorch module.
    """
    accelerate_version = version.parse(accelerate.__version__).base_version
    if version.parse(accelerate_version) < version.parse("0.17.0"):
        return method

    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_hf_hook") and hasattr(self._hf_hook, "pre_forward"):
            self._hf_hook.pre_forward(self)
        return method(self, *args, **kwargs)

    return wrapper
