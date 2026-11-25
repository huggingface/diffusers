# Copyright 2026 Pruna AI and The HuggingFace Team. All rights reserved.
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

from ...utils import deprecate
from ..p_image_oss.modular_blocks_p_image_oss import PImageOSSBlocks as _PImageOSSBlocks


# Deprecated aliases kept for backward compatibility. The warning is raised from `__new__` rather than `__init__` so
# that the parent's `__init__` signature, which `ConfigMixin` inspects when loading a config, stays intact.
class PH3ImageBlocks(_PImageOSSBlocks):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "PH3ImageBlocks",
            "0.45.0",
            "`PH3ImageBlocks` was renamed to `PImageOSSBlocks`. "
            "Import it with `from diffusers import PImageOSSBlocks` instead.",
        )
        return super().__new__(cls)
