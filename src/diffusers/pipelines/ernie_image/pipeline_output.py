# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional

import PIL.Image

from ...utils import BaseOutput


@dataclass
class ErnieImagePipelineOutput(BaseOutput):
    """
    Output class for ERNIE-Image pipelines.

    Args:
        images (`List[PIL.Image.Image]`):
            List of generated images.
        revised_prompts (`List[str]`, *optional*):
            List of PE-revised prompts. `None` when PE is disabled or unavailable.
    """

    images: List[PIL.Image.Image]
    revised_prompts: Optional[List[str]]
