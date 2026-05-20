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

from ...models.others.watermark_if import IFWatermarker as _IFWatermarker
from ...utils import deprecate


class IFWatermarker(_IFWatermarker):
    def __init__(self, *args, **kwargs):
        deprecate(
            "IFWatermarker",
            "1.0.0",
            "Importing `IFWatermarker` from `diffusers.pipelines.deepfloyd_if.watermark` is deprecated. "
            "Import it from `diffusers.models.others` instead.",
        )
        super().__init__(*args, **kwargs)
