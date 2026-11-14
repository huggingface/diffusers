# Copyright 2025 Open AI and The HuggingFace Team. All rights reserved.
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

from ...models.autoencoders.camera_shap_e import DifferentiableProjectiveCamera as _DifferentiableProjectiveCamera
from ...models.autoencoders.camera_shap_e import create_pan_cameras as _create_pan_cameras
from ...utils import deprecate


# Deprecated aliases kept for backward compatibility.
class DifferentiableProjectiveCamera(_DifferentiableProjectiveCamera):
    def __new__(cls, *args, **kwargs):
        deprecate(
            "DifferentiableProjectiveCamera",
            "1.0.0",
            "Importing `DifferentiableProjectiveCamera` from `diffusers.pipelines.shap_e.camera` is deprecated. "
            "Import it from `diffusers.models.autoencoders.camera_shap_e` instead.",
        )
        return super().__new__(cls)


def create_pan_cameras(size: int) -> _DifferentiableProjectiveCamera:
    deprecate(
        "create_pan_cameras",
        "1.0.0",
        "Importing `create_pan_cameras` from `diffusers.pipelines.shap_e.camera` is deprecated. "
        "Import it from `diffusers.models.autoencoders.camera_shap_e` instead.",
    )
    return _create_pan_cameras(size)
