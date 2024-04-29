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


from ..utils import deprecate
from .unets.unet_2d import UNet2DModel, UNet2DOutput


class UNet2DOutput(UNet2DOutput):
    deprecation_message = "Importing `UNet2DOutput` from `diffusers.models.unet_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d import UNet2DOutput`, instead."
    deprecate("UNet2DOutput", "0.29", deprecation_message)


class UNet2DModel(UNet2DModel):
    deprecation_message = "Importing `UNet2DModel` from `diffusers.models.unet_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d import UNet2DModel`, instead."
    deprecate("UNet2DModel", "0.29", deprecation_message)
