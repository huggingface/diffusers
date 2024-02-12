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
from .unets.unet_1d import UNet1DModel, UNet1DOutput


class UNet1DOutput(UNet1DOutput):
    deprecation_message = "Importing `UNet1DOutput` from `diffusers.models.unet_1d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d import UNet1DOutput`, instead."
    deprecate("UNet1DOutput", "0.29", deprecation_message)


class UNet1DModel(UNet1DModel):
    deprecation_message = "Importing `UNet1DModel` from `diffusers.models.unet_1d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d import UNet1DModel`, instead."
    deprecate("UNet1DModel", "0.29", deprecation_message)
