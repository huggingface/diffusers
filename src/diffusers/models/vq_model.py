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
from ..utils import deprecate
from .autoencoders.vq_model import VQEncoderOutput, VQModel


class VQEncoderOutput(VQEncoderOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `VQEncoderOutput` from `diffusers.models.vq_model` is deprecated and this will be removed in a future version. Please use `from diffusers.models.autoencoders.vq_model import VQEncoderOutput`, instead."
        deprecate("VQEncoderOutput", "0.31", deprecation_message)
        super().__init__(*args, **kwargs)


class VQModel(VQModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `VQModel` from `diffusers.models.vq_model` is deprecated and this will be removed in a future version. Please use `from diffusers.models.autoencoders.vq_model import VQModel`, instead."
        deprecate("VQModel", "0.31", deprecation_message)
        super().__init__(*args, **kwargs)
