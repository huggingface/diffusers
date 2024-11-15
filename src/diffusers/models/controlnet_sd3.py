# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from ..utils import deprecate, logging
from .controlnets.controlnet_sd3 import SD3ControlNetModel, SD3ControlNetOutput, SD3MultiControlNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SD3ControlNetOutput(SD3ControlNetOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SD3ControlNetOutput` from `diffusers.models.controlnet_sd3` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetOutput`, instead."
        deprecate("SD3ControlNetOutput", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class SD3ControlNetModel(SD3ControlNetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SD3ControlNetModel` from `diffusers.models.controlnet_sd3` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel`, instead."
        deprecate("SD3ControlNetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class SD3MultiControlNetModel(SD3MultiControlNetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SD3MultiControlNetModel` from `diffusers.models.controlnet_sd3` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sd3 import SD3MultiControlNetModel`, instead."
        deprecate("SD3MultiControlNetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)
