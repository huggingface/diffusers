# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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
from .controlnets.controlnet_flux import FluxControlNetModel, FluxControlNetOutput, FluxMultiControlNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FluxControlNetOutput(FluxControlNetOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FluxControlNetOutput` from `diffusers.models.controlnet_flux` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_flux import FluxControlNetOutput`, instead."
        deprecate("FluxControlNetOutput", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class FluxControlNetModel(FluxControlNetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FluxControlNetModel` from `diffusers.models.controlnet_flux` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel`, instead."
        deprecate("FluxControlNetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class FluxMultiControlNetModel(FluxMultiControlNetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FluxMultiControlNetModel` from `diffusers.models.controlnet_flux` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_flux import FluxMultiControlNetModel`, instead."
        deprecate("FluxMultiControlNetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)
