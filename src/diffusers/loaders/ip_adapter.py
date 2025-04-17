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
from .ip_adapter import FluxIPAdapterMixin, IPAdapterMixin, SD3IPAdapterMixin


class IPAdapterMixin(IPAdapterMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `IPAdapterMixin` from diffusers.loaders.ip_adapter has been deprecated. Please use `from diffusers.loaders.ip_adapter.ip_adapter import IPAdapterMixin` instead."
        deprecate("diffusers.loaders.ip_adapter.IPAdapterMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class FluxIPAdapterMixin(FluxIPAdapterMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FluxIPAdapterMixin` from diffusers.loaders.ip_adapter has been deprecated. Please use `from diffusers.loaders.ip_adapter.ip_adapter import FluxIPAdapterMixin` instead."
        deprecate("diffusers.loaders.ip_adapter.FluxIPAdapterMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)


class SD3IPAdapterMixin(SD3IPAdapterMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `SD3IPAdapterMixin` from diffusers.loaders.ip_adapter has been deprecated. Please use `from diffusers.loaders.ip_adapter.ip_adapter import SD3IPAdapterMixin` instead."
        deprecate("diffusers.loaders.ip_adapter.SD3IPAdapterMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)
