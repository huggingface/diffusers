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
from .single_file.single_file_model import (
    SINGLE_FILE_LOADABLE_CLASSES,  # noqa: F401
    FromOriginalModelMixin,
)


class FromOriginalModelMixin(FromOriginalModelMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FromOriginalModelMixin` from diffusers.loaders.single_file_model has been deprecated. Please use `from diffusers.loaders.single_file.single_file_model import FromOriginalModelMixin` instead."
        deprecate("diffusers.loaders.single_file_model.FromOriginalModelMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)
