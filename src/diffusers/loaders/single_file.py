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
from .single_file.single_file import FromSingleFileMixin


def load_single_file_sub_model(
    library_name,
    class_name,
    name,
    checkpoint,
    pipelines,
    is_pipeline_module,
    cached_model_config_path,
    original_config=None,
    local_files_only=False,
    torch_dtype=None,
    is_legacy_loading=False,
    disable_mmap=False,
    **kwargs,
):
    from .single_file.single_file import load_single_file_sub_model

    deprecation_message = "Importing `load_single_file_sub_model()` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file import load_single_file_sub_model` instead."
    deprecate("diffusers.loaders.single_file.load_single_file_sub_model", "0.36", deprecation_message)

    return load_single_file_sub_model(
        library_name,
        class_name,
        name,
        checkpoint,
        pipelines,
        is_pipeline_module,
        cached_model_config_path,
        original_config,
        local_files_only,
        torch_dtype,
        is_legacy_loading,
        disable_mmap,
        **kwargs,
    )


class FromSingleFileMixin(FromSingleFileMixin):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `FromSingleFileMixin` from diffusers.loaders.single_file has been deprecated. Please use `from diffusers.loaders.single_file.single_file import FromSingleFileMixin` instead."
        deprecate("diffusers.loaders.single_file.FromSingleFileMixin", "0.36", deprecation_message)
        super().__init__(*args, **kwargs)
