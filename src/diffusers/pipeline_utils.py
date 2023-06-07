# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# NOTE: This file is deprecated and will be removed in a future version.
# It only exists so that temporarely `from diffusers.pipelines import DiffusionPipeline` works

from .pipelines import DiffusionPipeline, ImagePipelineOutput  # noqa: F401
from .utils import deprecate


deprecate(
    "pipelines_utils",
    "0.22.0",
    "Importing `DiffusionPipeline` or `ImagePipelineOutput` from diffusers.pipeline_utils is deprecated. Please import from diffusers.pipelines.pipeline_utils instead.",
    standard_warn=False,
    stacklevel=3,
)
