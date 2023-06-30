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

from ...utils import deprecate
from ..controlnet.pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline  # noqa: F401


deprecate(
    "stable diffusion controlnet",
    "0.22.0",
    "Importing `FlaxStableDiffusionControlNetPipeline` from diffusers.pipelines.stable_diffusion.flax_pipeline_stable_diffusion_controlnet is deprecated. Please import `from diffusers import FlaxStableDiffusionControlNetPipeline` instead.",
    standard_warn=False,
    stacklevel=3,
)
