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

from typing import TYPE_CHECKING

from ....utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {}

if is_torch_available() and is_transformers_available():
    _import_structure["pipeline_magi"] = ["MagiPipeline"]
    _import_structure["pipeline_magi_i2v"] = ["MagiImageToVideoPipeline"]
    _import_structure["pipeline_magi_v2v"] = ["MagiVideoToVideoPipeline"]
    _import_structure["pipeline_output"] = ["MagiPipelineOutput"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if is_torch_available() and is_transformers_available():
            from .pipeline_magi import MagiPipeline
            from .pipeline_magi_i2v import MagiImageToVideoPipeline
            from .pipeline_magi_v2v import MagiVideoToVideoPipeline
            from .pipeline_output import MagiPipelineOutput
    except OptionalDependencyNotAvailable:
        pass
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )