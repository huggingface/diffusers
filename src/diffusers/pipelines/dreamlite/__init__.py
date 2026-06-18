# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
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

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
)


_import_structure = {
    "pipeline_dreamlite": ["DreamLitePipeline"],
    "pipeline_dreamlite_mobile": ["DreamLiteMobilePipeline"],
    "pipeline_output": ["DreamLitePipelineOutput"],
}


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_dreamlite import DreamLitePipeline
    from .pipeline_dreamlite_mobile import DreamLiteMobilePipeline
    from .pipeline_output import DreamLitePipelineOutput
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
