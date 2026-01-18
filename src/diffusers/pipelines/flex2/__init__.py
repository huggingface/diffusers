from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
)

_import_structure = {"pipeline_flex2": ["Flex2Pipeline"],
                     "pipeline_output": ["Flex2PipelineOutput"]}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_flex2 import Flex2Pipeline
    from .pipeline_output import Flex2PipelineOutput

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )