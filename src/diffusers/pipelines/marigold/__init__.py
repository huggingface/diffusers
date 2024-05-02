from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
)


_import_structure = {
    "pipeline_marigold_depth": ["MarigoldDepthOutput", "MarigoldDepthPipeline"],
    "pipeline_marigold_normals": ["MarigoldNormalsOutput", "MarigoldNormalsPipeline"],
}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_marigold_depth import MarigoldDepthOutput, MarigoldDepthPipeline
    from .pipeline_marigold_normals import MarigoldNormalsOutput, MarigoldNormalsPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
