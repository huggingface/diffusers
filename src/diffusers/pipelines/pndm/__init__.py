from ...utils import _LazyModule


_import_structure = {}
_import_structure["pipeline_pndm"] = ["PNDMPipeline"]


import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)
