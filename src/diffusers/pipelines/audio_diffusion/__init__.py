from ...utils import _LazyModule


_import_structure = {}
_dummy_objects = {}

_import_structure["mel"] = ["Mel"]
_import_structure["pipeline_audio_diffusion"] = ["AudioDiffusionPipeline"]

import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)
