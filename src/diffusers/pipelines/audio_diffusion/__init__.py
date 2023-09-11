from typing import TYPE_CHECKING

from ...utils import _LazyModule


_import_structure = {}
_import_structure["mel"] = ["Mel"]
_import_structure["pipeline_audio_diffusion"] = ["AudioDiffusionPipeline"]

if TYPE_CHECKING:
    from .mel import Mel
    from .pipeline_audio_diffusion import AudioDiffusionPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
