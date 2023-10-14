from typing import TYPE_CHECKING

from ...utils import DIFFUSERS_SLOW_IMPORT, _LazyModule


_import_structure = {
    "mel": ["Mel"],
    "pipeline_audio_diffusion": ["AudioDiffusionPipeline"],
}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
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
