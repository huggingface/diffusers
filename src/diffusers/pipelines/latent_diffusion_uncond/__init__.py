from typing import TYPE_CHECKING

from ...utils import DIFFUSERS_SLOW_IMPORT, _LazyModule


_import_structure = {"pipeline_latent_diffusion_uncond": ["LDMPipeline"]}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_latent_diffusion_uncond import LDMPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
