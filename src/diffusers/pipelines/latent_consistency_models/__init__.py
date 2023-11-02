from typing import TYPE_CHECKING

from ...utils import (
    _LazyModule,
)


_import_structure = {
    "pipeline_latent_consistency_text2img": ["LatentConsistencyModelPipeline"],
    "pipeline_latent_consistency_img2img": ["LatentConsistencyModelImg2ImgPipeline"],
}


if TYPE_CHECKING:
    from .pipeline_latent_consistency_text2img import LatentConsistencyModelPipeline
    from .pipeline_latent_consistency_img2img import LatentConsistencyModelImg2ImgPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
