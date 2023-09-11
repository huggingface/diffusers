from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {}
_dummy_objects = {}
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()

except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))

else:
    _import_structure["modeling_paella_vq_model"] = ["PaellaVQModel"]
    _import_structure["modeling_wuerstchen_diffnext"] = ["WuerstchenDiffNeXt"]
    _import_structure["modeling_wuerstchen_prior"] = ["WuerstchenPrior"]
    _import_structure["pipeline_wuerstchen"] = ["WuerstchenDecoderPipeline"]
    _import_structure["pipeline_wuerstchen_combined"] = ["WuerstchenCombinedPipeline"]
    _import_structure["pipeline_wuerstchen_prior"] = ["DEFAULT_STAGE_C_TIMESTEPS", "WuerstchenPriorPipeline"]


import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)
