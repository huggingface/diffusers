from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


_import_structure = {}
_dummy_objects = {}


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))

else:
    _import_structure["modeling_audioldm2"] = ["AudioLDM2ProjectionModel", "AudioLDM2UNet2DConditionModel"]
    _import_structure["pipeline_audioldm2"] = ["AudioLDM2Pipeline"]

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
