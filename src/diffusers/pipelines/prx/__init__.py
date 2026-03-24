from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["PRXPipelineOutput"]}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_prx"] = ["PRXPipeline"]

# Wrap T5GemmaEncoder to pass config.encoder (T5GemmaModuleConfig) instead of the
# composite T5GemmaConfig, which lacks flat attributes expected by T5GemmaEncoder.__init__.
try:
    if is_transformers_available():
        import transformers
        from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder as _T5GemmaEncoder

        class T5GemmaEncoder(_T5GemmaEncoder):
            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                if "config" not in kwargs:
                    from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig

                    config = T5GemmaConfig.from_pretrained(pretrained_model_name_or_path)
                    if hasattr(config, "encoder"):
                        kwargs["config"] = config.encoder
                return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        _additional_imports["T5GemmaEncoder"] = T5GemmaEncoder
        if not hasattr(transformers, "T5GemmaEncoder"):
            transformers.T5GemmaEncoder = T5GemmaEncoder
except ImportError:
    pass

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .pipeline_output import PRXPipelineOutput
        from .pipeline_prx import PRXPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
