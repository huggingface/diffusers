from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


# These modules contain pipelines from multiple libraries/frameworks
_dummy_objects = {}
_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_pt_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    _import_structure["modular_pipeline"] = [
        "ModularPipelineBlocks",
        "PipelineBlock",
        "AutoPipelineBlocks",
        "SequentialPipelineBlocks",
        "LoopSequentialPipelineBlocks",
        "ModularLoader",
        "PipelineState",
        "BlockState",
    ]
    _import_structure["modular_pipeline_utils"] = [
        "ComponentSpec",
        "ConfigSpec",
        "InputParam",
        "OutputParam",
    ]
    _import_structure["stable_diffusion_xl"] = ["StableDiffusionXLAutoPipeline", "StableDiffusionXLModularLoader"]
    _import_structure["components_manager"] = ["ComponentsManager"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *  # noqa F403
    else:
        from .modular_pipeline import (
            AutoPipelineBlocks,
            BlockState,
            LoopSequentialPipelineBlocks,
            ModularLoader,
            ModularPipelineBlocks,
            PipelineBlock,
            PipelineState,
            SequentialPipelineBlocks,
        )
        from .modular_pipeline_utils import (
            ComponentSpec,
            ConfigSpec,
            InputParam,
            OutputParam,
        )
        from .stable_diffusion_xl import (
            StableDiffusionXLAutoPipeline,
            StableDiffusionXLModularLoader,
        )
        from .components_manager import ComponentsManager
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
