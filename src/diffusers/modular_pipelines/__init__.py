from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
    logging,
)


logger = logging.get_logger(__name__)
logger.warning(
    "Modular Diffusers is currently an experimental feature under active development. The API is subject to breaking changes in future releases."
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
        "ModularPipeline",
        "AutoPipelineBlocks",
        "SequentialPipelineBlocks",
        "LoopSequentialPipelineBlocks",
        "PipelineState",
        "BlockState",
    ]
    _import_structure["modular_pipeline_utils"] = [
        "ComponentSpec",
        "ConfigSpec",
        "InputParam",
        "OutputParam",
        "InsertableDict",
    ]
    _import_structure["stable_diffusion_xl"] = ["StableDiffusionXLAutoBlocks", "StableDiffusionXLModularPipeline"]
    _import_structure["wan"] = ["WanAutoBlocks", "Wan22AutoBlocks", "WanModularPipeline"]
    _import_structure["flux"] = [
        "FluxAutoBlocks",
        "FluxModularPipeline",
        "FluxKontextAutoBlocks",
        "FluxKontextModularPipeline",
    ]
    _import_structure["qwenimage"] = [
        "QwenImageAutoBlocks",
        "QwenImageModularPipeline",
        "QwenImageEditModularPipeline",
        "QwenImageEditAutoBlocks",
        "QwenImageEditPlusModularPipeline",
        "QwenImageEditPlusAutoBlocks",
    ]
    _import_structure["components_manager"] = ["ComponentsManager"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *  # noqa F403
    else:
        from .components_manager import ComponentsManager
        from .flux import FluxAutoBlocks, FluxKontextAutoBlocks, FluxKontextModularPipeline, FluxModularPipeline
        from .modular_pipeline import (
            AutoPipelineBlocks,
            BlockState,
            LoopSequentialPipelineBlocks,
            ModularPipeline,
            ModularPipelineBlocks,
            PipelineState,
            SequentialPipelineBlocks,
        )
        from .modular_pipeline_utils import ComponentSpec, ConfigSpec, InputParam, InsertableDict, OutputParam
        from .qwenimage import (
            QwenImageAutoBlocks,
            QwenImageEditAutoBlocks,
            QwenImageEditModularPipeline,
            QwenImageEditPlusAutoBlocks,
            QwenImageEditPlusModularPipeline,
            QwenImageModularPipeline,
        )
        from .stable_diffusion_xl import StableDiffusionXLAutoBlocks, StableDiffusionXLModularPipeline
        from .wan import Wan22AutoBlocks, WanAutoBlocks, WanModularPipeline
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
