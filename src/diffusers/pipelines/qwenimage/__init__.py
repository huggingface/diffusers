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
_import_structure = {"pipeline_output": ["QwenImagePipelineOutput", "QwenImagePriorReduxPipelineOutput"]}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["modeling_qwenimage"] = ["ReduxImageEncoder"]
    _import_structure["pipeline_qwenimage"] = ["QwenImagePipeline"]
    _import_structure["pipeline_qwenimage_controlnet"] = ["QwenImageControlNetPipeline"]
    _import_structure["pipeline_qwenimage_controlnet_inpaint"] = ["QwenImageControlNetInpaintPipeline"]
    _import_structure["pipeline_qwenimage_edit"] = ["QwenImageEditPipeline"]
    _import_structure["pipeline_qwenimage_edit_inpaint"] = ["QwenImageEditInpaintPipeline"]
    _import_structure["pipeline_qwenimage_edit_plus"] = ["QwenImageEditPlusPipeline"]
    _import_structure["pipeline_qwenimage_img2img"] = ["QwenImageImg2ImgPipeline"]
    _import_structure["pipeline_qwenimage_inpaint"] = ["QwenImageInpaintPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .pipeline_qwenimage import QwenImagePipeline
        from .pipeline_qwenimage_controlnet import QwenImageControlNetPipeline
        from .pipeline_qwenimage_controlnet_inpaint import QwenImageControlNetInpaintPipeline
        from .pipeline_qwenimage_edit import QwenImageEditPipeline
        from .pipeline_qwenimage_edit_inpaint import QwenImageEditInpaintPipeline
        from .pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        from .pipeline_qwenimage_img2img import QwenImageImg2ImgPipeline
        from .pipeline_qwenimage_inpaint import QwenImageInpaintPipeline
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
