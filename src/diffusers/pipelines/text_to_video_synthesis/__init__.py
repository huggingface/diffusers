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
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))

else:
    _import_structure["pipeline_output"] = ["TextToVideoSDPipelineOutput"]
    _import_structure["pipeline_text_to_video_synth"] = ["TextToVideoSDPipeline"]
    _import_structure["pipeline_text_to_video_synth_img2img"] = ["VideoToVideoSDPipeline"]
    _import_structure["pipeline_text_to_video_zero"] = ["TextToVideoZeroPipeline"]

import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)
