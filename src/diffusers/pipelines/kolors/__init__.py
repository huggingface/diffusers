from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_sentencepiece_available,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()) and is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_and_sentencepiece_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_sentencepiece_objects))
else:
    _import_structure["pipeline_kolors"] = ["KolorsPipeline"]
    _import_structure["pipeline_kolors_img2img"] = ["KolorsImg2ImgPipeline"]
    _import_structure["text_encoder"] = ["ChatGLMModel"]
    _import_structure["tokenizer"] = ["ChatGLMTokenizer"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()) and is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_and_sentencepiece_objects import *

    else:
        from .pipeline_kolors import KolorsPipeline
        from .pipeline_kolors_img2img import KolorsImg2ImgPipeline
        from .text_encoder import ChatGLMModel
        from .tokenizer import ChatGLMTokenizer

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
