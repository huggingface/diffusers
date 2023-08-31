from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from PIL import Image

from ...utils import (
    _LazyModule,
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    get_objects_from_module,
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
    _import_structure["image_encoder"] = ["PaintByExampleImageEncoder"]
    _import_structure["pipeline_paint_by_example"] = ["PaintByExamplePipeline"]

import sys

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)

for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
