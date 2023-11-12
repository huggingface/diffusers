from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule
from ..utils.import_utils import is_torch_available, is_transformers_available


_import_structure = {}

if is_torch_available():
    _import_structure["single_file"] = ["FromOriginalControlnetMixin", "FromSingleFileMixinFromOriginalVAEMixin"]
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    if is_transformers_available():
        _import_structure["single_file"].extend(["FromSingleFileMixin"])
        _import_structure["lora"] = ["LoraLoaderMixin", "StableDiffusionXLLoraLoaderMixin"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .single_file import FromOriginalControlnetMixin, FromSingleFileMixinFromOriginalVAEMixin
        from .unet import UNet2DConditionLoadersMixin

    if is_transformers_available():
        from .lora import LoraLoaderMixin, StableDiffusionXLLoraLoaderMixin
        from .single_file import FromSingleFileMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
