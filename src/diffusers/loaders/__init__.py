from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule
from ..utils.import_utils import is_peft_available, is_torch_available, is_transformers_available


_import_structure = {}

if is_torch_available():
    _import_structure["single_file_model"] = ["FromOriginalModelMixin"]
    _import_structure["transformer_flux"] = ["FluxTransformer2DLoadersMixin"]
    _import_structure["transformer_sd3"] = ["SD3Transformer2DLoadersMixin"]
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]
    if is_transformers_available():
        _import_structure["single_file"] = ["FromSingleFileMixin"]
        _import_structure["lora_pipeline"] = [
            "AceStepLoraLoaderMixin",
            "AmusedLoraLoaderMixin",
            "AnimaLoraLoaderMixin",
            "StableDiffusionLoraLoaderMixin",
            "SD3LoraLoaderMixin",
            "AuraFlowLoraLoaderMixin",
            "StableDiffusionXLLoraLoaderMixin",
            "LTX2LoraLoaderMixin",
            "LTXVideoLoraLoaderMixin",
            "LoraLoaderMixin",
            "FluxLoraLoaderMixin",
            "CogVideoXLoraLoaderMixin",
            "CogView4LoraLoaderMixin",
            "Mochi1LoraLoaderMixin",
            "HunyuanVideoLoraLoaderMixin",
            "SanaLoraLoaderMixin",
            "Lumina2LoraLoaderMixin",
            "WanLoraLoaderMixin",
            "HeliosLoraLoaderMixin",
            "KandinskyLoraLoaderMixin",
            "HiDreamImageLoraLoaderMixin",
            "SkyReelsV2LoraLoaderMixin",
            "QwenImageLoraLoaderMixin",
            "Krea2LoraLoaderMixin",
            "ZImageLoraLoaderMixin",
            "Flux2LoraLoaderMixin",
            "Ideogram4LoraLoaderMixin",
            "ErnieImageLoraLoaderMixin",
            "CosmosLoraLoaderMixin",
            "MiniMaxH3LoraLoaderMixin",
        ]
        _import_structure["textual_inversion"] = ["TextualInversionLoaderMixin"]
        _import_structure["ip_adapter"] = [
            "IPAdapterMixin",
            "FluxIPAdapterMixin",
            "SD3IPAdapterMixin",
            "ModularIPAdapterMixin",
        ]

_import_structure["peft"] = ["PeftAdapterMixin"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .single_file_model import FromOriginalModelMixin
        from .transformer_flux import FluxTransformer2DLoadersMixin
        from .transformer_sd3 import SD3Transformer2DLoadersMixin
        from .unet import UNet2DConditionLoadersMixin
        from .utils import AttnProcsLayers

        if is_transformers_available():
            from .ip_adapter import (
                FluxIPAdapterMixin,
                IPAdapterMixin,
                ModularIPAdapterMixin,
                SD3IPAdapterMixin,
            )
            from .lora_pipeline import (
                AceStepLoraLoaderMixin,
                AmusedLoraLoaderMixin,
                AnimaLoraLoaderMixin,
                AuraFlowLoraLoaderMixin,
                CogVideoXLoraLoaderMixin,
                CogView4LoraLoaderMixin,
                CosmosLoraLoaderMixin,
                ErnieImageLoraLoaderMixin,
                Flux2LoraLoaderMixin,
                FluxLoraLoaderMixin,
                HeliosLoraLoaderMixin,
                HiDreamImageLoraLoaderMixin,
                HunyuanVideoLoraLoaderMixin,
                Ideogram4LoraLoaderMixin,
                KandinskyLoraLoaderMixin,
                Krea2LoraLoaderMixin,
                LoraLoaderMixin,
                LTX2LoraLoaderMixin,
                LTXVideoLoraLoaderMixin,
                Lumina2LoraLoaderMixin,
                MiniMaxH3LoraLoaderMixin,
                Mochi1LoraLoaderMixin,
                QwenImageLoraLoaderMixin,
                SanaLoraLoaderMixin,
                SD3LoraLoaderMixin,
                SkyReelsV2LoraLoaderMixin,
                StableDiffusionLoraLoaderMixin,
                StableDiffusionXLLoraLoaderMixin,
                WanLoraLoaderMixin,
                ZImageLoraLoaderMixin,
            )
            from .single_file import FromSingleFileMixin
            from .textual_inversion import TextualInversionLoaderMixin

    from .peft import PeftAdapterMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
