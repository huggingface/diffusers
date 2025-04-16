from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule, deprecate
from ..utils.import_utils import is_peft_available, is_torch_available, is_transformers_available


def text_encoder_lora_state_dict(text_encoder):
    deprecate(
        "text_encoder_load_state_dict in `models`",
        "0.27.0",
        "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",
    )
    state_dict = {}

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


if is_transformers_available():

    def text_encoder_attn_modules(text_encoder):
        deprecate(
            "text_encoder_attn_modules in `models`",
            "0.27.0",
            "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",
        )
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))
        else:
            raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

        return attn_modules


_import_structure = {}

if is_torch_available():
    _import_structure["ip_adapter.transformer_flux"] = ["FluxTransformer2DLoadersMixin"]
    _import_structure["ip_adapter.transformer_sd3"] = ["SD3Transformer2DLoadersMixin"]
    _import_structure["single_file.single_file_model"] = ["FromOriginalModelMixin"]
    _import_structure["unet.unet"] = ["UNet2DConditionLoadersMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]
    if is_transformers_available():
        _import_structure["single_file.single_file"] = ["FromSingleFileMixin"]
        _import_structure["lora.lora_pipeline"] = [
            "AmusedLoraLoaderMixin",
            "StableDiffusionLoraLoaderMixin",
            "SD3LoraLoaderMixin",
            "AuraFlowLoraLoaderMixin",
            "StableDiffusionXLLoraLoaderMixin",
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
            "LoraBaseMixin",
        ]
        _import_structure["textual_inversion"] = ["TextualInversionLoaderMixin"]
        _import_structure["ip_adapter.ip_adapter"] = [
            "IPAdapterMixin",
            "FluxIPAdapterMixin",
            "SD3IPAdapterMixin",
        ]

_import_structure["peft"] = ["PeftAdapterMixin"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .ip_adapter import FluxTransformer2DLoadersMixin, SD3Transformer2DLoadersMixin
        from .single_file import FromOriginalModelMixin
        from .unet import UNet2DConditionLoadersMixin
        from .utils import AttnProcsLayers

        if is_transformers_available():
            from .ip_adapter import FluxIPAdapterMixin, IPAdapterMixin, SD3IPAdapterMixin
            from .lora import (
                AmusedLoraLoaderMixin,
                AuraFlowLoraLoaderMixin,
                CogVideoXLoraLoaderMixin,
                CogView4LoraLoaderMixin,
                FluxLoraLoaderMixin,
                HunyuanVideoLoraLoaderMixin,
                LoraBaseMixin,
                LoraLoaderMixin,
                LTXVideoLoraLoaderMixin,
                Lumina2LoraLoaderMixin,
                Mochi1LoraLoaderMixin,
                SanaLoraLoaderMixin,
                SD3LoraLoaderMixin,
                StableDiffusionLoraLoaderMixin,
                StableDiffusionXLLoraLoaderMixin,
                WanLoraLoaderMixin,
            )
            from .single_file import FromSingleFileMixin
            from .textual_inversion import TextualInversionLoaderMixin

    from .peft import PeftAdapterMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
