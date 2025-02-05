from typing import Optional

import torch.nn as nn

from ...utils import is_accelerate_available


if is_accelerate_available():
    from accelerate import init_empty_weights


def _replace_with_quanto_layers(model, quantization_config, modules_to_not_convert: list):
    # Quanto imports diffusers internally. These are placed here to avoid circular imports
    from optimum.quanto import QLayerNorm, QLinear, qfloat8, qint2, qint4, qint8

    def _get_weight_type(dtype: str):
        return {"float8": qfloat8, "int8": qint8, "int4": qint4, "int2": qint2}[dtype]

    def _get_activation_type(dtype: Optional[str]):
        return {None: None, "float8": qfloat8, "int8": qint8}[dtype]

    def _replace_layers(model, quantization_config, modules_to_not_convert):
        has_children = list(model.children())
        if not has_children:
            return model

        for name, module in model.named_children():
            _replace_layers(module, quantization_config, modules_to_not_convert)

            if name in modules_to_not_convert:
                continue

            if isinstance(module, nn.Linear):
                with init_empty_weights():
                    model._modules[name] = QLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        dtype=module.weight.dtype,
                        weights=_get_weight_type(quantization_config.weights),
                        activations=_get_activation_type(quantization_config.activations),
                    )
                    model._modules[name].source_cls = type(module)
                    model._modules[name].requires_grad_(False)

        return model

    model = _replace_layers(model, quantization_config, modules_to_not_convert)

    return model
