import torch.nn as nn

from ...utils import is_accelerate_available, logging


logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate import init_empty_weights


def _replace_with_quanto_layers(model, quantization_config, modules_to_not_convert: list, pre_quantized=False):
    # Quanto imports diffusers internally. These are placed here to avoid circular imports
    from optimum.quanto import QLinear, freeze, qfloat8, qint2, qint4, qint8

    def _get_weight_type(dtype: str):
        return {"float8": qfloat8, "int8": qint8, "int4": qint4, "int2": qint2}[dtype]

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
                    qlinear = QLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        dtype=module.weight.dtype,
                        weights=_get_weight_type(quantization_config.weights_dtype),
                    )
                    model._modules[name] = qlinear
                    model._modules[name].source_cls = type(module)
                    model._modules[name].requires_grad_(False)

        return model

    model = _replace_layers(model, quantization_config, modules_to_not_convert)
    has_been_replaced = any(isinstance(replaced_module, QLinear) for _, replaced_module in model.named_modules())

    if not has_been_replaced:
        logger.warning(
            f"{model.__class__.__name__} does not appear to have any `nn.Linear` modules. Quantization will not be applied."
            " Please check your model architecture, or submit an issue on Github if you think this is a bug."
            " https://github.com/huggingface/diffusers/issues/new"
        )

    # We need to freeze the pre_quantized model in order for the loaded state_dict and model state dict
    # to match when trying to load weights with load_model_dict_into_meta
    if pre_quantized:
        freeze(model)

    return model
