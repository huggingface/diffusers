import torch.nn as nn

from ...utils import is_accelerate_available, is_nunchaku_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights


logger = logging.get_logger(__name__)


def _replace_with_nunchaku_linear(
    model,
    svdq_linear_cls,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = svdq_linear_cls(
                        in_features,
                        out_features,
                        rank=quantization_config.rank,
                        bias=module.bias is not None,
                        torch_dtype=module.weight.dtype,
                    )
                    has_been_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_nunchaku_linear(
                module,
                svdq_linear_cls,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_nunchaku_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    if is_nunchaku_available():
        from nunchaku.models.linear import SVDQW4A4Linear

    model, _ = _replace_with_nunchaku_linear(
        model, SVDQW4A4Linear, modules_to_not_convert, current_key_name, quantization_config
    )

    has_been_replaced = any(
        isinstance(replaced_module, SVDQW4A4Linear) for _, replaced_module in model.named_modules()
    )
    if not has_been_replaced:
        logger.warning(
            "You are loading your model in the SVDQuant method but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
