# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from
https://github.com/huggingface/transformers/blob/c409cd81777fb27aadc043ed3d8339dbc020fb3b/src/transformers/integrations/bitsandbytes.py
"""

import inspect
from inspect import signature
from typing import Union

from ...utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
from ..quantization_config import QuantizationMethod


if is_torch_available():
    import torch
    import torch.nn as nn

if is_bitsandbytes_available():
    import bitsandbytes as bnb

if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module

logger = logging.get_logger(__name__)


def _replace_with_bnb_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
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

                    if quantization_config.quantization_method() == "llm_int8":
                        model._modules[name] = bnb.nn.Linear8bitLt(
                            in_features,
                            out_features,
                            module.bias is not None,
                            has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                        )
                        has_been_replaced = True
                    else:
                        if (
                            quantization_config.llm_int8_skip_modules is not None
                            and name in quantization_config.llm_int8_skip_modules
                        ):
                            pass
                        else:
                            extra_kwargs = (
                                {"quant_storage": quantization_config.bnb_4bit_quant_storage}
                                if "quant_storage" in list(signature(bnb.nn.Linear4bit).parameters)
                                else {}
                            )
                            model._modules[name] = bnb.nn.Linear4bit(
                                in_features,
                                out_features,
                                module.bias is not None,
                                quantization_config.bnb_4bit_compute_dtype,
                                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                                quant_type=quantization_config.bnb_4bit_quant_type,
                                **extra_kwargs,
                            )
                            has_been_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    """
    Helper function to replace the `nn.Linear` layers within `model` with either `bnb.nn.Linear8bit` or
    `bnb.nn.Linear4bit` using the `bitsandbytes` library.

    References:
        * `bnb.nn.Linear8bit`: [LLM.int8(): 8-bit Matrix Multiplication for Transformers at
          Scale](https://arxiv.org/abs/2208.07339)
        * `bnb.nn.Linear4bit`: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `[]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `modules_to_not_convert` in
            full precision for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
        quantization_config ('transformers.utils.quantization_config.BitsAndBytesConfig'):
            To configure and manage settings related to quantization, a technique used to compress neural network
            models by reducing the precision of the weights and activations, thus making models more efficient in terms
            of both storage and computation.
    """
    model, has_been_replaced = _replace_with_bnb_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


# Copied from PEFT: https://github.com/huggingface/peft/blob/47b3712898539569c02ec5b3ed4a6c36811331a1/src/peft/utils/integrations.py#L41
def dequantize_bnb_weight(weight: "torch.nn.Parameter", state=None):
    """
    Helper function to dequantize 4bit or 8bit bnb weights.

    If the weight is not a bnb quantized weight, it will be returned as is.
    """
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    if cls_name == "Params4bit":
        output_tensor = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        logger.warning_once(
            f"The model is going to be dequantized in {output_tensor.dtype} - if you want to upcast it to another dtype, make sure to pass the desired dtype when quantizing the model through `bnb_4bit_quant_type` argument of `BitsAndBytesConfig`"
        )
        return output_tensor

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    return bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()


def _create_accelerate_new_hook(old_hook):
    r"""
    Creates a new hook based on the old hook. Use it only if you know what you are doing ! This method is a copy of:
    https://github.com/huggingface/peft/blob/748f7968f3a31ec06a1c2b0328993319ad9a150a/src/peft/utils/other.py#L245 with
    some changes
    """
    old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
    old_hook_attr = old_hook.__dict__
    filtered_old_hook_attr = {}
    old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
    for k in old_hook_attr.keys():
        if k in old_hook_init_signature.parameters:
            filtered_old_hook_attr[k] = old_hook_attr[k]
    new_hook = old_hook_cls(**filtered_old_hook_attr)
    return new_hook


def _dequantize_and_replace(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Converts a quantized model into its dequantized original version. The newly converted model will have some
    performance drop compared to the original model before quantization - use it only for specific usecases such as
    QLoRA adapters merging.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    quant_method = quantization_config.quantization_method()

    target_cls = bnb.nn.Linear8bitLt if quant_method == "llm_int8" else bnb.nn.Linear4bit

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, target_cls) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)

            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                bias = getattr(module, "bias", None)

                device = module.weight.device
                with init_empty_weights():
                    new_module = torch.nn.Linear(module.in_features, module.out_features, bias=bias is not None)

                if quant_method == "llm_int8":
                    state = module.state
                else:
                    state = None

                new_module.weight = torch.nn.Parameter(dequantize_bnb_weight(module.weight, state))

                if bias is not None:
                    new_module.bias = bias

                # Create a new hook and attach it in case we use accelerate
                if hasattr(module, "_hf_hook"):
                    old_hook = module._hf_hook
                    new_hook = _create_accelerate_new_hook(old_hook)

                    remove_hook_from_module(module)
                    add_hook_to_module(new_module, new_hook)

                new_module.to(device)
                model._modules[name] = new_module
                has_been_replaced = True
        if len(list(module.children())) > 0:
            _, has_been_replaced = _dequantize_and_replace(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def dequantize_and_replace(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
):
    model, has_been_replaced = _dequantize_and_replace(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    if not has_been_replaced:
        logger.warning(
            "For some reason the model has not been properly dequantized. You might see unexpected behavior."
        )

    return model


def _check_bnb_status(module) -> Union[bool, bool]:
    is_loaded_in_4bit_bnb = (
        hasattr(module, "is_loaded_in_4bit")
        and module.is_loaded_in_4bit
        and getattr(module, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
    )
    is_loaded_in_8bit_bnb = (
        hasattr(module, "is_loaded_in_8bit")
        and module.is_loaded_in_8bit
        and getattr(module, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
    )
    return is_loaded_in_4bit_bnb or is_loaded_in_8bit_bnb, is_loaded_in_4bit_bnb, is_loaded_in_8bit_bnb
