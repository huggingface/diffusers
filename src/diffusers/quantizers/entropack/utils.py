# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils import is_torch_available, logging


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


def _replace_with_entropack_linear(
    model: "ModelMixin",
    quantization_config,
    modules_to_not_convert: list[str],
) -> "ModelMixin":
    import entropack

    codec_options = dict(quantization_config.entropack_lossy_config or {})
    codec_options.update(target_bpp=quantization_config.target_bpp, execution_backend="cuda")
    codec_config = entropack.EntroPackLossyConfig(**codec_options)
    linear_cls = {
        "default": entropack.CompressedLinear,
        "fp8": entropack.CompressedFP8Linear,
        "int8": entropack.CompressedINT8Linear,
    }[quantization_config.linear_type]
    linear_kwargs = {}
    if quantization_config.linear_type == "int8":
        linear_kwargs = {
            "use_convrot": quantization_config.use_convrot,
            "convrot_groupsize": quantization_config.convrot_groupsize,
        }

    replaced = []

    def replace(module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name not in modules_to_not_convert:
                if not any((key + "." in full_name) or (key == full_name) for key in modules_to_not_convert):
                    with torch.device("meta"):
                        new_module = linear_cls(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            config=codec_config,
                            dtype=child.weight.dtype,
                            device="meta",
                            **linear_kwargs,
                        )
                        new_module.weight = nn.Parameter(
                            torch.empty(
                                child.out_features, child.in_features, dtype=child.weight.dtype, device="meta"
                            ),
                            requires_grad=False,
                        )
                        module._modules[name] = new_module
                    module._modules[name].source_cls = type(child)
                    module._modules[name].requires_grad_(False)
                    replaced.append(full_name)
            else:
                replace(child, full_name)

    replace(model, "")

    if not replaced:
        logger.warning(
            f"{model.__class__.__name__} does not appear to have any `nn.Linear` modules. Quantization will not be "
            "applied. Please check your model architecture or submit an issue on GitHub if you think this is a bug."
        )

    return model


def _dequantize_and_replace(model: "ModelMixin") -> "ModelMixin":
    import entropack

    for name, module in model.named_children():
        if isinstance(module, entropack.CompressedLinear):
            device = module.weight.device
            with torch.device("meta"):
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    dtype=module._init_dtype,
                )
            new_module.weight = nn.Parameter(
                module.dequantize(device).to(device=device, dtype=module._init_dtype), requires_grad=False
            )
            if module.bias is not None:
                new_module.bias = nn.Parameter(module.bias.data.to(device), requires_grad=False)
            model._modules[name] = new_module
        else:
            _dequantize_and_replace(module)
    return model
