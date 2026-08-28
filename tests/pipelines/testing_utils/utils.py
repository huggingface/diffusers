# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import torch

from diffusers.models.attention import AttentionModuleMixin


"""
TODO (related to methods like `check_qkv_fusion_matches_attn_procs_length()`):
After https://github.com/huggingface/diffusers/pull/14113 is merged, move those
checks out of pipeline-level testing and ensure that these are sufficiently
tested in model-level tests.
"""


def check_same_shape(tensor_list):
    shapes = [tensor.shape for tensor in tensor_list]
    return all(shape == shapes[0] for shape in shapes[1:])


def check_qkv_fusion_matches_attn_procs_length(model, original_attn_processors):
    current_attn_processors = model.attn_processors
    return len(current_attn_processors) == len(original_attn_processors)


def check_qkv_fusion_processors_exist(model):
    current_attn_processors = model.attn_processors
    proc_names = [v.__class__.__name__ for _, v in current_attn_processors.items()]
    return all(p.startswith("Fused") for p in proc_names)


def check_qkv_fused_layers_exist(model, layer_names):
    is_fused_submodules = []
    for submodule in model.modules():
        if not isinstance(submodule, AttentionModuleMixin) or not submodule._supports_qkv_fusion:
            continue
        is_fused_attribute_set = submodule.fused_projections
        is_fused_layer = True
        for layer in layer_names:
            is_fused_layer = is_fused_layer and getattr(submodule, layer, None) is not None
        is_fused = is_fused_attribute_set and is_fused_layer
        is_fused_submodules.append(is_fused)
    return all(is_fused_submodules)


def cast_module_to_dtype(module, dtype):
    """Cast `module` to `dtype` in place, keeping its `_keep_in_fp32_modules` submodules in float32.

    `Module.to(dtype)` ignores the declaration: it casts every floating point tensor and only logs a warning, so a
    component that declares `_keep_in_fp32_modules` ends up feeding half-precision weights to a forward pass that
    expects float32 ones and dies on a dtype mismatch. `from_pretrained(torch_dtype=...)` is the path that honours
    the declaration, and `enable_layerwise_casting` folds it into its skip patterns.

    Each tensor is cast at most once, straight from its current dtype to its target. Casting the whole module and
    restoring the kept submodules afterwards would round-trip them through the low-precision dtype and lose the
    precision the declaration exists to preserve.

    Modules that declare nothing take the plain `.to()` path.
    """
    keep_in_fp32_modules = getattr(module, "_keep_in_fp32_modules", None)
    if not keep_in_fp32_modules:
        return module.to(dtype=dtype)
    if isinstance(keep_in_fp32_modules, str):
        # `from_pretrained` accepts a bare string as well as a list.
        keep_in_fp32_modules = [keep_in_fp32_modules]

    def target_dtype(name):
        return torch.float32 if any(part in name.split(".") for part in keep_in_fp32_modules) else dtype

    for name, param in module.named_parameters():
        if param.is_floating_point():
            param.data = param.data.to(dtype=target_dtype(name))
    for name, buffer in module.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=target_dtype(name))

    return module


def cast_pipeline_to_dtype(pipe, dtype):
    """`cast_module_to_dtype` for every `torch.nn.Module` component of `pipe`, leaving the rest untouched."""
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            cast_module_to_dtype(component, dtype)
    return pipe
