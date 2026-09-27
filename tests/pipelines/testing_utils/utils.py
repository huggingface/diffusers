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
