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

import numpy as np
import torch

from diffusers import DiffusionPipeline
from diffusers.models.attention import AttentionModuleMixin

from ...testing_utils import assert_tensors_close


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


def assert_outputs_close(actual, expected, atol=1e-4, rtol=0.0, msg=""):
    """
    `assert_tensors_close` for pipeline outputs, which are usually numpy arrays (`output_type="np"`). Mirrors the
    model-level assertion style (concise diff messages) while accepting numpy/torch outputs.
    """
    assert_tensors_close(
        torch.as_tensor(to_np(actual)), torch.as_tensor(to_np(expected)), atol=atol, rtol=rtol, msg=msg
    )


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


# Some models (e.g. unCLIP) are extremely likely to significantly deviate depending on which hardware is used.
# This helper function is used to check that the image doesn't deviate on average more than 10 pixels from a
# reference image.
def assert_mean_pixel_difference(image, expected_image, expected_max_diff=10):
    image = np.asarray(DiffusionPipeline.numpy_to_pil(image)[0], dtype=np.float32)
    expected_image = np.asarray(DiffusionPipeline.numpy_to_pil(expected_image)[0], dtype=np.float32)
    avg_diff = np.abs(image - expected_image).mean()
    assert avg_diff < expected_max_diff, f"Error image deviates {avg_diff} pixels on average"
