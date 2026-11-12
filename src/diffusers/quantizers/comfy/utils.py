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

import torch
from torch import nn


class ComfyInt8Linear(nn.Module):
    """Linear inference using stored INT8 codes and Comfy-Kitchen's dynamic activation quantization."""

    def __init__(self, linear, scale_shape, marker, compute_dtype):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype
        self.convrot = marker.get("convrot", False)
        self.convrot_groupsize = marker.get("convrot_groupsize", 256)
        self.weight = nn.Parameter(
            torch.zeros((self.out_features, self.in_features), device="meta", dtype=torch.int8), requires_grad=False
        )
        self.register_buffer("weight_scale", torch.zeros(scale_shape, device="meta", dtype=torch.float32))
        self.bias = linear.bias

    def _apply(self, fn, recurse=True):
        # Move scales with the weights without rounding them to the activation dtype.
        scale = self._buffers.pop("weight_scale")
        try:
            super()._apply(fn, recurse=recurse)
            self.register_buffer("weight_scale", scale.to(device=fn(scale).device, dtype=torch.float32))
        except Exception:
            self.register_buffer("weight_scale", scale)
            raise
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck

        return ck.int8_linear(
            inputs.to(self.compute_dtype).contiguous(),
            self.weight,
            self.weight_scale,
            self.bias,
            out_dtype=self.compute_dtype,
            convrot=self.convrot,
            convrot_groupsize=self.convrot_groupsize,
        )
