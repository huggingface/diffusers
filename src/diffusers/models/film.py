# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.scale_bias = nn.Linear(in_features, out_features * 2)

    def forward(self, x, conditioning_emb):
        scale_bias = self.scale_bias(conditioning_emb)
        scale, bias = torch.chunk(scale_bias, 2, -1)
        return x * (scale + 1.0) + bias
