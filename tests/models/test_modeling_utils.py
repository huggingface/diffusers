# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

from diffusers.models.modeling_utils import no_init_weights


def test_no_init_weights_preserves_torch_init_return_contract():
    tensor = torch.empty(2, 3)

    with no_init_weights():
        truncated = torch.nn.init.trunc_normal_(tensor)
        zeroed = torch.nn.init.zeros_(tensor)

    assert truncated is tensor
    assert zeroed is tensor
