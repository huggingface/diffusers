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

from scripts.convert_rae_stage2_to_diffusers import unwrap_state_dict


def test_unwrap_state_dict_strips_supported_prefixes():
    tensor = torch.randn(1)

    assert unwrap_state_dict({"model.module.blocks.0.weight": tensor}) == {"blocks.0.weight": tensor}
    assert unwrap_state_dict({"model.blocks.0.weight": tensor}) == {"blocks.0.weight": tensor}
    assert unwrap_state_dict({"module.blocks.0.weight": tensor}) == {"blocks.0.weight": tensor}
