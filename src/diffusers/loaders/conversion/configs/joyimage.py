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

"""Original configuration helpers and model presets for the joyimage assembly recipe."""

TRANSFORMER_CONFIG = {
    "hidden_size": 4096,
    "in_channels": 16,
    "num_attention_heads": 32,
    "num_layers": 40,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_dim_list": [16, 56, 56],
    "text_dim": 4096,
    "rope_type": "rope",
    "theta": 10000,
}

__all__ = ["TRANSFORMER_CONFIG"]
