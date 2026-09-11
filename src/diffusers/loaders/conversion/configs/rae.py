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

"""Original configuration helpers and model presets for the rae assembly recipe."""

DECODER_CONFIGS = {
    "ViTB": {
        "decoder_hidden_size": 768,
        "decoder_intermediate_size": 3072,
        "decoder_num_attention_heads": 12,
        "decoder_num_hidden_layers": 12,
    },
    "ViTL": {
        "decoder_hidden_size": 1024,
        "decoder_intermediate_size": 4096,
        "decoder_num_attention_heads": 16,
        "decoder_num_hidden_layers": 24,
    },
    "ViTXL": {
        "decoder_hidden_size": 1152,
        "decoder_intermediate_size": 4096,
        "decoder_num_attention_heads": 16,
        "decoder_num_hidden_layers": 28,
    },
}

__all__ = ["DECODER_CONFIGS"]
