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

"""Original configuration helpers and model presets for the blip_diffusion assembly recipe."""

BLIP2_CONFIG = {
    "vision_config": {
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "intermediate_size": 4096,
        "hidden_act": "quick_gelu",
    },
    "qformer_config": {
        "cross_attention_frequency": 1,
        "encoder_hidden_size": 1024,
        "vocab_size": 30523,
    },
    "num_query_tokens": 16,
}

__all__ = ["BLIP2_CONFIG"]
