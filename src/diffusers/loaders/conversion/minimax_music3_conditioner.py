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


from .core import Conversion


def minimax_music3_conditioner_conversion(config):
    return Conversion(
        mapping={
            "cond_layer_logits": "layer_weight_logits",
            "cond_layer_scale": "layer_scale",
            "latent_conditioners.0.weight": "proj.weight",
            "latent_conditioners.0.bias": "proj.bias",
        }
    )
