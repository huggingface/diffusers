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


def stable_audio_projection_conversion(config):
    mapping = {}
    for old, new in (("seconds_start", "start_number_conditioner"), ("seconds_total", "end_number_conditioner")):
        mapping[f"{old}.embedder.embedding.0.weights"] = f"{new}.time_positional_embedding.0.weights"
        mapping.update(
            {f"{old}.embedder.embedding.1.{p}": f"{new}.time_positional_embedding.1.{p}" for p in ("weight", "bias")}
        )
    if config["text_encoder_dim"] != config["conditioning_dim"]:
        mapping.update({f"prompt.proj_out.{p}": f"text_projection.{p}" for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
