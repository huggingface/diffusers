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


def flux_ip_adapter_conversion(config):
    mapping = {
        f"ip_adapter_proj_model.{name}.{p}": f"image_proj.{name}.{p}"
        for name in ("norm", "proj")
        for p in ("weight", "bias")
    }
    for i in range(config["num_layers"]):
        for part in ("k", "v"):
            for p in ("weight", "bias"):
                mapping[f"double_blocks.{i}.processor.ip_adapter_double_stream_{part}_proj.{p}"] = (
                    f"ip_adapter.{i}.to_{part}_ip.{p}"
                )
    return Conversion(mapping=mapping)
