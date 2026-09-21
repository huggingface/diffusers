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


def paella_conversion(config):
    modules = ["in_block.1", "up_blocks.0.0", "out_block.0"]
    residuals = []
    for i in range(config["levels"]):
        if i > 0:
            modules.append(f"down_blocks.{2 * i - 1}")
        residuals.append(f"down_blocks.{2 * i}")
    output = 2 * config["levels"] - 1
    mapping = {
        "vquantizer.codebook.weight": "vquantizer.embedding.weight",
        f"down_blocks.{output}.0.weight": f"down_blocks.{output}.0.weight",
    }
    modules.append(f"down_blocks.{output}.1")
    mapping.update(
        {
            f"down_blocks.{output}.1.{name}": f"down_blocks.{output}.1.{name}"
            for name in ("running_mean", "running_var", "num_batches_tracked")
        }
    )
    index = 1
    for i in range(config["levels"]):
        for _ in range(config["bottleneck_blocks"] if i == 0 else 1):
            residuals.append(f"up_blocks.{index}")
            index += 1
        if i < config["levels"] - 1:
            modules.append(f"up_blocks.{index}")
            index += 1
    for prefix in residuals:
        mapping[prefix + ".gammas"] = prefix + ".gammas"
        modules.extend(f"{prefix}.{name}" for name in ("depthwise.1", "channelwise.0", "channelwise.2"))
    mapping.update({f"{name}.{p}": f"{name}.{p}" for name in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
