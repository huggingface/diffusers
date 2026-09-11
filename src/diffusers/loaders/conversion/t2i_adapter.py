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


def t2i_adapter_conversion(config):
    modules = []
    channels, depth = config["channels"], config["num_res_blocks"]
    light = config["adapter_type"] == "light_adapter"
    if not light:
        modules.append(("conv_in", "adapter.conv_in"))
    count = len(channels) + int(light)
    for i in range(count):
        if light:
            modules.extend((f"body.{i}.{name}", f"adapter.body.{i}.{name}") for name in ("in_conv", "out_conv"))
        else:
            input_channel = channels[i - 1] if i > 0 else channels[0]
            if config["adapter_type"] == "full_adapter_xl" and i not in (1, 2):
                input_channel = channels[i]
            if input_channel != channels[i]:
                modules.append((f"body.{i * depth}.in_conv", f"adapter.body.{i}.in_conv"))
        for j in range(depth):
            old = f"body.{i}.body.{j}" if light else f"body.{i * depth + j}"
            new = f"adapter.body.{i}.resnets.{j}"
            modules.extend((f"{old}.{name}", f"{new}.{name}") for name in ("block1", "block2"))
    return Conversion(mapping={f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
