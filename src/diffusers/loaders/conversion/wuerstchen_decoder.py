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
from .wuerstchen_prior import _wuerstchen_block_rules


def wuerstchen_decoder_conversion(config):
    modules = ["clip_mapper", "embedding.1", "clf.1"]
    injections = list(config["inject_effnet"]) + list(reversed(config["inject_effnet"]))
    modules.extend(f"effnet_mappers.{i}" for i, active in enumerate(injections) if active)
    blocks = []
    count = len(config["c_hidden"])
    for direction in ("down", "up"):
        for i in range(count):
            level = i if direction == "down" else count - 1 - i
            offset = int(direction == "down" and level > 0)
            kinds = config["level_config"][level]
            for j in range(config["blocks"][level]):
                blocks.extend(
                    (f"{direction}_blocks.{i}.{offset + j * len(kinds) + k}", kind, config["c_hidden"][level])
                    for k, kind in enumerate(kinds)
                )
            if direction == "down" and level > 0:
                modules.append(f"down_blocks.{i}.0.1")
            elif direction == "up" and level > 0:
                modules.append(f"up_blocks.{i}.{config['blocks'][level] * len(kinds)}.1")
    mapping, rules = _wuerstchen_block_rules(blocks)
    mapping.update({f"{name}.{p}": f"{name}.{p}" for name in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)
