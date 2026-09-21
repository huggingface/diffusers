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

from .core import Conversion, Rule
from .sana import sana_conversion


def sana_controlnet_conversion(config):
    base = sana_conversion(config)

    def control_key(key):
        if not key.startswith("blocks."):
            return key
        _, index, name = key.split(".", 2)
        return f"controlnet.{index}.copied_block.{name}"

    mapping = {control_key(old): new for old, new in base.mapping.items() if not old.startswith("final_layer.")}
    rules = tuple(
        Rule(tuple(control_key(key) for key in rule.original), rule.diffusers, rule.transform) for rule in base.rules
    )
    for p in ("weight", "bias"):
        mapping[f"controlnet.0.before_proj.{p}"] = f"input_block.{p}"
        mapping.update(
            {f"controlnet.{i}.after_proj.{p}": f"controlnet_blocks.{i}.{p}" for i in range(config["num_layers"])}
        )
    return Conversion(mapping=mapping, rules=rules)
