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
from .wan import wan_conversion


def wan_vace_conversion(config):
    base = wan_conversion(config)
    mapping = dict(base.mapping)
    for p in ("weight", "bias"):
        mapping[f"vace_patch_embedding.{p}"] = f"vace_patch_embedding.{p}"
    block_mapping = {
        old.removeprefix("blocks.0."): new.removeprefix("blocks.0.")
        for old, new in base.mapping.items()
        if old.startswith("blocks.0.")
    }
    for i in range(len(config["vace_layers"])):
        prefix = f"vace_blocks.{i}"
        mapping.update({f"{prefix}.{old}": f"{prefix}.{new}" for old, new in block_mapping.items()})
        for p in ("weight", "bias"):
            mapping[f"{prefix}.after_proj.{p}"] = f"{prefix}.proj_out.{p}"
            if i == 0:
                mapping[f"{prefix}.before_proj.{p}"] = f"{prefix}.proj_in.{p}"
    return Conversion(mapping=mapping)
