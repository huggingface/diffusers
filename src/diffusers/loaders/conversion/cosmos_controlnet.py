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
from .cosmos import cosmos_conversion


def cosmos_controlnet_conversion(config):
    # Shared modules come from the base checkpoint; keep their source namespace distinct from control weights.
    base = cosmos_conversion({**config, "num_layers": config["n_controlnet_blocks"], "original_format": "cosmos2"})
    mapping = {}
    for old, new in base.mapping.items():
        if old.startswith("final_layer."):
            continue
        if old.startswith("blocks."):
            mapping[old.replace("blocks.", "control_blocks.", 1)] = new.replace(
                "transformer_blocks.", "control_blocks.", 1
            )
        else:
            mapping["base." + old] = new.replace("patch_embed.", "patch_embed_base.", 1)
    mapping["control_embedder.proj.1.weight"] = "patch_embed.proj.weight"
    for i in range(config["n_controlnet_blocks"]):
        for p in ("weight", "bias"):
            key = f"control_blocks.{i}.after_proj.{p}"
            mapping[key] = key
            if i == 0:
                key = f"control_blocks.{i}.before_proj.{p}"
                mapping[key] = key
    return Conversion(mapping=mapping)
