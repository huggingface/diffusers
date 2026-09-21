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


from .animatediff import animatediff_conversion
from .controlnet import controlnet_conversion
from .core import Conversion, Rule


def sparse_controlnet_conversion(config):
    spatial_config = {
        **config,
        "down_block_types": [kind.replace("Motion", "2D") for kind in config["down_block_types"]],
        "conditioning_embedding_out_channels": config["conditioning_embedding_out_channels"] or (16, 32, 96, 256),
    }
    base = controlnet_conversion(spatial_config)
    mapping = {
        key: key
        for key in sorted(base.diffusers_keys)
        if not (config["use_simplified_condition_embedding"] and key.startswith("controlnet_cond_embedding."))
    }
    if config["use_simplified_condition_embedding"]:
        mapping.update(
            {f"controlnet_cond_embedding.{p}": f"controlnet_cond_embedding.{p}" for p in ("weight", "bias")}
        )
    if config.get("transformer_layers_per_mid_block") is not None:
        mid = controlnet_conversion(
            {**spatial_config, "transformer_layers_per_block": config["transformer_layers_per_mid_block"]}
        )
        mapping = {old: new for old, new in mapping.items() if not new.startswith("mid_block.")}
        mapping.update({key: key for key in sorted(mid.diffusers_keys) if key.startswith("mid_block.")})
    motion = animatediff_conversion(
        {
            "block_out_channels": config["block_out_channels"],
            "motion_layers_per_block": config["layers_per_block"],
            "motion_transformer_layers_per_block": config["temporal_transformer_layers_per_block"],
            "use_motion_mid_block": False,
            "conv_in_channels": None,
        }
    )
    mapping.update(
        {
            old: new
            for old, new in motion.mapping.items()
            if new.startswith("down_blocks.") and ".attn2." not in new and ".norm2." not in new
        }
    )
    rules = tuple(
        Rule((rule.original[0],), rule.diffusers)
        for rule in motion.rules
        if rule.diffusers[0].startswith("down_blocks.")
    )
    return Conversion(mapping=mapping, rules=rules)
