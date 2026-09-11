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
from .ldm_vae import ldm_vae_conversion


def asymmetric_vae_conversion(config):
    encoder = ldm_vae_conversion(
        {
            **config,
            "block_out_channels": config["down_block_out_channels"],
            "layers_per_block": config["layers_per_down_block"],
            "mid_block_add_attention": True,
        }
    )
    decoder = ldm_vae_conversion(
        {
            **config,
            "block_out_channels": config["up_block_out_channels"],
            "layers_per_block": config["layers_per_up_block"],
            "mid_block_add_attention": True,
        }
    )
    mapping = {old: new for old, new in encoder.mapping.items() if new.startswith(("encoder.", "quant_conv."))}
    mapping.update(
        {old: new for old, new in decoder.mapping.items() if new.startswith(("decoder.", "post_quant_conv."))}
    )
    mapping.update(
        {
            f"decoder.encoder.layers.{i}.{p}": f"decoder.condition_encoder.layers.{i}.{p}"
            for i in range(5)
            for p in ("weight", "bias")
        }
    )
    rules = tuple(rule for rule in encoder.rules if rule.diffusers[0].startswith("encoder.")) + tuple(
        rule for rule in decoder.rules if rule.diffusers[0].startswith("decoder.")
    )
    return Conversion(mapping=mapping, rules=rules)
