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


def flux2_vae_conversion(config):
    encoder = ldm_vae_conversion(config)
    decoder = ldm_vae_conversion(
        {**config, "block_out_channels": config["decoder_block_out_channels"] or config["block_out_channels"]}
    )
    mapping = {}
    for old, new in encoder.mapping.items():
        if new.startswith("encoder."):
            mapping[old] = new
        elif new.startswith("quant_conv."):
            mapping["encoder." + old] = new
    for old, new in decoder.mapping.items():
        if new.startswith("decoder."):
            mapping[old] = new
        elif new.startswith("post_quant_conv."):
            mapping["decoder." + old] = new
    mapping.update({f"bn.{name}": f"bn.{name}" for name in ("running_mean", "running_var", "num_batches_tracked")})
    rules = tuple(rule for rule in encoder.rules if rule.diffusers[0].startswith("encoder.")) + tuple(
        rule for rule in decoder.rules if rule.diffusers[0].startswith("decoder.")
    )
    return Conversion(mapping=mapping, rules=rules)
