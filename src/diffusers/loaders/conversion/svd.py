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
from .ldm_unet import ldm_unet_conversion


def svd_conversion(config):
    spatial_config = {
        **config,
        "down_block_types": [name.replace("SpatioTemporal", "2D") for name in config["down_block_types"]],
        "up_block_types": [name.replace("SpatioTemporal", "2D") for name in config["up_block_types"]],
        "addition_embed_type": "text_time",
    }
    base = ldm_unet_conversion(spatial_config)
    mapping, resnets, attention_prefixes = {}, [], set()
    for old, new in base.mapping.items():
        if ".resnets." in new:
            prefix, rest = new.split(".resnets.")
            index, suffix = rest.split(".", 1)
            parent = f"{prefix}.resnets.{index}"
            mapping[old] = parent + ".spatial_res_block." + suffix
            if suffix == "norm1.weight":
                resnets.append((old.removesuffix(".in_layers.0.weight"), parent))
        else:
            mapping[old] = new
        if ".attentions." in new and ".transformer_blocks." in new:
            mapping[old.replace(".transformer_blocks.", ".time_stack.")] = new.replace(
                ".transformer_blocks.", ".temporal_transformer_blocks."
            )
        if ".attentions." in new and new.endswith(".proj_in.bias"):
            attention_prefixes.add((old.removesuffix(".proj_in.bias"), new.removesuffix(".proj_in.bias")))
    for old, new in resnets:
        mapping[old + ".time_mixer.mix_factor"] = new + ".time_mixer.mix_factor"
        for a, b in (
            ("in_layers.0", "norm1"),
            ("in_layers.2", "conv1"),
            ("out_layers.0", "norm2"),
            ("out_layers.3", "conv2"),
            ("emb_layers.1", "time_emb_proj"),
        ):
            mapping.update(
                {f"{old}.time_stack.{a}.{p}": f"{new}.temporal_res_block.{b}.{p}" for p in ("weight", "bias")}
            )
    for old, new in sorted(attention_prefixes):
        mapping[old + ".time_mixer.mix_factor"] = new + ".time_mixer.mix_factor"
        for i, j in ((0, 1), (2, 2)):
            mapping.update(
                {f"{old}.time_pos_embed.{i}.{p}": f"{new}.time_pos_embed.linear_{j}.{p}" for p in ("weight", "bias")}
            )
        indices = {
            int(key.removeprefix(new + ".transformer_blocks.").split(".")[0])
            for key in base.diffusers_keys
            if key.startswith(new + ".transformer_blocks.")
        }
        for i in sorted(indices):
            mapping.update(
                {
                    f"{old}.time_stack.{i}.{name}.{p}": f"{new}.temporal_transformer_blocks.{i}.{name}.{p}"
                    for name in ("norm_in", "ff_in.net.0.proj", "ff_in.net.2")
                    for p in ("weight", "bias")
                }
            )
    return Conversion(mapping=mapping)
