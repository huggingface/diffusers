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
from .ldm_unet import ldm_unet_conversion
from .transforms import Reshape


def unet_3d_conversion(config):
    spatial_config = {
        **config,
        "down_block_types": [name.replace("3D", "2D") for name in config["down_block_types"]],
        "up_block_types": [name.replace("3D", "2D") for name in config["up_block_types"]],
        "transformer_layers_per_block": 1,
    }
    base = ldm_unet_conversion(spatial_config)
    mapping, rules, resnets, attention_prefixes = {}, [], [], []
    channels = config["block_out_channels"]
    for old, new in base.mapping.items():
        if new.startswith("conv_norm_out.") and config["norm_num_groups"] is None:
            continue
        if new.startswith("mid_block.resnets.1."):
            old = old.replace("middle_block.2.", "middle_block.3.")
        elif ".downsamplers." in new:
            old = old.replace(".0.op.", ".op.")
        elif ".upsamplers." in new and config["up_block_types"][int(new.split(".")[1])] == "CrossAttnUpBlock3D":
            old = old.replace(".2.conv.", ".3.conv.")
        mapping[old] = new
        if ".resnets." in new and new.endswith(".norm1.weight"):
            resnets.append(
                (
                    old.removesuffix(".in_layers.0.weight"),
                    new.removesuffix(".norm1.weight").replace(".resnets.", ".temp_convs."),
                )
            )
        if ".attentions." in new and new.endswith(".proj_in.bias"):
            source = old.removesuffix(".proj_in.bias")
            target = new.removesuffix(".proj_in.bias")
            source = source.rsplit(".", 1)[0] + ".2"
            if target.startswith("mid_block"):
                channel = channels[-1]
            else:
                index = int(target.split(".")[1])
                channel = channels[-index - 1] if target.startswith("up_blocks") else channels[index]
            attention_prefixes.append((source, target.replace(".attentions.", ".temp_attentions."), channel, channel))
    for old, new in resnets:
        for i in range(1, 5):
            for layer in (0, 2 if i == 1 else 3):
                mapping.update(
                    {
                        f"{old}.temopral_conv.conv{i}.{layer}.{p}": f"{new}.conv{i}.{layer}.{p}"
                        for p in ("weight", "bias")
                    }
                )
    attention_prefixes.append(("input_blocks.0.1", "transformer_in", channels[0], 8 * config["attention_head_dim"]))
    for old, new, channel, inner in attention_prefixes:
        block_mapping, block_rules = _temporal_transformer_rules(old, new, channel, inner)
        mapping.update(block_mapping)
        rules.extend(block_rules)
    if config.get("time_cond_proj_dim") is not None:
        mapping["time_embed.cond_proj.weight"] = "time_embedding.cond_proj.weight"
    return Conversion(mapping=mapping, rules=tuple(rules))


def _temporal_transformer_rules(old, new, channel, inner):
    modules = [(old + ".norm", new + ".norm")]
    mapping = {f"{old}.{name}.bias": f"{new}.{name}.bias" for name in ("proj_in", "proj_out")}
    rules = [
        Rule((old + ".proj_in.weight",), (new + ".proj_in.weight",), Reshape((inner, channel, 1), (inner, channel))),
        Rule((old + ".proj_out.weight",), (new + ".proj_out.weight",), Reshape((channel, inner, 1), (channel, inner))),
    ]
    a, b = old + ".transformer_blocks.0", new + ".transformer_blocks.0"
    modules.extend(
        (f"{a}.{name}", f"{b}.{name}")
        for name in ("norm1", "norm2", "norm3", "ff.net.0.proj", "ff.net.2", "attn1.to_out.0", "attn2.to_out.0")
    )
    mapping.update(
        {
            f"{a}.{attn}.to_{part}.weight": f"{b}.{attn}.to_{part}.weight"
            for attn in ("attn1", "attn2")
            for part in ("q", "k", "v")
        }
    )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return mapping, rules
