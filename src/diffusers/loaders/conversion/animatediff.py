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
from .transforms import MergeEqual


def animatediff_conversion(config):
    mapping, modules, rules = {}, [], []
    count = len(config["block_out_channels"])
    layers = config["motion_layers_per_block"]
    layers = [layers] * count if isinstance(layers, int) else list(layers)
    depths = config["motion_transformer_layers_per_block"]
    depths = [depths] * count if isinstance(depths, int) else list(depths)
    blocks = [(f"down_blocks.{i}", layers[i], depths[i]) for i in range(count)]
    blocks.extend((f"up_blocks.{i}", layers[count - 1 - i] + 1, depths[count - 1 - i]) for i in range(count))
    if config["use_motion_mid_block"]:
        blocks.append(
            (
                "mid_block",
                config["motion_mid_block_layers_per_block"],
                config["motion_transformer_layers_per_mid_block"],
            )
        )
    if config["conv_in_channels"]:
        modules.append(("conv_in", "conv_in"))
    for prefix, layers, depth in blocks:
        for i in range(layers):
            new = f"{prefix}.motion_modules.{i}"
            old = new + ".temporal_transformer"
            modules.extend((f"{old}.{name}", f"{new}.{name}") for name in ("norm", "proj_in", "proj_out"))
            n = depth[i] if isinstance(depth, (list, tuple)) else depth
            for j in range(n):
                a, b = f"{old}.transformer_blocks.{j}", f"{new}.transformer_blocks.{j}"
                modules.extend(
                    (f"{a}.{source}", f"{b}.{target}")
                    for source, target in (
                        ("norms.0", "norm1"),
                        ("norms.1", "norm2"),
                        ("ff_norm", "norm3"),
                        ("ff.net.0.proj", "ff.net.0.proj"),
                        ("ff.net.2", "ff.net.2"),
                        ("attention_blocks.0.to_out.0", "attn1.to_out.0"),
                        ("attention_blocks.1.to_out.0", "attn2.to_out.0"),
                    )
                )
                rules.append(
                    Rule(
                        tuple(f"{a}.attention_blocks.{k}.pos_encoder.pe" for k in (0, 1)),
                        (b + ".pos_embed.pe",),
                        MergeEqual(2),
                    )
                )
                mapping.update(
                    {
                        f"{a}.attention_blocks.{k}.to_{part}.weight": f"{b}.attn{k + 1}.to_{part}.weight"
                        for k in (0, 1)
                        for part in ("q", "k", "v")
                    }
                )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
