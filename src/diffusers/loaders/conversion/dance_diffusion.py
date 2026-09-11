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
from .transforms import Chain, Reshape, Split


def dance_diffusion_conversion(config):
    count = len(config["down_block_types"])
    channels = config["block_out_channels"]
    modules, resnets, attentions = [], [], []
    mapping = {}
    fourier_dim = (config["time_embedding_dim"] or channels[0] * 2) // 2
    rules = [Rule(("timestep_embed.weight",), ("time_proj.weight",), Reshape((fourier_dim, 1), (fourier_dim,)))]

    def source_prefix(depth):
        return "net" if depth == 0 else "net.3." + "main.7." * (depth - 1) + "main"

    previous = config["in_channels"] + config["extra_in_channels"]
    for i, channel in enumerate(channels):
        old = source_prefix(i)
        for j in range(3):
            resnets.append(
                (f"{old}.{j if i == 0 else 2 * j + 1}", f"down_blocks.{i}.resnets.{j}", previous != channel, False)
            )
            previous = channel
            if "Attn" in config["down_block_types"][i]:
                attentions.append((f"{old}.{2 * j + 2}", f"down_blocks.{i}.attentions.{j}", channel))
        if i > 0:
            mapping[old + ".0.kernel"] = f"down_blocks.{i}.down.kernel"
    old = source_prefix(count)
    mapping[old + ".0.kernel"] = "mid_block.down.kernel"
    mapping[old + ".14.kernel"] = "mid_block.up.kernel"
    for j, source in enumerate((1, 3, 5, 8, 10, 12)):
        resnets.append((f"{old}.{source}", f"mid_block.resnets.{j}", False, False))
        attentions.append((f"{old}.{source + 1}", f"mid_block.attentions.{j}", channels[-1]))
    previous = channels[-1]
    for i, kind in enumerate(config["up_block_types"]):
        depth = count - i - 1
        old = source_prefix(depth)
        output = channels[depth - 1] if i < len(config["up_block_types"]) - 1 else config["out_channels"]
        middle = output if "Attn" in kind else previous
        for j in range(3):
            input_channel = 2 * previous if j == 0 else middle
            output_channel = output if j == 2 else middle
            resnets.append(
                (
                    f"{old}.{j + 4 if depth == 0 else 2 * j + 8}",
                    f"up_blocks.{i}.resnets.{j}",
                    input_channel != output_channel,
                    depth == 0 and j == 2,
                )
            )
            if "Attn" in kind:
                attentions.append((f"{old}.{2 * j + 9}", f"up_blocks.{i}.attentions.{j}", output_channel))
        previous = output
        if depth > 0:
            mapping[old + ".14.kernel"] = f"up_blocks.{i}.up.kernel"
    for old, new, shortcut, last in resnets:
        modules.extend(
            (f"{old}.main.{i}", f"{new}.{name}") for i, name in ((0, "conv_1"), (1, "group_norm_1"), (3, "conv_2"))
        )
        if not last:
            modules.append((old + ".main.4", new + ".group_norm_2"))
        if shortcut:
            mapping[old + ".skip.weight"] = new + ".conv_skip.weight"
    for old, new, channel in attentions:
        modules.append((old + ".norm", new + ".group_norm"))
        rules.append(
            Rule(
                (old + ".qkv_proj.weight",),
                tuple(f"{new}.{part}.weight" for part in ("query", "key", "value")),
                Chain((Reshape((3 * channel, channel, 1), (3 * channel, channel)), Split((channel,) * 3))),
            )
        )
        rules.append(
            Rule(
                (old + ".qkv_proj.bias",),
                tuple(f"{new}.{part}.bias" for part in ("query", "key", "value")),
                Split((channel,) * 3),
            )
        )
        rules.append(
            Rule(
                (old + ".out_proj.weight",),
                (new + ".proj_attn.weight",),
                Reshape((channel, channel, 1), (channel, channel)),
            )
        )
        mapping[old + ".out_proj.bias"] = new + ".proj_attn.bias"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
