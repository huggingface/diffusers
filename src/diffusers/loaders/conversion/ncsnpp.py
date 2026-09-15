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
from .transforms import Permute


def ncsnpp_conversion(config):
    if not all("Skip" in kind for kind in (*config["down_block_types"], *config["up_block_types"])):
        raise ValueError("The NCSN++ checkpoint format requires SkipDown/SkipUp block types.")
    mapping = {"all_modules.0.W": "time_proj.weight"}
    modules = [
        ("all_modules.1", "time_embedding.linear_1"),
        ("all_modules.2", "time_embedding.linear_2"),
        ("all_modules.3", "conv_in"),
    ]
    resnets, attentions = [], []
    index = 4
    channels, layers = config["block_out_channels"], config["layers_per_block"]
    previous = channels[0]
    for i, channel in enumerate(channels):
        for j in range(layers):
            resnets.append((index, f"down_blocks.{i}.resnets.{j}", previous != channel))
            index += 1
            previous = channel
            if "Attn" in config["down_block_types"][i]:
                attentions.append((index, f"down_blocks.{i}.attentions.{j}"))
                index += 1
        if i < len(channels) - 1:
            resnets.append((index, f"down_blocks.{i}.resnet_down", True))
            index += 1
            modules.append((f"all_modules.{index}.Conv_0", f"down_blocks.{i}.skip_conv"))
            index += 1
    resnets.append((index, "mid_block.resnets.0", False))
    attentions.append((index + 1, "mid_block.attentions.0"))
    resnets.append((index + 2, "mid_block.resnets.1", False))
    index += 3
    reversed_channels = list(reversed(channels))
    previous = reversed_channels[0]
    for i, channel in enumerate(reversed_channels):
        next_channel = reversed_channels[min(i + 1, len(channels) - 1)]
        for j in range(layers + 1):
            skip = next_channel if j == layers else channel
            resnets.append((index, f"up_blocks.{i}.resnets.{j}", previous + skip != channel))
            previous = channel
            index += 1
        if "Attn" in config["up_block_types"][i]:
            attentions.append((index, f"up_blocks.{i}.attentions.0"))
            index += 1
        if i < len(channels) - 1:
            modules.append((f"all_modules.{index}", f"up_blocks.{i}.skip_norm"))
            modules.append((f"all_modules.{index + 1}", f"up_blocks.{i}.skip_conv"))
            resnets.append((index + 2, f"up_blocks.{i}.resnet_up", True))
            index += 3
    modules.extend([(f"all_modules.{index}", "conv_norm_out"), (f"all_modules.{index + 1}", "conv_out")])
    for index, new, shortcut in resnets:
        old = f"all_modules.{index}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("Conv_0", "conv1"),
                ("Conv_1", "conv2"),
                ("GroupNorm_0", "norm1"),
                ("GroupNorm_1", "norm2"),
                ("Dense_0", "time_emb_proj"),
            )
        )
        if shortcut:
            modules.append((old + ".Conv_2", new + ".conv_shortcut"))
    rules = []
    for index, new in attentions:
        old = f"all_modules.{index}"
        modules.append((old + ".GroupNorm_0", new + ".group_norm"))
        for i, projection in enumerate(("to_q", "to_k", "to_v", "to_out.0")):
            mapping[f"{old}.NIN_{i}.b"] = f"{new}.{projection}.bias"
            rules.append(Rule((f"{old}.NIN_{i}.W",), (f"{new}.{projection}.weight",), Permute((1, 0))))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
