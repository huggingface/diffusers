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
from .transforms import Reshape


def ldm_vae_conversion(config):
    modules, resnets, rules, attentions = [], [], [], []
    channels = config["block_out_channels"]
    layers = config["layers_per_block"]
    for name in ("quant_conv", "post_quant_conv"):
        if config.get("use_" + name, True):
            modules.append((name, name))
    for component in ("encoder", "decoder"):
        modules.extend(
            (f"{component}.{old}", f"{component}.{new}")
            for old, new in (("conv_in", "conv_in"), ("conv_out", "conv_out"), ("norm_out", "conv_norm_out"))
        )
        resnets.extend(
            (f"{component}.mid.block_{i + 1}", f"{component}.mid_block.resnets.{i}", False) for i in range(2)
        )
        if config["mid_block_add_attention"]:
            attentions.append((f"{component}.mid.attn_1", f"{component}.mid_block.attentions.0", channels[-1]))
    previous = channels[0]
    for i, channel in enumerate(channels):
        for j in range(layers):
            resnets.append(
                (f"encoder.down.{i}.block.{j}", f"encoder.down_blocks.{i}.resnets.{j}", previous != channel)
            )
            previous = channel
        if i < len(channels) - 1:
            modules.append((f"encoder.down.{i}.downsample.conv", f"encoder.down_blocks.{i}.downsamplers.0.conv"))
    previous = channels[-1]
    for i, channel in enumerate(reversed(channels)):
        original_index = len(channels) - 1 - i
        for j in range(layers + 1):
            resnets.append(
                (f"decoder.up.{original_index}.block.{j}", f"decoder.up_blocks.{i}.resnets.{j}", previous != channel)
            )
            previous = channel
        if i < len(channels) - 1:
            modules.append((f"decoder.up.{original_index}.upsample.conv", f"decoder.up_blocks.{i}.upsamplers.0.conv"))
    for old, new, shortcut in resnets:
        modules.extend((f"{old}.{name}", f"{new}.{name}") for name in ("norm1", "norm2", "conv1", "conv2"))
        if shortcut:
            modules.append((old + ".nin_shortcut", new + ".conv_shortcut"))
    for component, direction in (("encoder", "down"), ("decoder", "up")):
        block_types = config.get(f"{direction}_block_types", ())
        for i, kind in enumerate(block_types):
            if not kind.startswith("Attn"):
                continue
            original_index = i if component == "encoder" else len(channels) - 1 - i
            channel = channels[original_index]
            for j in range(layers + int(component == "decoder")):
                attentions.append(
                    (
                        f"{component}.{direction}.{original_index}.attn.{j}",
                        f"{component}.{direction}_blocks.{i}.attentions.{j}",
                        channel,
                    )
                )
    for old, new, width in attentions:
        modules.append((old + ".norm", new + ".group_norm"))
        for a, b in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("proj_out", "to_out.0")):
            rules.append(
                Rule((f"{old}.{a}.weight",), (f"{new}.{b}.weight",), Reshape((width, width, 1, 1), (width, width)))
            )
            modules.append((f"{old}.{a}", f"{new}.{b}"))
    transformed = {key for rule in rules for key in rule.original}
    mapping = {
        f"{old}.{p}": f"{new}.{p}"
        for old, new in modules
        for p in ("weight", "bias")
        if f"{old}.{p}" not in transformed
    }
    return Conversion(mapping=mapping, rules=tuple(rules))
