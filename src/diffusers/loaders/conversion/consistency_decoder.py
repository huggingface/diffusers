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

import re

from .core import Conversion
from .ldm_vae import ldm_vae_conversion


def consistency_decoder_conversion(config):
    """Map the original Python decoder graph plus the LDM encoder/quantization component."""
    encoder_config = {
        "block_out_channels": config["encoder_block_out_channels"],
        "layers_per_block": config["encoder_layers_per_block"],
        "mid_block_add_attention": True,
        "down_block_types": config["encoder_down_block_types"],
    }
    encoder = ldm_vae_conversion(encoder_config)
    mapping = {
        old: new for old, new in encoder.mapping.items() if new.startswith("encoder.") or new.startswith("quant_conv.")
    }
    rules = [rule for rule in encoder.rules if all(key.startswith("encoder.") for key in rule.diffusers)]
    modules = [
        ("embed_image.f", "conv_in"),
        ("output.gn", "conv_norm_out"),
        ("output.f", "conv_out"),
        ("embed_time.f_1", "time_embedding.linear_1"),
        ("embed_time.f_2", "time_embedding.linear_2"),
    ]
    mapping["decoder.embed_time.emb.weight"] = "decoder_unet.time_proj.weight"
    channels, layers = config["decoder_block_out_channels"], config["decoder_layers_per_block"]
    count = len(channels)
    resnets = [(f"mid.{j}", f"mid_block.resnets.{j}", False) for j in range(2)]
    previous = channels[0]
    for i, channel in enumerate(channels):
        for j in range(layers):
            resnets.append((f"down.{i}.{j}", f"down_blocks.{i}.resnets.{j}", previous != channel))
            previous = channel
        if i < count - 1:
            resnets.append((f"down.{i}.{layers}", f"down_blocks.{i}.downsamplers.0", False))
    previous = channels[-1]
    for i, channel in enumerate(reversed(channels)):
        level = count - 1 - i
        for j in range(layers + 1):
            skip = channels[max(level - 1, 0)] if j == layers else channel
            resnets.append((f"up.{level}.{j}", f"up_blocks.{i}.resnets.{j}", previous + skip != channel))
            previous = channel
        if i < count - 1:
            resnets.append((f"up.{level}.{layers + 1}", f"up_blocks.{i}.upsamplers.0", False))
    for old, new, shortcut in resnets:
        pairs = [("gn_1", "norm1"), ("gn_2", "norm2"), ("f_1", "conv1"), ("f_2", "conv2"), ("f_t", "time_emb_proj")]
        if shortcut:
            pairs.append(("f_s", "conv_shortcut"))
        modules.extend((f"{old}.{a}", f"{new}.{b}") for a, b in pairs)
    if config["decoder_add_attention"]:
        raise ValueError("The original consistency decoder has no attention layers.")
    mapping.update(
        {f"decoder.{old}.{p}": f"decoder_unet.{new}.{p}" for old, new in modules for p in ("weight", "bias")}
    )
    if config.get("original_format") == "consistency_decoder_jit":

        def jit_key(key):
            if not key.startswith("decoder."):
                return key
            key = key.removeprefix("decoder.")
            if key == "embed_time.emb.weight":
                return "decoder.embed_time.weight"
            match = re.match(r"(down|up)\.(\d+)\.(\d+)\.(.*)", key)
            if match:
                direction, stage, index, tail = match.groups()
                sampler_index = layers if direction == "down" else layers + 1
                block = (
                    ("downsamp" if direction == "down" else "upsamp")
                    if int(index) == sampler_index
                    else "conv_" + index
                )
                key = f"{direction}_{stage}_{block}.{tail}"
            else:
                key = re.sub(r"^mid\.(\d+)\.", r"mid_\1.", key)
            prefix, leaf = key.rsplit(".", 1)
            module = prefix.rsplit(".", 1)[-1]
            if leaf == "bias":
                leaf = "b"
            elif leaf == "weight":
                leaf = "g" if module.startswith("gn") else "w"
            return f"decoder.blocks.{prefix}.{leaf}"

        mapping = {jit_key(old): new for old, new in mapping.items()}
    return Conversion(mapping=mapping, rules=rules)
