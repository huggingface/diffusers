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


import math

from .core import Conversion, Rule
from .transforms import Squeeze


def hunyuan_image_vae_conversion(config: dict) -> Conversion:
    """Map the original image VAE, whose convolutions retain a singleton temporal axis."""
    modules, resnets = [], []
    for component in ("encoder", "decoder"):
        encoder = component == "encoder"
        direction = "down" if encoder else "up"
        channels = list(config["block_out_channels"])
        if not encoder:
            channels.reverse()
        modules.extend((f"{component}.{name}", f"{component}.{name}") for name in ("conv_in", "conv_out", "norm_out"))
        resnets.extend(
            (f"{component}.mid.block_{i + 1}", f"{component}.mid_block.resnets.{i}", False) for i in range(2)
        )
        old, new = f"{component}.mid.attn_1", f"{component}.mid_block.attentions.0"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (("norm", "norm"), ("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("proj_out", "proj"))
        )
        previous, index = channels[0], 0
        for i, channel in enumerate(channels):
            for j in range(config["layers_per_block"] + int(not encoder)):
                resnets.append(
                    (
                        f"{component}.{direction}.{i}.block.{j}",
                        f"{component}.{direction}_blocks.{index}",
                        previous != channel,
                    )
                )
                index += 1
                previous = channel
            if i < int(math.log2(config["spatial_compression_ratio"])) and i < len(channels) - 1:
                modules.append(
                    (
                        f"{component}.{direction}.{i}.{direction}sample.conv",
                        f"{component}.{direction}_blocks.{index}.conv",
                    )
                )
                index += 1
                if config["downsample_match_channel" if encoder else "upsample_match_channel"]:
                    previous = channels[i + 1]
    for old, new, shortcut in resnets:
        modules.extend((f"{old}.{name}", f"{new}.{name}") for name in ("norm1", "norm2", "conv1", "conv2"))
        if shortcut:
            modules.append((old + ".nin_shortcut", new + ".conv_shortcut"))
    mapping, rules = {}, []
    temporal = config.get("original_format", "hunyuan_image_vae") == "hunyuan_image_vae"
    for old, new in modules:
        mapping[old + ".bias"] = new + ".bias"
        if temporal and "norm" not in old.rsplit(".", 1)[-1]:
            rules.append(Rule((old + ".weight",), (new + ".weight",), Squeeze(dim=2, ndim=5)))
        else:
            mapping[old + ".weight"] = new + ".weight"
    return Conversion(mapping=mapping, rules=rules)
