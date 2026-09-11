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

from .core import Conversion


def hunyuan_video15_vae_conversion(config):
    mapping, modules, resnets = {}, [], []
    spatial = int(math.log2(config["spatial_compression_ratio"]))
    temporal = int(math.log2(config["temporal_compression_ratio"]))
    for component in ("encoder", "decoder"):
        encoder = component == "encoder"
        direction = "down" if encoder else "up"
        channels = list(config["block_out_channels"])
        if not encoder:
            channels.reverse()
        modules.extend((f"{component}.{name}.conv", f"{component}.{name}.conv") for name in ("conv_in", "conv_out"))
        mapping[f"{component}.norm_out.gamma"] = f"{component}.norm_out.gamma"
        resnets.extend(
            (f"{component}.mid.block_{j + 1}", f"{component}.mid_block.resnets.{j}", False) for j in range(2)
        )
        old, new = f"{component}.mid.attn_1", f"{component}.mid_block.attentions.0"
        mapping[old + ".norm.gamma"] = new + ".norm.gamma"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("proj_out", "proj_out"))
        )
        previous = channels[0]
        for i, channel in enumerate(channels):
            old, new = f"{component}.{direction}.{i}", f"{component}.{direction}_blocks.{i}"
            for j in range(config["layers_per_block"] + int(not encoder)):
                resnets.append((f"{old}.block.{j}", f"{new}.resnets.{j}", previous != channel))
                previous = channel
            resampling = i < spatial if encoder else i < max(spatial, temporal)
            if resampling:
                modules.append((f"{old}.{direction}sample.conv.conv", f"{new}.{direction}samplers.0.conv.conv"))
                if config["downsample_match_channel" if encoder else "upsample_match_channel"]:
                    previous = channels[i + 1]
    for old, new, shortcut in resnets:
        mapping.update({f"{old}.{norm}.gamma": f"{new}.{norm}.gamma" for norm in ("norm1", "norm2")})
        modules.extend((f"{old}.{conv}.conv", f"{new}.{conv}.conv") for conv in ("conv1", "conv2"))
        if shortcut:
            modules.append((old + ".nin_shortcut", new + ".conv_shortcut"))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
