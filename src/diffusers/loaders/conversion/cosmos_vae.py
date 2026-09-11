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


def cosmos_vae_conversion(config):
    modules = [("quant_conv.conv3d", "quant_conv"), ("post_quant_conv.conv3d", "post_quant_conv")]
    projections, resnets, attentions = [], [], []
    spatial = int(math.log2(config["spatial_compression_ratio"])) - int(math.log2(config["patch_size"]))
    temporal = int(math.log2(config["temporal_compression_ratio"])) - int(math.log2(config["patch_size"]))
    for component, field in (("encoder", "encoder_block_out_channels"), ("decoder", "decode_block_out_channels")):
        channels = list(config[field])
        if component == "decoder":
            channels.reverse()
        count = len(channels) - 1
        modules.append((f"{component}.norm_out.norm", f"{component}.norm_out.norm"))
        projections.extend((f"{component}.{name}", f"{component}.{name}") for name in ("conv_in", "conv_out"))
        resnets.extend(
            (f"{component}.mid.block_{j + 1}", f"{component}.mid_block.resnets.{j}", False) for j in range(2)
        )
        attentions.append((f"{component}.mid.attn_1", f"{component}.mid_block", 0))
        resolution = config["resolution"] // config["patch_size"]
        if component == "decoder":
            resolution //= 2 ** (count - 1)
        for i in range(count):
            encoder = component == "encoder"
            original_index = i if encoder else count - 1 - i
            direction = "down" if encoder else "up"
            old, new = f"{component}.{direction}.{original_index}", f"{component}.{direction}_blocks.{i}"
            depth = config["num_layers"] + int(not encoder)
            for j in range(depth):
                resnets.append((f"{old}.block.{j}", f"{new}.resnets.{j}", j == 0 and channels[i] != channels[i + 1]))
                if resolution in config["attention_resolutions"]:
                    attentions.append((f"{old}.attn.{j}", new, j))
            if i < count - 1:
                sampler = direction + "sample"
                if encoder:
                    enabled = (i < spatial, i < temporal)
                    resolution //= 2
                else:
                    time_up = 0 < i < temporal + 1
                    space_up = time_up or (i < spatial and spatial > temporal)
                    enabled = (time_up, space_up)
                    resolution *= 2
                for j, active in enumerate((*enabled, any(enabled)), 1):
                    if active:
                        modules.append((f"{old}.{sampler}.conv{j}.conv3d", f"{new}.{sampler}rs.0.conv{j}"))
    for old, new, shortcut in resnets:
        modules.extend((f"{old}.{norm}.norm", f"{new}.{norm}.norm") for norm in ("norm1", "norm2"))
        projections.extend((f"{old}.{conv}", f"{new}.{conv}") for conv in ("conv1", "conv2"))
        if shortcut:
            modules.append((old + ".nin_shortcut.conv3d", new + ".conv_shortcut"))
    for old, new in projections:
        modules.extend([(old + ".0.conv3d", new + ".conv_s"), (old + ".1.conv3d", new + ".conv_t")])
    for old, new, index in attentions:
        for j, kind in enumerate(("attentions", "temp_attentions")):
            a, b = f"{old}.{j}", f"{new}.{kind}.{index}"
            modules.append((a + ".norm.norm", b + ".norm.norm"))
            modules.extend(
                (f"{a}.{source}.conv3d", f"{b}.{target}")
                for source, target in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("proj_out", "to_out.0"))
            )
    return Conversion(mapping={f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
