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


def ltx2_audio_vae_conversion(config):
    mapping = {
        "per_channel_statistics.mean-of-means": "latents_mean",
        "per_channel_statistics.std-of-means": "latents_std",
    }
    modules, resnets, attentions = [], [], []
    suffix = ".conv" if config["causality_axis"] is not None else ""
    channels = [config["base_channels"] * m for m in config["ch_mult"]]
    for component in ("encoder", "decoder"):
        modules.extend(f"{component}.{name}{suffix}" for name in ("conv_in", "conv_out"))
        if config["norm_type"] == "group":
            modules.append(component + ".norm_out")
        resnets.extend((f"{component}.mid.block_{i}", False) for i in (1, 2))
        if config["mid_block_add_attention"]:
            attentions.append(component + ".mid.attn_1")
    previous = channels[0]
    for i, channel in enumerate(channels):
        for j in range(config["num_res_blocks"]):
            resnets.append((f"encoder.down.{i}.block.{j}", previous != channel))
            previous = channel
            if config["resolution"] // 2**i in (config["attn_resolutions"] or ()):
                attentions.append(f"encoder.down.{i}.attn.{j}")
        if i < len(channels) - 1:
            modules.append(f"encoder.down.{i}.downsample.conv")
    previous = channels[-1]
    for i in reversed(range(len(channels))):
        for j in range(config["num_res_blocks"] + 1):
            resnets.append((f"decoder.up.{i}.block.{j}", previous != channels[i]))
            previous = channels[i]
            if config["resolution"] // 2**i in (config["attn_resolutions"] or ()):
                attentions.append(f"decoder.up.{i}.attn.{j}")
        if i > 0:
            modules.append(f"decoder.up.{i}.upsample.conv{suffix}")
    for prefix, shortcut in resnets:
        modules.extend(prefix + "." + name + suffix for name in ("conv1", "conv2"))
        if shortcut:
            modules.append(prefix + ".nin_shortcut" + suffix)
        if config["norm_type"] == "group":
            modules.extend(prefix + "." + name for name in ("norm1", "norm2"))
    for prefix in attentions:
        modules.extend(prefix + "." + name for name in ("q", "k", "v", "proj_out"))
        if config["norm_type"] == "group":
            modules.append(prefix + ".norm")
    mapping.update({f"{name}.{p}": f"{name}.{p}" for name in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
