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


def ltx_vae_conversion(config):
    return Conversion(mapping=_ltx_vae_mapping(config, ltx2=False))


def _ltx_vae_mapping(config, ltx2):
    mapping = {
        "per_channel_statistics.mean-of-means": "latents_mean",
        "per_channel_statistics.std-of-means": "latents_std",
    }
    modules, resnets, embeddings = [], [], []
    for component in ("encoder", "decoder"):
        modules.extend((f"{component}.{name}.conv", f"{component}.{name}.conv") for name in ("conv_in", "conv_out"))
    channels = config["block_out_channels"]
    modern = ltx2 or config["down_block_types"][-1] == "LTXVideo095DownBlock3D"
    count = len(channels) - (1 if modern and not ltx2 else 0)
    source_index = 0
    for i in range(count):
        old, new = f"encoder.down_blocks.{source_index}", f"encoder.down_blocks.{i}"
        resnets.extend(
            (f"{old}.res_blocks.{j}", f"{new}.resnets.{j}", False, False, False)
            for j in range(config["layers_per_block"][i])
        )
        source_index += 1
        if config["spatio_temporal_scaling"][i]:
            suffix = "conv" if not modern or config["downsample_type"][i] == "conv" else "conv.conv"
            modules.append((f"encoder.down_blocks.{source_index}.{suffix}", f"{new}.downsamplers.0.{suffix}"))
            source_index += 1
        if not modern and i + 1 < count and channels[i] != channels[i + 1]:
            resnets.append((f"encoder.down_blocks.{source_index}", new + ".conv_out", True, False, False))
            source_index += 1
    resnets.extend(
        (f"encoder.down_blocks.{source_index}.res_blocks.{j}", f"encoder.mid_block.resnets.{j}", False, False, False)
        for j in range(config["layers_per_block"][-1])
    )
    channels = list(reversed(config["decoder_block_out_channels"] or config["block_out_channels"]))
    layers = list(reversed(config["decoder_layers_per_block"] or config["layers_per_block"]))
    scaling = list(reversed(config["decoder_spatio_temporal_scaling"] or config["spatio_temporal_scaling"]))
    noise = list(reversed(config["decoder_inject_noise"]))
    factors = list(reversed(config["upsample_factor"]))
    timed = config["timestep_conditioning"]
    resnets.extend(
        (f"decoder.up_blocks.0.res_blocks.{j}", f"decoder.mid_block.resnets.{j}", False, noise[0], timed)
        for j in range(layers[0])
    )
    if timed:
        embeddings.append(("decoder.up_blocks.0.time_embedder", "decoder.mid_block.time_embedder"))
        embeddings.append(("decoder.time_embedder" if ltx2 else "decoder.last_time_embedder", "decoder.time_embedder"))
        mapping["decoder.scale_shift_table" if ltx2 else "decoder.last_scale_shift_table"] = (
            "decoder.scale_shift_table"
        )
        mapping["decoder.timestep_scale_multiplier"] = "decoder.timestep_scale_multiplier"
    source_index = 1
    previous = channels[0]
    for i, out_channels in enumerate(channels):
        new = f"decoder.up_blocks.{i}"
        if previous != out_channels:
            resnets.append((f"decoder.up_blocks.{source_index}", new + ".conv_in", True, noise[i + 1], timed))
            source_index += 1
        if scaling[i]:
            modules.append((f"decoder.up_blocks.{source_index}.conv.conv", new + ".upsamplers.0.conv.conv"))
            source_index += 1
        old = f"decoder.up_blocks.{source_index}"
        resnets.extend(
            (f"{old}.res_blocks.{j}", f"{new}.resnets.{j}", False, noise[i + 1], timed) for j in range(layers[i + 1])
        )
        if timed:
            embeddings.append((old + ".time_embedder", new + ".time_embedder"))
        source_index += 1
        previous = out_channels // factors[i]
    for old, new, shortcut, inject, timed in resnets:
        modules.extend((f"{old}.{name}.conv", f"{new}.{name}.conv") for name in ("conv1", "conv2"))
        if shortcut:
            modules.append((old + (".conv_shortcut.conv" if ltx2 else ".conv_shortcut"), new + ".conv_shortcut.conv"))
            modules.append((old + (".norm3" if ltx2 else ".norm3.norm"), new + ".norm3"))
        if inject:
            for name in ("per_channel_scale1", "per_channel_scale2"):
                mapping[f"{old}.{name}"] = f"{new}.{name}"
        if timed:
            mapping[old + ".scale_shift_table"] = new + ".scale_shift_table"
    for old, new in embeddings:
        modules.extend(
            (f"{old}.timestep_embedder.{name}", f"{new}.timestep_embedder.{name}") for name in ("linear_1", "linear_2")
        )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return mapping
