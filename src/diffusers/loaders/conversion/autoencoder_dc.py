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


def autoencoder_dc_conversion(config):
    mapping, modules, rules = {}, [], []
    encoder_stem_downsample = config["encoder_layers_per_block"][0] == 0
    decoder_head_upsample = config["decoder_layers_per_block"][0] == 0
    encoder_original = "encoder.project_in.conv" + (
        ".conv" if encoder_stem_downsample and config["downsample_block_type"] == "pixel_unshuffle" else ""
    )
    modules.append((encoder_original, "encoder.conv_in.conv" if encoder_stem_downsample else "encoder.conv_in"))
    modules.append(
        (
            "encoder.project_out." + ("main." if config["encoder_out_shortcut"] else "") + "op_list.0.conv",
            "encoder.conv_out",
        )
    )
    modules.append(
        ("decoder.project_in." + ("main." if config["decoder_in_shortcut"] else "") + "conv", "decoder.conv_in")
    )
    modules.append(("decoder.project_out.op_list.0", "decoder.norm_out"))
    modules.append(
        (
            "decoder.project_out.op_list.2.conv" + (".conv" if decoder_head_upsample else ""),
            "decoder.conv_out.conv" if decoder_head_upsample else "decoder.conv_out",
        )
    )
    for component, direction in (("encoder", "down"), ("decoder", "up")):
        channels = config[f"{component}_block_out_channels"]
        layers = config[f"{component}_layers_per_block"]
        kinds = config[f"{component}_block_types"]
        kinds = [kinds] * len(channels) if isinstance(kinds, str) else kinds
        norms = "rms_norm" if component == "encoder" else config["decoder_norm_types"]
        norms = [norms] * len(channels) if isinstance(norms, str) else norms
        for i, channel in enumerate(channels):
            resample = i < len(channels) - 1 and layers[i] > 0
            if resample:
                j = layers[i] if component == "encoder" else 0
                old = f"{component}.stages.{i}.op_list.{j}.main.conv"
                if component == "decoder" or config["downsample_block_type"] == "pixel_unshuffle":
                    old += ".conv"
                modules.append((old, f"{component}.{direction}_blocks.{i}.{j}.conv"))
            for j in range(layers[i]):
                index = j + int(component == "decoder" and resample)
                old = f"{component}.stages.{i}.op_list.{index}"
                new = f"{component}.{direction}_blocks.{i}.{index}"
                if kinds[i] == "ResBlock":
                    modules.append((old + ".main.conv1.conv", new + ".conv1"))
                    mapping[old + ".main.conv2.conv.weight"] = new + ".conv2.weight"
                    modules.append((old + ".main.conv2.norm", new + ".norm"))
                elif kinds[i] == "EfficientViTBlock":
                    source, target = old + ".context_module.main", new + ".attn"
                    rules.append(
                        Rule(
                            (source + ".qkv.conv.weight",),
                            tuple(f"{target}.to_{part}.weight" for part in ("q", "k", "v")),
                            Chain(
                                (Reshape((3 * channel, channel, 1, 1), (3 * channel, channel)), Split((channel,) * 3))
                            ),
                        )
                    )
                    scales = config[f"{component}_qkv_multiscales"][i]
                    for k in range(len(scales)):
                        mapping[f"{source}.aggreg.{k}.0.weight"] = f"{target}.to_qkv_multiscale.{k}.proj_in.weight"
                        mapping[f"{source}.aggreg.{k}.1.weight"] = f"{target}.to_qkv_multiscale.{k}.proj_out.weight"
                    width = channel * (len(scales) + 1)
                    rules.append(
                        Rule(
                            (source + ".proj.conv.weight",),
                            (target + ".to_out.weight",),
                            Reshape((channel, width, 1, 1), (channel, width)),
                        )
                    )
                    modules.append((source + ".proj.norm", target + ".norm_out"))
                    for a, b in (
                        ("inverted_conv.conv", "conv_inverted"),
                        ("depth_conv.conv", "conv_depth"),
                        ("point_conv.norm", "norm"),
                    ):
                        modules.append((f"{old}.local_module.main.{a}", f"{new}.conv_out.{b}"))
                    mapping[old + ".local_module.main.point_conv.conv.weight"] = new + ".conv_out.conv_point.weight"
                else:
                    raise ValueError(f"Unknown DC-AE block type {kinds[i]}.")
                if norms[i] == "batch_norm":
                    source_norm = old + (
                        ".main.conv2.norm" if kinds[i] == "ResBlock" else ".context_module.main.proj.norm"
                    )
                    target_norm = new + (".norm" if kinds[i] == "ResBlock" else ".attn.norm_out")
                    for name in ("running_mean", "running_var", "num_batches_tracked"):
                        mapping[f"{source_norm}.{name}"] = f"{target_norm}.{name}"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
