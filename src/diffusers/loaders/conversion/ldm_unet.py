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


def ldm_unet_conversion(config):
    if any(kind.startswith("K") for kind in config["down_block_types"]):
        from .k_upscaler import k_upscaler_conversion

        return k_upscaler_conversion(config)
    if config.get("addition_embed_type") == "text" and any(
        "SimpleCrossAttn" in kind for kind in config["down_block_types"]
    ):
        from .if_unet import if_unet_conversion

        return if_unet_conversion(config)
    if config.get("attention_type") in ("gated", "gated-text-image"):
        from .gligen import gligen_conversion

        return gligen_conversion(config)
    if config.get("original_format") == "versatile_image":
        from .versatile_image_unet import versatile_image_unet_conversion

        return versatile_image_unet_conversion(config)
    if any("SimpleCrossAttn" in kind for kind in config["down_block_types"]):
        from .unclip_unet import unclip_unet_conversion

        return unclip_unet_conversion(config)
    return Conversion(mapping=_ldm_unet_mapping(config, controlnet=False))


def _ldm_unet_mapping(config, controlnet):
    modules = [
        ("time_embed.0", "time_embedding.linear_1"),
        ("time_embed.2", "time_embedding.linear_2"),
        ("input_blocks.0.0", "conv_in"),
    ]
    mapping, resnets, attentions = {}, [], []
    channels = config["block_out_channels"]
    count = len(channels)
    layers = config["layers_per_block"]
    layers = [layers] * count if isinstance(layers, int) else layers
    depths = config["transformer_layers_per_block"]
    depths = [depths] * count if isinstance(depths, int) else depths
    if config.get("class_embed_type") in ("timestep", "projection"):
        modules.extend([("label_emb.0.0", "class_embedding.linear_1"), ("label_emb.0.2", "class_embedding.linear_2")])
    elif config.get("class_embed_type") == "simple_projection":
        modules.append(("film_emb", "class_embedding"))
    elif config.get("num_class_embeds") is not None:
        mapping["label_emb.weight"] = "class_embedding.weight"
    if config.get("addition_embed_type") == "text_time":
        modules.extend([("label_emb.0.0", "add_embedding.linear_1"), ("label_emb.0.2", "add_embedding.linear_2")])
    idx, previous = 1, channels[0]
    for i, channel in enumerate(channels):
        for j in range(layers[i]):
            resnets.append((f"input_blocks.{idx}.0", f"down_blocks.{i}.resnets.{j}", previous != channel))
            previous = channel
            if config["down_block_types"][i] == "CrossAttnDownBlock2D":
                depth = depths[i][j] if isinstance(depths[i], (list, tuple)) else depths[i]
                attentions.append((f"input_blocks.{idx}.1", f"down_blocks.{i}.attentions.{j}", depth))
            idx += 1
        if i < count - 1:
            modules.append((f"input_blocks.{idx}.0.op", f"down_blocks.{i}.downsamplers.0.conv"))
            idx += 1
    if config.get("mid_block_type", "UNetMidBlock2DCrossAttn") is not None:
        resnets.extend((f"middle_block.{i * 2}", f"mid_block.resnets.{i}", False) for i in range(2))
        depth = depths[-1][-1] if isinstance(depths[-1], (list, tuple)) else depths[-1]
        attentions.append(("middle_block.1", "mid_block.attentions.0", depth))
    if controlnet:
        modules.extend((f"zero_convs.{i}.0", f"controlnet_down_blocks.{i}") for i in range(idx))
        modules.append(("middle_block_out.0", "controlnet_mid_block"))
        cond_layers = 2 * (len(config["conditioning_embedding_out_channels"]) - 1)
        modules.append(("input_hint_block.0", "controlnet_cond_embedding.conv_in"))
        modules.extend(
            (f"input_hint_block.{2 * (i + 1)}", f"controlnet_cond_embedding.blocks.{i}") for i in range(cond_layers)
        )
        modules.append((f"input_hint_block.{2 * (cond_layers + 1)}", "controlnet_cond_embedding.conv_out"))
    else:
        modules.extend([("out.0", "conv_norm_out"), ("out.2", "conv_out")])
        up_channels = list(reversed(channels))
        up_layers = list(reversed(layers))
        up_depths = config.get("reverse_transformer_layers_per_block") or list(reversed(depths))
        previous, idx = up_channels[0], 0
        for i, channel in enumerate(up_channels):
            input_channel = up_channels[min(i + 1, count - 1)]
            attention = config["up_block_types"][i] == "CrossAttnUpBlock2D"
            for j in range(up_layers[i] + 1):
                skip_channel = input_channel if j == up_layers[i] else channel
                resnets.append(
                    (f"output_blocks.{idx}.0", f"up_blocks.{i}.resnets.{j}", previous + skip_channel != channel)
                )
                previous = channel
                if attention:
                    depth = up_depths[i][j] if isinstance(up_depths[i], (list, tuple)) else up_depths[i]
                    attentions.append((f"output_blocks.{idx}.1", f"up_blocks.{i}.attentions.{j}", depth))
                if j == up_layers[i] and i < count - 1:
                    modules.append(
                        (f"output_blocks.{idx}.{2 if attention else 1}.conv", f"up_blocks.{i}.upsamplers.0.conv")
                    )
                idx += 1
    for old, new, shortcut in resnets:
        pairs = [
            ("in_layers.0", "norm1"),
            ("in_layers.2", "conv1"),
            ("out_layers.0", "norm2"),
            ("out_layers.3", "conv2"),
            ("emb_layers.1", "time_emb_proj"),
        ]
        if shortcut:
            pairs.append(("skip_connection", "conv_shortcut"))
        modules.extend((f"{old}.{a}", f"{new}.{b}") for a, b in pairs)
    for old, new, depth in attentions:
        modules.extend((f"{old}.{name}", f"{new}.{name}") for name in ("norm", "proj_in", "proj_out"))
        for i in range(depth):
            a, b = f"{old}.transformer_blocks.{i}", f"{new}.transformer_blocks.{i}"
            modules.extend(
                (f"{a}.{name}", f"{b}.{name}")
                for name in (
                    "norm1",
                    "norm2",
                    "norm3",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "attn1.to_out.0",
                    "attn2.to_out.0",
                )
            )
            mapping.update(
                {
                    f"{a}.{attn}.to_{part}.weight": f"{b}.{attn}.to_{part}.weight"
                    for attn in ("attn1", "attn2")
                    for part in ("q", "k", "v")
                }
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return mapping
