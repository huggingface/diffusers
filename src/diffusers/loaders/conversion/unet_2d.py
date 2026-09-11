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
from .transforms import Chain, Permute, Reshape, Split


def unet_2d_conversion(config):
    original_format = config.get("original_format")
    if original_format is None:
        original_format = "ncsnpp" if config["time_embedding_type"] == "fourier" else "ddpm"
    if original_format == "ncsnpp":
        from .ncsnpp import ncsnpp_conversion

        return ncsnpp_conversion(config)
    if original_format not in ("ddpm", "ldm", "consistency"):
        raise ValueError("UNet2D original_format must be 'ddpm', 'ldm', 'consistency', or 'ncsnpp'.")
    ddpm = original_format == "ddpm"
    mapping, modules, resnets, attentions = {}, [], [], []
    modules.extend(
        [
            ("temb.dense.0" if ddpm else "time_embed.0", "time_embedding.linear_1"),
            ("temb.dense.1" if ddpm else "time_embed.2", "time_embedding.linear_2"),
            ("conv_in" if ddpm else "input_blocks.0.0", "conv_in"),
            ("norm_out" if ddpm else "out.0", "conv_norm_out"),
            ("conv_out" if ddpm else "out.2", "conv_out"),
        ]
    )
    if config.get("num_class_embeds") is not None:
        mapping["label_emb.weight"] = "class_embedding.weight"
    channels, layers = config["block_out_channels"], config["layers_per_block"]
    count = len(channels)
    layers = [layers] * count if isinstance(layers, int) else layers
    previous, index = channels[0], 1
    for i, channel in enumerate(channels):
        attention = "Attn" in config["down_block_types"][i]
        for j in range(layers[i]):
            old = f"down.{i}.block.{j}" if ddpm else f"input_blocks.{index}.0"
            resnets.append((old, f"down_blocks.{i}.resnets.{j}", previous != channel))
            if attention:
                old = f"down.{i}.attn.{j}" if ddpm else f"input_blocks.{index}.1"
                attentions.append((old, f"down_blocks.{i}.attentions.{j}", channel))
            previous = channel
            index += 1
        if i < count - 1:
            old = f"down.{i}.downsample" if ddpm else f"input_blocks.{index}.0"
            new = f"down_blocks.{i}.downsamplers.0"
            residual = (
                config["down_block_types"][i] == "ResnetDownsampleBlock2D" or config.get("downsample_type") == "resnet"
            )
            if residual:
                resnets.append((old, new, False))
            else:
                modules.append((old + ".op", new + ".conv"))
            index += 1
    if config.get("mid_block_type", "UNetMidBlock2D") is not None:
        resnets.extend(
            (f"mid.block_{j + 1}" if ddpm else f"middle_block.{2 * j}", f"mid_block.resnets.{j}", False)
            for j in range(2)
        )
        if config.get("add_attention", True):
            attentions.append(("mid.attn_1" if ddpm else "middle_block.1", "mid_block.attentions.0", channels[-1]))
    reversed_channels = list(reversed(channels))
    previous, index = reversed_channels[0], 0
    for i, channel in enumerate(reversed_channels):
        attention = "Attn" in config["up_block_types"][i]
        next_channel = reversed_channels[min(i + 1, count - 1)]
        for j in range(layers[count - 1 - i] + 1):
            old = f"up.{count - 1 - i}.block.{j}" if ddpm else f"output_blocks.{index}.0"
            skip = next_channel if j == layers[count - 1 - i] else channel
            resnets.append((old, f"up_blocks.{i}.resnets.{j}", previous + skip != channel))
            if attention:
                old = f"up.{count - 1 - i}.attn.{j}" if ddpm else f"output_blocks.{index}.1"
                attentions.append((old, f"up_blocks.{i}.attentions.{j}", channel))
            previous = channel
            index += 1
        if i < count - 1:
            old = f"up.{count - 1 - i}.upsample" if ddpm else f"output_blocks.{index - 1}.{2 if attention else 1}"
            new = f"up_blocks.{i}.upsamplers.0"
            residual = (
                config["up_block_types"][i] == "ResnetUpsampleBlock2D" or config.get("upsample_type") == "resnet"
            )
            if residual:
                resnets.append((old, new, False))
            else:
                modules.append((old + ".conv", new + ".conv"))
    for old, new, shortcut in resnets:
        pairs = (
            [
                ("norm1", "norm1"),
                ("conv1", "conv1"),
                ("norm2", "norm2"),
                ("conv2", "conv2"),
                ("temb_proj", "time_emb_proj"),
            ]
            if ddpm
            else [
                ("in_layers.0", "norm1"),
                ("in_layers.2", "conv1"),
                ("out_layers.0", "norm2"),
                ("out_layers.3", "conv2"),
                ("emb_layers.1", "time_emb_proj"),
            ]
        )
        if shortcut:
            pairs.append(("nin_shortcut" if ddpm else "skip_connection", "conv_shortcut"))
        modules.extend((f"{old}.{a}", f"{new}.{b}") for a, b in pairs)
    rules = []
    for old, new, channel in attentions:
        modules.append((old + ".norm", new + ".group_norm"))
        head_dim = config["attention_head_dim"]
        if isinstance(head_dim, (tuple, list)):
            parts = new.split(".")
            index = (
                count - 1
                if parts[0] == "mid_block"
                else (count - 1 - int(parts[1]) if parts[0] == "up_blocks" else int(parts[1]))
            )
            head_dim = head_dim[index]
        dim = head_dim or channel
        heads = channel // dim
        inner = heads * dim
        if inner == 0:
            raise ValueError("Attention head dimension must not exceed the block width.")
        if ddpm:
            for source, target in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("proj_out", "to_out.0")):
                shape = (channel, inner) if source == "proj_out" else (inner, channel)
                rules.append(
                    Rule((f"{old}.{source}.weight",), (f"{new}.{target}.weight",), Reshape(shape + (1, 1), shape))
                )
                mapping[f"{old}.{source}.bias"] = f"{new}.{target}.bias"
        else:
            for p in ("weight", "bias"):
                trailing = (channel,) if p == "weight" else ()
                original_shape = (3 * inner, channel, 1) if p == "weight" else (3 * inner,)
                transforms = [Reshape(original_shape, (3 * inner,) + trailing)]
                if original_format == "ldm":
                    transforms.extend(
                        [
                            Reshape((3 * inner,) + trailing, (heads, 3, dim) + trailing),
                            Permute((1, 0, 2, 3) if trailing else (1, 0, 2)),
                            Reshape((3, heads, dim) + trailing, (3 * inner,) + trailing),
                        ]
                    )
                transforms.append(Split((inner,) * 3))
                rules.append(
                    Rule(
                        (f"{old}.qkv.{p}",),
                        tuple(f"{new}.to_{part}.{p}" for part in ("q", "k", "v")),
                        Chain(tuple(transforms)),
                    )
                )
            rules.append(
                Rule(
                    (old + ".proj_out.weight",),
                    (new + ".to_out.0.weight",),
                    Reshape((channel, inner, 1), (channel, inner)),
                )
            )
            mapping[old + ".proj_out.bias"] = new + ".to_out.0.bias"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
