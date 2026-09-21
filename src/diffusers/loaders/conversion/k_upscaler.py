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


def k_upscaler_conversion(config):
    if config["time_embedding_type"] != "fourier" or config["mid_block_type"] is not None:
        raise ValueError("Original k-upscaler uses Fourier time embeddings and no separate middle block.")
    channels = config["block_out_channels"]
    count = len(channels)
    time_dim = config.get("time_embedding_dim") or channels[0] * 2
    modules = [
        ("inner_model.proj_in", "conv_in"),
        ("inner_model.proj_out", "conv_out"),
        ("inner_model.mapping.0", "time_embedding.linear_1"),
        ("inner_model.mapping.2", "time_embedding.linear_2"),
    ]
    mapping = {}
    if config.get("time_cond_proj_dim") is not None:
        mapping["inner_model.mapping_cond.weight"] = "time_embedding.cond_proj.weight"
    rules = [
        Rule(
            ("inner_model.timestep_embed.weight",),
            ("time_proj.weight",),
            Reshape((time_dim // 2, 1), (time_dim // 2,)),
        )
    ]
    layers = config["layers_per_block"]
    layers = [layers] * count if isinstance(layers, int) else layers
    for direction in ("down", "up"):
        for i, kind in enumerate(config[f"{direction}_block_types"]):
            level = i if direction == "down" else count - 1 - i
            channel = channels[level]
            other = channels[max(level - 1, 0)]
            first_up = direction == "up" and other == channel == time_dim
            self_attn = i == count - 1 if direction == "down" else first_up
            attention = "CrossAttn" in kind
            stride = (3 if self_attn else 2) if attention else 1
            source = f"inner_model.u_net.{'d' if direction == 'down' else 'u'}_blocks.{i}"
            for j in range(layers[level]):
                index = stride * j + int(direction == "down")
                old, new = f"{source}.{index}", f"{direction}_blocks.{i}.resnets.{j}"
                modules.extend(
                    (f"{old}.{a}", f"{new}.{b}")
                    for a, b in (
                        ("main.0.mapper", "norm1.linear"),
                        ("main.2", "conv1"),
                        ("main.4.mapper", "norm2.linear"),
                        ("main.6", "conv2"),
                    )
                )
                if direction == "down":
                    input_width = other if j == 0 else channel
                    output_width = channel
                else:
                    input_width = channel * (1 if first_up else 2) if j == 0 else channel
                    output_width = other if j == layers[level] - 1 else channel
                if input_width != output_width:
                    mapping[f"{old}.skip.weight"] = f"{new}.conv_shortcut.weight"
                if not attention:
                    continue
                target = f"{direction}_blocks.{i}.attentions.{j}"
                width = channel if direction == "down" else output_width
                head_dim = config["attention_head_dim"]
                head_dim = head_dim[level] if isinstance(head_dim, (tuple, list)) else head_dim
                inner = width // head_dim * head_dim
                context = config["cross_attention_dim"]
                context = context[level] if isinstance(context, (tuple, list)) else context
                if self_attn:
                    old = f"{source}.{index + 1}"
                    modules.append((f"{old}.norm_in.mapper", f"{target}.norm1.linear"))
                    rules.append(
                        Rule(
                            (f"{old}.qkv_proj.weight",),
                            tuple(f"{target}.attn1.to_{part}.weight" for part in ("q", "k", "v")),
                            Chain((Reshape((3 * inner, width, 1, 1), (3 * inner, width)), Split((inner,) * 3))),
                        )
                    )
                    rules.append(
                        Rule(
                            (f"{old}.qkv_proj.bias",),
                            tuple(f"{target}.attn1.to_{part}.bias" for part in ("q", "k", "v")),
                            Split((inner,) * 3),
                        )
                    )
                    rules.append(
                        Rule(
                            (f"{old}.out_proj.weight",),
                            (f"{target}.attn1.to_out.0.weight",),
                            Reshape((width, inner, 1, 1), (width, inner)),
                        )
                    )
                    mapping[f"{old}.out_proj.bias"] = f"{target}.attn1.to_out.0.bias"
                old = f"{source}.{index + 1 + int(self_attn)}"
                modules.extend(
                    [
                        (f"{old}.norm_dec.mapper", f"{target}.norm2.linear"),
                        (f"{old}.norm_enc", f"{target}.attn2.norm_cross"),
                    ]
                )
                rules.append(
                    Rule(
                        (f"{old}.kv_proj.weight",),
                        (f"{target}.attn2.to_k.weight", f"{target}.attn2.to_v.weight"),
                        Split((inner,) * 2),
                    )
                )
                rules.append(
                    Rule(
                        (f"{old}.kv_proj.bias",),
                        (f"{target}.attn2.to_k.bias", f"{target}.attn2.to_v.bias"),
                        Split((inner,) * 2),
                    )
                )
                for a, b, shape in (("q_proj", "to_q", (inner, width)), ("out_proj", "to_out.0", (width, inner))):
                    rules.append(
                        Rule((f"{old}.{a}.weight",), (f"{target}.attn2.{b}.weight",), Reshape(shape + (1, 1), shape))
                    )
                    mapping[f"{old}.{a}.bias"] = f"{target}.attn2.{b}.bias"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)
