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
from .transforms import Chain, Permute, ReorderChunks, Reshape, Split


def minimax_h3_vae_conversion(config):
    modules = [
        (name, name)
        for name in (
            "quant_conv",
            "post_quant_conv",
            "encoder.conv_in",
            "encoder.norm_out",
            "encoder.conv_out",
            "decoder.norm_out",
            "decoder.proj_out",
        )
    ]
    modules.append(("decoder.x_embedder", "decoder.proj_in"))
    mapping = {"decoder.register_tokens": "decoder.register_tokens"}
    previous = config["block_out_channels"][0]
    for i, channel in enumerate(config["block_out_channels"]):
        for j in range(config["layers_per_block"]):
            old, new = f"encoder.down.{i}.block.{j}", f"encoder.down_blocks.{i}.resnets.{j}"
            modules.extend((f"{old}.{name}", f"{new}.{name}") for name in ("norm1", "conv1", "norm2", "conv2"))
            if previous != channel:
                modules.append((old + ".nin_shortcut", new + ".conv_shortcut"))
            previous = channel
        if config["spatial_downsample_factors"][i] * config["temporal_downsample_factors"][i] > 1:
            modules.append((f"encoder.down.{i}.downsample.conv", f"encoder.down_blocks.{i}.downsamplers.0.conv"))
    heads, head_dim = config["decoder_num_attention_heads"], config["decoder_attention_head_dim"]
    hidden = heads * head_dim
    rules = []
    for i in range(config["decoder_num_layers"]):
        prefix = f"decoder.transformer_blocks.{i}"
        mapping.update(
            {f"{prefix}.{name}": f"{prefix}.{name}" for name in ("norm1.weight", "norm2.weight", "scale1", "scale2")}
        )
        modules.extend(
            [(prefix + ".attn.to_out", prefix + ".attn.to_out.0"), (prefix + ".ff.w2", prefix + ".ff.net.2")]
        )
        for p in ("weight", "bias"):
            trailing = (hidden,) if p == "weight" else ()
            transform = Chain(
                (
                    Reshape((3 * hidden,) + trailing, (heads, 3, head_dim) + trailing),
                    Permute((1, 0, 2, 3) if p == "weight" else (1, 0, 2)),
                    Reshape((3, heads, head_dim) + trailing, (3 * hidden,) + trailing),
                    Split((hidden,) * 3),
                )
            )
            rules.append(
                Rule(
                    (f"{prefix}.attn.to_qkv.{p}",),
                    tuple(f"{prefix}.attn.to_{part}.{p}" for part in ("q", "k", "v")),
                    transform,
                )
            )
            rules.append(Rule((f"{prefix}.ff.w1.{p}",), (f"{prefix}.ff.net.0.proj.{p}",), ReorderChunks((1, 0))))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
