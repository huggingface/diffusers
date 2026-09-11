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
from .transforms import Split


def mochi_vae_conversion(config):
    modules = [
        ("decoder.blocks.0.0", "decoder.conv_in"),
        ("decoder.output_proj", "decoder.proj_out"),
        ("encoder.layers.0", "encoder.proj_in"),
        ("encoder.output_norm", "encoder.norm_out.norm_layer"),
    ]
    mapping = {"encoder.output_proj.weight": "encoder.proj_out.weight"}
    resnets, attentions = [], []
    layers = config["layers_per_block"]
    decoder_count = len(config["decoder_block_out_channels"]) - 1
    resnets.extend((f"decoder.blocks.0.{i + 1}", f"decoder.block_in.resnets.{i}") for i in range(layers[-1]))
    for i in range(decoder_count):
        resnets.extend(
            (f"decoder.blocks.{i + 1}.blocks.{j}", f"decoder.up_blocks.{i}.resnets.{j}") for j in range(layers[-i - 2])
        )
        modules.append((f"decoder.blocks.{i + 1}.proj", f"decoder.up_blocks.{i}.proj"))
    resnets.extend(
        (f"decoder.blocks.{decoder_count + 1}.{i}", f"decoder.block_out.resnets.{i}") for i in range(layers[0])
    )
    for i in range(layers[0]):
        old = f"encoder.layers.{i + 1}"
        resnets.append((old, f"encoder.block_in.resnets.{i}"))
        if config["add_attention_block"][0]:
            attentions.append((old + ".attn_block", "encoder.block_in", i, config["encoder_block_out_channels"][0]))
    encoder_count = len(config["encoder_block_out_channels"]) - 1
    offset = 1 + layers[0]
    for i in range(encoder_count):
        old, new = f"encoder.layers.{offset + i}", f"encoder.down_blocks.{i}"
        modules.append((old + ".layers.0", new + ".conv_in.conv"))
        for j in range(layers[i + 1]):
            resnets.append((f"{old}.layers.{j + 1}", f"{new}.resnets.{j}"))
            if config["add_attention_block"][i + 1]:
                attentions.append(
                    (f"{old}.layers.{j + 1}.attn_block", new, j, config["encoder_block_out_channels"][i + 1])
                )
    for i in range(layers[-1]):
        old = f"encoder.layers.{offset + encoder_count + i}"
        resnets.append((old, f"encoder.block_out.resnets.{i}"))
        if config["add_attention_block"][-1]:
            attentions.append((old + ".attn_block", "encoder.block_out", i, config["encoder_block_out_channels"][-1]))
    for old, new in resnets:
        modules.extend(
            (f"{old}.stack.{i}", f"{new}.{name}")
            for i, name in ((0, "norm1.norm_layer"), (2, "conv1.conv"), (3, "norm2.norm_layer"), (5, "conv2.conv"))
        )
    rules = []
    for old, new, index, hidden in attentions:
        modules.extend(
            [
                (old + ".norm", f"{new}.norms.{index}.norm_layer"),
                (old + ".attn.out", f"{new}.attentions.{index}.to_out.0"),
            ]
        )
        rules.append(
            Rule(
                (old + ".attn.qkv.weight",),
                tuple(f"{new}.attentions.{index}.to_{part}.weight" for part in ("q", "k", "v")),
                Split((hidden,) * 3),
            )
        )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
