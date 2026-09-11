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


def kandinsky3_conversion(config):
    modules = [
        ("to_time_embed.1", "time_embedding.linear_1"),
        ("to_time_embed.3", "time_embedding.linear_2"),
        ("in_layer", "conv_in"),
        ("out_layer.0", "conv_norm_out"),
        ("out_layer.2", "conv_out"),
        ("projection_ln", "encoder_hid_proj.projection_norm"),
    ]
    weights = [("projection_lin", "encoder_hid_proj.projection_linear")]
    attentions = [("feature_pooling", "add_time_condition", False)]
    resnets = []
    channels = [config["block_out_channels"][0] // 2] + list(config["block_out_channels"])
    count, layers = len(channels) - 1, config["layers_per_block"]
    flags = (False, True, True, True)
    for i in range(count):
        old, new = f"down_samples.{i}", f"down_blocks.{i}"
        if flags[i]:
            attentions.append((old + ".self_attention_block", new + ".attentions.0", True))
        for j in range(layers):
            input_channel = channels[i] if j == 0 else channels[i + 1]
            resnets.append(
                (
                    f"{old}.resnet_attn_blocks.{j}.0",
                    f"{new}.resnets_in.{j}",
                    input_channel != channels[i + 1],
                    False,
                    False,
                )
            )
            resnets.append(
                (
                    f"{old}.resnet_attn_blocks.{j}.2",
                    f"{new}.resnets_out.{j}",
                    False,
                    False,
                    j == layers - 1 and i < count - 1,
                )
            )
            if flags[i]:
                attentions.append((f"{old}.resnet_attn_blocks.{j}.1", f"{new}.attentions.{j + 1}", True))
    for i in range(count):
        level = count - 1 - i
        input_channel, output_channel = channels[level + 1], channels[level]
        cat = 0 if i == 0 else input_channel
        old, new = f"up_samples.{i}", f"up_blocks.{i}"
        if flags[-i - 1]:
            attentions.append((old + ".self_attention_block", new + ".attentions.0", True))
        pairs = (
            [(input_channel + cat, input_channel)]
            + [(input_channel, input_channel)] * (layers - 2)
            + [(input_channel, output_channel)]
        )
        for j, (a, b) in zip(range(layers), pairs):
            resnets.append(
                (f"{old}.resnet_attn_blocks.{j}.0", f"{new}.resnets_in.{j}", False, j == 0 and i > 0, False)
            )
            resnets.append((f"{old}.resnet_attn_blocks.{j}.2", f"{new}.resnets_out.{j}", a != b, False, False))
            if flags[-i - 1]:
                attentions.append((f"{old}.resnet_attn_blocks.{j}.1", f"{new}.attentions.{j + 1}", True))
    for old, new, shortcut, up, down in resnets:
        for i in range(4):
            modules.extend(
                [
                    (
                        f"{old}.resnet_blocks.{i}.group_norm.context_mlp.1",
                        f"{new}.resnet_blocks.{i}.group_norm.context_mlp.1",
                    ),
                    (f"{old}.resnet_blocks.{i}.projection", f"{new}.resnet_blocks.{i}.projection"),
                ]
            )
        if shortcut:
            modules.append((old + ".shortcut_projection", new + ".shortcut_projection"))
        for active, index, direction in ((up, 1, "up"), (down, 2, "down")):
            if active:
                modules.extend(
                    [
                        (
                            f"{old}.resnet_blocks.{index}.{direction}_sample",
                            f"{new}.resnet_blocks.{index}.{direction}_sample",
                        ),
                        (f"{old}.shortcut_{direction}_sample", f"{new}.shortcut_{direction}_sample"),
                    ]
                )
    for old, new, conditioned in attentions:
        weights.extend(
            (f"{old}.attention.{a}", f"{new}.attention.{b}")
            for a, b in (("to_query", "to_q"), ("to_key", "to_k"), ("to_value", "to_v"), ("output_layer", "to_out.0"))
        )
        if conditioned:
            modules.extend(
                (f"{old}.{name}.context_mlp.1", f"{new}.{name}.context_mlp.1") for name in ("in_norm", "out_norm")
            )
            weights.extend((f"{old}.feed_forward.{i}", f"{new}.feed_forward.{i}") for i in (0, 2))
    mapping = {old + ".weight": new + ".weight" for old, new in weights}
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
