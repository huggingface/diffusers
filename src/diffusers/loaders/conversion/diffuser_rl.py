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


def diffuser_rl_conversion(config):
    if config["layers_per_block"] != 1 or not config["use_timestep_embedding"]:
        raise ValueError("Diffuser temporal checkpoints require two residual blocks per level and timestep embedding.")
    value = config["out_block_type"] == "ValueFunction"
    original_format = config.get("original_format", "diffuser_rl")
    if original_format not in ("diffuser_rl", "diffuser_rl_legacy"):
        raise ValueError("Diffuser original_format must be 'diffuser_rl' or 'diffuser_rl_legacy'.")
    sampler_index = 2 if value or original_format == "diffuser_rl_legacy" else 3
    modules = [("time_mlp.1", "time_mlp.linear_1"), ("time_mlp.3", "time_mlp.linear_2")]
    resnets = []
    channels = config["block_out_channels"]
    previous = config["in_channels"] + config["extra_in_channels"]
    for i, channel in enumerate(channels):
        old = f"{'blocks' if value else 'downs'}.{i}"
        for j in range(2):
            resnets.append((f"{old}.{j}", f"down_blocks.{i}.resnets.{j}", previous != channel))
            previous = channel
        if i < len(channels) - 1 or config["downsample_each_block"]:
            modules.append((f"{old}.{sampler_index}.conv", f"down_blocks.{i}.downsample.conv"))
    if value:
        resnets.extend([("mid_block1", "mid_block.res1", True), ("mid_block2", "mid_block.res2", True)])
        modules.extend(
            [
                ("mid_down1.conv", "mid_block.down1.conv"),
                ("mid_down2.conv", "mid_block.down2.conv"),
                ("final_block.0", "out_block.final_block.0"),
                ("final_block.2", "out_block.final_block.2"),
            ]
        )
    else:
        resnets.extend([("mid_block1", "mid_block.resnets.0", False), ("mid_block2", "mid_block.resnets.1", False)])
        previous = channels[-1]
        for i in range(len(config["up_block_types"])):
            output = channels[-i - 2] if i < len(config["up_block_types"]) - 1 else channels[0]
            resnets.extend(
                [
                    (f"ups.{i}.0", f"up_blocks.{i}.resnets.0", 2 * previous != output),
                    (f"ups.{i}.1", f"up_blocks.{i}.resnets.1", False),
                ]
            )
            if i < len(channels) - 1:
                modules.append((f"ups.{i}.{sampler_index}.conv", f"up_blocks.{i}.upsample.conv"))
            previous = output
        modules.extend(
            [
                ("final_conv.0.block.0", "out_block.final_conv1d_1"),
                ("final_conv.0.block.2", "out_block.final_conv1d_gn"),
                ("final_conv.1", "out_block.final_conv1d_2"),
            ]
        )
    for old, new, shortcut in resnets:
        modules.append((old + ".time_mlp.1", new + ".time_emb"))
        for i, name in enumerate(("conv_in", "conv_out")):
            modules.extend(
                [
                    (f"{old}.blocks.{i}.block.0", f"{new}.{name}.conv1d"),
                    (f"{old}.blocks.{i}.block.2", f"{new}.{name}.group_norm"),
                ]
            )
        if shortcut:
            modules.append((old + ".residual_conv", new + ".residual_conv"))
    return Conversion(mapping={f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
