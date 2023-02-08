# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
""" Conversion script for the NCSNPP checkpoints. """

import argparse
import json

import torch

from diffusers import ScoreSdeVePipeline, ScoreSdeVeScheduler, UNet2DModel


def convert_ncsnpp_checkpoint(checkpoint, config):
    """
    Takes a state dict and the path to
    """
    new_model_architecture = UNet2DModel(**config)
    new_model_architecture.time_proj.W.data = checkpoint["all_modules.0.W"].data
    new_model_architecture.time_proj.weight.data = checkpoint["all_modules.0.W"].data
    new_model_architecture.time_embedding.linear_1.weight.data = checkpoint["all_modules.1.weight"].data
    new_model_architecture.time_embedding.linear_1.bias.data = checkpoint["all_modules.1.bias"].data

    new_model_architecture.time_embedding.linear_2.weight.data = checkpoint["all_modules.2.weight"].data
    new_model_architecture.time_embedding.linear_2.bias.data = checkpoint["all_modules.2.bias"].data

    new_model_architecture.conv_in.weight.data = checkpoint["all_modules.3.weight"].data
    new_model_architecture.conv_in.bias.data = checkpoint["all_modules.3.bias"].data

    new_model_architecture.conv_norm_out.weight.data = checkpoint[list(checkpoint.keys())[-4]].data
    new_model_architecture.conv_norm_out.bias.data = checkpoint[list(checkpoint.keys())[-3]].data
    new_model_architecture.conv_out.weight.data = checkpoint[list(checkpoint.keys())[-2]].data
    new_model_architecture.conv_out.bias.data = checkpoint[list(checkpoint.keys())[-1]].data

    module_index = 4

    def set_attention_weights(new_layer, old_checkpoint, index):
        new_layer.query.weight.data = old_checkpoint[f"all_modules.{index}.NIN_0.W"].data.T
        new_layer.key.weight.data = old_checkpoint[f"all_modules.{index}.NIN_1.W"].data.T
        new_layer.value.weight.data = old_checkpoint[f"all_modules.{index}.NIN_2.W"].data.T

        new_layer.query.bias.data = old_checkpoint[f"all_modules.{index}.NIN_0.b"].data
        new_layer.key.bias.data = old_checkpoint[f"all_modules.{index}.NIN_1.b"].data
        new_layer.value.bias.data = old_checkpoint[f"all_modules.{index}.NIN_2.b"].data

        new_layer.proj_attn.weight.data = old_checkpoint[f"all_modules.{index}.NIN_3.W"].data.T
        new_layer.proj_attn.bias.data = old_checkpoint[f"all_modules.{index}.NIN_3.b"].data

        new_layer.group_norm.weight.data = old_checkpoint[f"all_modules.{index}.GroupNorm_0.weight"].data
        new_layer.group_norm.bias.data = old_checkpoint[f"all_modules.{index}.GroupNorm_0.bias"].data

    def set_resnet_weights(new_layer, old_checkpoint, index):
        new_layer.conv1.weight.data = old_checkpoint[f"all_modules.{index}.Conv_0.weight"].data
        new_layer.conv1.bias.data = old_checkpoint[f"all_modules.{index}.Conv_0.bias"].data
        new_layer.norm1.weight.data = old_checkpoint[f"all_modules.{index}.GroupNorm_0.weight"].data
        new_layer.norm1.bias.data = old_checkpoint[f"all_modules.{index}.GroupNorm_0.bias"].data

        new_layer.conv2.weight.data = old_checkpoint[f"all_modules.{index}.Conv_1.weight"].data
        new_layer.conv2.bias.data = old_checkpoint[f"all_modules.{index}.Conv_1.bias"].data
        new_layer.norm2.weight.data = old_checkpoint[f"all_modules.{index}.GroupNorm_1.weight"].data
        new_layer.norm2.bias.data = old_checkpoint[f"all_modules.{index}.GroupNorm_1.bias"].data

        new_layer.time_emb_proj.weight.data = old_checkpoint[f"all_modules.{index}.Dense_0.weight"].data
        new_layer.time_emb_proj.bias.data = old_checkpoint[f"all_modules.{index}.Dense_0.bias"].data

        if new_layer.in_channels != new_layer.out_channels or new_layer.up or new_layer.down:
            new_layer.conv_shortcut.weight.data = old_checkpoint[f"all_modules.{index}.Conv_2.weight"].data
            new_layer.conv_shortcut.bias.data = old_checkpoint[f"all_modules.{index}.Conv_2.bias"].data

    for i, block in enumerate(new_model_architecture.downsample_blocks):
        has_attentions = hasattr(block, "attentions")
        for j in range(len(block.resnets)):
            set_resnet_weights(block.resnets[j], checkpoint, module_index)
            module_index += 1
            if has_attentions:
                set_attention_weights(block.attentions[j], checkpoint, module_index)
                module_index += 1

        if hasattr(block, "downsamplers") and block.downsamplers is not None:
            set_resnet_weights(block.resnet_down, checkpoint, module_index)
            module_index += 1
            block.skip_conv.weight.data = checkpoint[f"all_modules.{module_index}.Conv_0.weight"].data
            block.skip_conv.bias.data = checkpoint[f"all_modules.{module_index}.Conv_0.bias"].data
            module_index += 1

    set_resnet_weights(new_model_architecture.mid_block.resnets[0], checkpoint, module_index)
    module_index += 1
    set_attention_weights(new_model_architecture.mid_block.attentions[0], checkpoint, module_index)
    module_index += 1
    set_resnet_weights(new_model_architecture.mid_block.resnets[1], checkpoint, module_index)
    module_index += 1

    for i, block in enumerate(new_model_architecture.up_blocks):
        has_attentions = hasattr(block, "attentions")
        for j in range(len(block.resnets)):
            set_resnet_weights(block.resnets[j], checkpoint, module_index)
            module_index += 1
        if has_attentions:
            set_attention_weights(
                block.attentions[0], checkpoint, module_index
            )  # why can there only be a single attention layer for up?
            module_index += 1

        if hasattr(block, "resnet_up") and block.resnet_up is not None:
            block.skip_norm.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
            block.skip_norm.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
            module_index += 1
            block.skip_conv.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
            block.skip_conv.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
            module_index += 1
            set_resnet_weights(block.resnet_up, checkpoint, module_index)
            module_index += 1

    new_model_architecture.conv_norm_out.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
    new_model_architecture.conv_norm_out.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
    module_index += 1
    new_model_architecture.conv_out.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
    new_model_architecture.conv_out.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data

    return new_model_architecture.state_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default="/Users/arthurzucker/Work/diffusers/ArthurZ/diffusion_pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the checkpoint to convert.",
    )

    parser.add_argument(
        "--config_file",
        default="/Users/arthurzucker/Work/diffusers/ArthurZ/config.json",
        type=str,
        required=False,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument(
        "--dump_path",
        default="/Users/arthurzucker/Work/diffusers/ArthurZ/diffusion_model_new.pt",
        type=str,
        required=False,
        help="Path to the output model.",
    )

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    with open(args.config_file) as f:
        config = json.loads(f.read())

    converted_checkpoint = convert_ncsnpp_checkpoint(
        checkpoint,
        config,
    )

    if "sde" in config:
        del config["sde"]

    model = UNet2DModel(**config)
    model.load_state_dict(converted_checkpoint)

    try:
        scheduler = ScoreSdeVeScheduler.from_config("/".join(args.checkpoint_path.split("/")[:-1]))

        pipe = ScoreSdeVePipeline(unet=model, scheduler=scheduler)
        pipe.save_pretrained(args.dump_path)
    except:  # noqa: E722
        model.save_pretrained(args.dump_path)
