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
from distutils.command import check
import json
import torch
from diffusers import UNetUnconditionalModel 

def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return '.'.join(path.split('.')[n_shave_prefix_segments:])
    else:
        return '.'.join(path.split('.')[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace('in_layers.0', 'norm1')
        new_item = new_item.replace('in_layers.2', 'conv1')

        new_item = new_item.replace('out_layers.0', 'norm2')
        new_item = new_item.replace('out_layers.3', 'conv2')

        new_item = new_item.replace('emb_layers.1', 'time_emb_proj')
        new_item = new_item.replace('skip_connection', 'conv_shortcut')

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({'old': old_item, 'new': new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace('norm.weight', 'group_norm.weight')
        new_item = new_item.replace('norm.bias', 'group_norm.bias')

        new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({'old': old_item, 'new': new_item})

    return mapping


def assign_to_checkpoint(paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map['query']] = query.reshape(target_shape)
            checkpoint[path_map['key']] = key.reshape(target_shape)
            checkpoint[path_map['value']] = value.reshape(target_shape)

    for path in paths:
        new_path = path['new']

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace('middle_block.0', 'mid.resnets.0')
        new_path = new_path.replace('middle_block.1', 'mid.attentions.0')
        new_path = new_path.replace('middle_block.2', 'mid.resnets.1')

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement['old'], replacement['new'])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path['old']][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path['old']]


def convert_ncsnpp_checkpoint(checkpoint, config):
    """
    Takes a state dict and the path to
    """
    new_model_architecture  = UNetUnconditionalModel(**config)
    new_model_architecture.time_steps.W.data= checkpoint['all_modules.0.W'].data
    new_model_architecture.time_steps.weight.data = checkpoint['all_modules.0.W'].data
    new_model_architecture.time_embedding.linear_1.weight.data = checkpoint['all_modules.1.weight'].data
    new_model_architecture.time_embedding.linear_1.bias.data = checkpoint['all_modules.1.bias'].data
    
    new_model_architecture.time_embedding.linear_2.weight.data = checkpoint['all_modules.2.weight'].data
    new_model_architecture.time_embedding.linear_2.bias.data= checkpoint['all_modules.2.bias'].data

    new_model_architecture.conv_in.weight.data = checkpoint['all_modules.3.weight'].data
    new_model_architecture.conv_in.bias.data = checkpoint['all_modules.3.bias'].data

    new_model_architecture.conv_norm_out.weight.data = checkpoint[list(checkpoint.keys())[-4]].data
    new_model_architecture.conv_norm_out.bias.data = checkpoint[list(checkpoint.keys())[-3]].data
    new_model_architecture.conv_out.weight.data = checkpoint[list(checkpoint.keys())[-2]].data
    new_model_architecture.conv_out.bias.data = checkpoint[list(checkpoint.keys())[-1]].data

    module_index = 4


    def set_attention_weights(new_layer,old_checkpoint,index):
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

    def set_resnet_weights(new_layer,old_checkpoint,index):
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
            set_resnet_weights(block.resnets[j],checkpoint, module_index)
            module_index += 1
            if has_attentions:
                set_attention_weights(block.attentions[j],checkpoint, module_index)
                module_index += 1
                
        if hasattr(block, "downsamplers") and block.downsamplers is not None:
            set_resnet_weights(block.resnet_down,checkpoint, module_index)
            module_index += 1
            block.skip_conv.weight.data = checkpoint[f"all_modules.{module_index}.Conv_0.weight"].data
            block.skip_conv.bias.data = checkpoint[f"all_modules.{module_index}.Conv_0.bias"].data
            module_index += 1
        

                    
    set_resnet_weights(new_model_architecture.mid.resnets[0],checkpoint,module_index)
    module_index += 1
    set_attention_weights(new_model_architecture.mid.attentions[0],checkpoint, module_index)
    module_index += 1
    set_resnet_weights(new_model_architecture.mid.resnets[1],checkpoint,module_index)
    module_index += 1

    for i, block in enumerate(new_model_architecture.upsample_blocks):
        has_attentions = hasattr(block, "attentions")
        for j in range(len(block.resnets)):
            set_resnet_weights(block.resnets[j],checkpoint, module_index)
            module_index += 1
        if has_attentions:
            set_attention_weights(block.attentions[0],checkpoint, module_index) # why can there only be a single attention layer for up? 
            module_index += 1
            
        if hasattr(block, "resnet_up") and block.resnet_up is not None:
            block.skip_norm.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
            block.skip_norm.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
            module_index += 1
            block.skip_conv.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
            block.skip_conv.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
            module_index += 1
            set_resnet_weights(block.resnet_up,checkpoint, module_index)
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
        "--checkpoint_path", default="/Users/arthurzucker/Work/diffusers/ArthurZ/diffusion_model.pt", type=str, required=False, help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--config_file",
        default="/Users/arthurzucker/Work/diffusers/ArthurZ/config.json",
        type=str,
        required=False,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument(
        "--dump_path", default="/Users/arthurzucker/Work/diffusers/ArthurZ/diffusion_model_new.pt", type=str, required=False, help="Path to the output model."
    )

    args = parser.parse_args()



            
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    with open(args.config_file) as f:
        config = json.loads(f.read())


    converted_checkpoint = convert_ncsnpp_checkpoint(checkpoint, config,)
    torch.save(converted_checkpoint, args.dump_path)
