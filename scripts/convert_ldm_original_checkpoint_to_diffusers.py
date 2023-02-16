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
""" Conversion script for the LDM checkpoints. """

import argparse
import json

import torch

from diffusers import DDPMScheduler, LDMPipeline, UNet2DModel, VQModel


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
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

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def convert_ldm_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = checkpoint["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = checkpoint["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = checkpoint["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = checkpoint["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = checkpoint["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = checkpoint["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = checkpoint["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = checkpoint["out.0.bias"]
    new_checkpoint["conv_out.weight"] = checkpoint["out.2.weight"]
    new_checkpoint["conv_out.bias"] = checkpoint["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in checkpoint if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in checkpoint if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in checkpoint if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in checkpoint if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in checkpoint if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in checkpoint if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["num_res_blocks"] + 1)
        layer_in_block_id = (i - 1) % (config["num_res_blocks"] + 1)

        resnets = [key for key in input_blocks[i] if f"input_blocks.{i}.0" in key]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in checkpoint:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = checkpoint[
                f"input_blocks.{i}.0.op.weight"
            ]
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = checkpoint[
                f"input_blocks.{i}.0.op.bias"
            ]
            continue

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        resnet_op = {"old": "resnets.2.op", "new": "downsamplers.0.op"}
        assign_to_checkpoint(
            paths, new_checkpoint, checkpoint, additional_replacements=[meta_path, resnet_op], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {
                "old": f"input_blocks.{i}.1",
                "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}",
            }
            to_split = {
                f"input_blocks.{i}.1.qkv.bias": {
                    "key": f"down_blocks.{block_id}.attentions.{layer_in_block_id}.key.bias",
                    "query": f"down_blocks.{block_id}.attentions.{layer_in_block_id}.query.bias",
                    "value": f"down_blocks.{block_id}.attentions.{layer_in_block_id}.value.bias",
                },
                f"input_blocks.{i}.1.qkv.weight": {
                    "key": f"down_blocks.{block_id}.attentions.{layer_in_block_id}.key.weight",
                    "query": f"down_blocks.{block_id}.attentions.{layer_in_block_id}.query.weight",
                    "value": f"down_blocks.{block_id}.attentions.{layer_in_block_id}.value.weight",
                },
            }
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                checkpoint,
                additional_replacements=[meta_path],
                attention_paths_to_split=to_split,
                config=config,
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, checkpoint, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, checkpoint, config=config)

    attentions_paths = renew_attention_paths(attentions)
    to_split = {
        "middle_block.1.qkv.bias": {
            "key": "mid_block.attentions.0.key.bias",
            "query": "mid_block.attentions.0.query.bias",
            "value": "mid_block.attentions.0.value.bias",
        },
        "middle_block.1.qkv.weight": {
            "key": "mid_block.attentions.0.key.weight",
            "query": "mid_block.attentions.0.query.weight",
            "value": "mid_block.attentions.0.value.weight",
        },
    }
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, checkpoint, attention_paths_to_split=to_split, config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["num_res_blocks"] + 1)
        layer_in_block_id = i % (config["num_res_blocks"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[meta_path], config=config)

            if ["conv.weight", "conv.bias"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = checkpoint[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = checkpoint[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                to_split = {
                    f"output_blocks.{i}.1.qkv.bias": {
                        "key": f"up_blocks.{block_id}.attentions.{layer_in_block_id}.key.bias",
                        "query": f"up_blocks.{block_id}.attentions.{layer_in_block_id}.query.bias",
                        "value": f"up_blocks.{block_id}.attentions.{layer_in_block_id}.value.bias",
                    },
                    f"output_blocks.{i}.1.qkv.weight": {
                        "key": f"up_blocks.{block_id}.attentions.{layer_in_block_id}.key.weight",
                        "query": f"up_blocks.{block_id}.attentions.{layer_in_block_id}.query.weight",
                        "value": f"up_blocks.{block_id}.attentions.{layer_in_block_id}.value.weight",
                    },
                }
                assign_to_checkpoint(
                    paths,
                    new_checkpoint,
                    checkpoint,
                    additional_replacements=[meta_path],
                    attention_paths_to_split=to_split if any("qkv" in key for key in attentions) else None,
                    config=config,
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = checkpoint[old_path]

    return new_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)

    with open(args.config_file) as f:
        config = json.loads(f.read())

    converted_checkpoint = convert_ldm_checkpoint(checkpoint, config)

    if "ldm" in config:
        del config["ldm"]

    model = UNet2DModel(**config)
    model.load_state_dict(converted_checkpoint)

    try:
        scheduler = DDPMScheduler.from_config("/".join(args.checkpoint_path.split("/")[:-1]))
        vqvae = VQModel.from_pretrained("/".join(args.checkpoint_path.split("/")[:-1]))

        pipe = LDMPipeline(unet=model, scheduler=scheduler, vae=vqvae)
        pipe.save_pretrained(args.dump_path)
    except:  # noqa: E722
        model.save_pretrained(args.dump_path)
