# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Conversion script for the AudioLDM2 checkpoints."""

import argparse
import re
from typing import List, Union

import torch
import yaml
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    ClapConfig,
    ClapModel,
    GPT2Config,
    GPT2Model,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
    T5Config,
    T5EncoderModel,
)

from diffusers import (
    AudioLDM2Pipeline,
    AudioLDM2ProjectionModel,
    AudioLDM2UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import is_safetensors_available
from diffusers.utils.import_utils import BACKENDS_MAPPING


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.shave_segments
def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_resnet_paths
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


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_vae_resnet_paths
def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_attention_paths
def renew_attention_paths(old_list):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "to_q.weight")
        new_item = new_item.replace("q.bias", "to_q.bias")

        new_item = new_item.replace("k.weight", "to_k.weight")
        new_item = new_item.replace("k.bias", "to_k.bias")

        new_item = new_item.replace("v.weight", "to_v.weight")
        new_item = new_item.replace("v.bias", "to_v.bias")

        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

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

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["to_q.weight", "to_k.weight", "to_v.weight"]
    proj_key = "to_out.0.weight"
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys or ".".join(key.split(".")[-3:]) == proj_key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key].squeeze()


def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a UNet config for diffusers based on the config of the original AudioLDM2 model.
    """
    unet_params = original_config["model"]["params"]["unet_config"]["params"]
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]

    block_out_channels = [unet_params["model_channels"] * mult for mult in unet_params["channel_mult"]]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params["attention_resolutions"] else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params["attention_resolutions"] else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    vae_scale_factor = 2 ** (len(vae_params["ch_mult"]) - 1)

    cross_attention_dim = list(unet_params["context_dim"]) if "context_dim" in unet_params else block_out_channels
    if len(cross_attention_dim) > 1:
        # require two or more cross-attention layers per-block, each of different dimension
        cross_attention_dim = [cross_attention_dim for _ in range(len(block_out_channels))]

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params["in_channels"],
        "out_channels": unet_params["out_channels"],
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params["num_res_blocks"],
        "transformer_layers_per_block": unet_params["transformer_depth"],
        "cross_attention_dim": tuple(cross_attention_dim),
    }

    return config


# Adapted from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_vae_diffusers_config
def create_vae_diffusers_config(original_config, checkpoint, image_size: int):
    """
    Creates a VAE config for diffusers based on the config of the original AudioLDM2 model. Compared to the original
    Stable Diffusion conversion, this function passes a *learnt* VAE scaling factor to the diffusers VAE.
    """
    vae_params = original_config["model"]["params"]["first_stage_config"]["params"]["ddconfig"]
    _ = original_config["model"]["params"]["first_stage_config"]["params"]["embed_dim"]

    block_out_channels = [vae_params["ch"] * mult for mult in vae_params["ch_mult"]]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    scaling_factor = checkpoint["scale_factor"] if "scale_by_std" in original_config["model"]["params"] else 0.18215

    config = {
        "sample_size": image_size,
        "in_channels": vae_params["in_channels"],
        "out_channels": vae_params["out_ch"],
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params["z_channels"],
        "layers_per_block": vae_params["num_res_blocks"],
        "scaling_factor": float(scaling_factor),
    }
    return config


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_diffusers_schedular
def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config["model"]["params"]["timesteps"],
        beta_start=original_config["model"]["params"]["linear_start"],
        beta_end=original_config["model"]["params"]["linear_end"],
        beta_schedule="scaled_linear",
    )
    return schedular


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted UNet checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())

    unet_key = "model.diffusion_model."
    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
        print(f"Checkpoint {path} has both EMA and non-EMA weights.")
        print(
            "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
            " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
        )
        for key in keys:
            if key.startswith("model.diffusion_model"):
                flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
    else:
        if sum(k.startswith("model_ema") for k in keys) > 100:
            print(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )

        # strip the unet prefix from the weight names
        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}." in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}." in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}." in key]
        for layer_id in range(num_output_blocks)
    }

    # Check how many Transformer blocks we have per layer
    if isinstance(config.get("cross_attention_dim"), (list, tuple)):
        if isinstance(config["cross_attention_dim"][0], (list, tuple)):
            # in this case we have multiple cross-attention layers per-block
            num_attention_layers = len(config.get("cross_attention_dim")[0])
    else:
        num_attention_layers = 1

    if config.get("extra_self_attn_layer"):
        num_attention_layers += 1

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.0" not in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = [
                {
                    "old": f"input_blocks.{i}.{1 + layer_id}",
                    "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id * num_attention_layers + layer_id}",
                }
                for layer_id in range(num_attention_layers)
            ]
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=meta_path, config=config
            )

    resnet_0 = middle_blocks[0]
    resnet_1 = middle_blocks[num_middle_blocks - 1]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    meta_path = {"old": "middle_block.0", "new": "mid_block.resnets.0"}
    assign_to_checkpoint(
        resnet_0_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    resnet_1_paths = renew_resnet_paths(resnet_1)
    meta_path = {"old": f"middle_block.{len(middle_blocks) - 1}", "new": "mid_block.resnets.1"}
    assign_to_checkpoint(
        resnet_1_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(1, num_middle_blocks - 1):
        attentions = middle_blocks[i]
        attentions_paths = renew_attention_paths(attentions)
        meta_path = {"old": f"middle_block.{i}", "new": f"mid_block.attentions.{i - 1}"}
        assign_to_checkpoint(
            attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
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
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.0" not in key]

            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                attentions.remove(f"output_blocks.{i}.{index}.conv.bias")
                attentions.remove(f"output_blocks.{i}.{index}.conv.weight")

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = [
                    {
                        "old": f"output_blocks.{i}.{1 + layer_id}",
                        "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id * num_attention_layers + layer_id}",
                    }
                    for layer_id in range(num_attention_layers)
                ]
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=meta_path, config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


CLAP_KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

CLAP_KEYS_TO_IGNORE = [
    "text_transform",
    "audio_transform",
    "stft",
    "logmel_extractor",
    "tscam_conv",
    "head",
    "attn_mask",
]

CLAP_EXPECTED_MISSING_KEYS = ["text_model.embeddings.token_type_ids"]


def convert_open_clap_checkpoint(checkpoint):
    """
    Takes a state dict and returns a converted CLAP checkpoint.
    """
    # extract state dict for CLAP text embedding model, discarding the audio component
    model_state_dict = {}
    model_key = "clap.model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(model_key):
            model_state_dict[key.replace(model_key, "")] = checkpoint.get(key)

    new_checkpoint = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"

    for key, value in model_state_dict.items():
        # check if key should be ignored in mapping - if so map it to a key name that we'll filter out at the end
        for key_to_ignore in CLAP_KEYS_TO_IGNORE:
            if key_to_ignore in key:
                key = "spectrogram"

        # check if any key needs to be modified
        for key_to_modify, new_key in CLAP_KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # replace sequential layers with list
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer) // 3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # Because in CLAP they use `nn.Sequential`...
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            new_checkpoint[key.replace("qkv", "query")] = query_layer
            new_checkpoint[key.replace("qkv", "key")] = key_layer
            new_checkpoint[key.replace("qkv", "value")] = value_layer
        elif key != "spectrogram":
            new_checkpoint[key] = value

    return new_checkpoint


def create_transformers_vocoder_config(original_config):
    """
    Creates a config for transformers SpeechT5HifiGan based on the config of the vocoder model.
    """
    vocoder_params = original_config["model"]["params"]["vocoder_config"]["params"]

    config = {
        "model_in_dim": vocoder_params["num_mels"],
        "sampling_rate": vocoder_params["sampling_rate"],
        "upsample_initial_channel": vocoder_params["upsample_initial_channel"],
        "upsample_rates": list(vocoder_params["upsample_rates"]),
        "upsample_kernel_sizes": list(vocoder_params["upsample_kernel_sizes"]),
        "resblock_kernel_sizes": list(vocoder_params["resblock_kernel_sizes"]),
        "resblock_dilation_sizes": [
            list(resblock_dilation) for resblock_dilation in vocoder_params["resblock_dilation_sizes"]
        ],
        "normalize_before": False,
    }

    return config


def extract_sub_model(checkpoint, key_prefix):
    """
    Takes a state dict and returns the state dict for a particular sub-model.
    """

    sub_model_state_dict = {}
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(key_prefix):
            sub_model_state_dict[key.replace(key_prefix, "")] = checkpoint.get(key)

    return sub_model_state_dict


def convert_hifigan_checkpoint(checkpoint, config):
    """
    Takes a state dict and config, and returns a converted HiFiGAN vocoder checkpoint.
    """
    # extract state dict for vocoder
    vocoder_state_dict = extract_sub_model(checkpoint, key_prefix="first_stage_model.vocoder.")

    # fix upsampler keys, everything else is correct already
    for i in range(len(config.upsample_rates)):
        vocoder_state_dict[f"upsampler.{i}.weight"] = vocoder_state_dict.pop(f"ups.{i}.weight")
        vocoder_state_dict[f"upsampler.{i}.bias"] = vocoder_state_dict.pop(f"ups.{i}.bias")

    if not config.normalize_before:
        # if we don't set normalize_before then these variables are unused, so we set them to their initialised values
        vocoder_state_dict["mean"] = torch.zeros(config.model_in_dim)
        vocoder_state_dict["scale"] = torch.ones(config.model_in_dim)

    return vocoder_state_dict


def convert_projection_checkpoint(checkpoint):
    projection_state_dict = {}
    conditioner_state_dict = extract_sub_model(checkpoint, key_prefix="cond_stage_models.0.")

    projection_state_dict["sos_embed"] = conditioner_state_dict["start_of_sequence_tokens.weight"][0]
    projection_state_dict["sos_embed_1"] = conditioner_state_dict["start_of_sequence_tokens.weight"][1]

    projection_state_dict["eos_embed"] = conditioner_state_dict["end_of_sequence_tokens.weight"][0]
    projection_state_dict["eos_embed_1"] = conditioner_state_dict["end_of_sequence_tokens.weight"][1]

    projection_state_dict["projection.weight"] = conditioner_state_dict["input_sequence_embed_linear.0.weight"]
    projection_state_dict["projection.bias"] = conditioner_state_dict["input_sequence_embed_linear.0.bias"]

    projection_state_dict["projection_1.weight"] = conditioner_state_dict["input_sequence_embed_linear.1.weight"]
    projection_state_dict["projection_1.bias"] = conditioner_state_dict["input_sequence_embed_linear.1.bias"]

    return projection_state_dict


# Adapted from https://github.com/haoheliu/AudioLDM2/blob/81ad2c6ce015c1310387695e2dae975a7d2ed6fd/audioldm2/utils.py#L143
DEFAULT_CONFIG = {
    "model": {
        "params": {
            "linear_start": 0.0015,
            "linear_end": 0.0195,
            "timesteps": 1000,
            "channels": 8,
            "scale_by_std": True,
            "unet_config": {
                "target": "audioldm2.latent_diffusion.openaimodel.UNetModel",
                "params": {
                    "context_dim": [None, 768, 1024],
                    "in_channels": 8,
                    "out_channels": 8,
                    "model_channels": 128,
                    "attention_resolutions": [8, 4, 2],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 3, 5],
                    "num_head_channels": 32,
                    "transformer_depth": 1,
                },
            },
            "first_stage_config": {
                "target": "audioldm2.variational_autoencoder.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 8,
                    "ddconfig": {
                        "z_channels": 8,
                        "resolution": 256,
                        "in_channels": 1,
                        "out_ch": 1,
                        "ch": 128,
                        "ch_mult": [1, 2, 4],
                        "num_res_blocks": 2,
                    },
                },
            },
            "cond_stage_config": {
                "crossattn_audiomae_generated": {
                    "target": "audioldm2.latent_diffusion.modules.encoders.modules.SequenceGenAudioMAECond",
                    "params": {
                        "sequence_gen_length": 8,
                        "sequence_input_embed_dim": [512, 1024],
                    },
                }
            },
            "vocoder_config": {
                "target": "audioldm2.first_stage_model.vocoder",
                "params": {
                    "upsample_rates": [5, 4, 2, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 8, 4, 4],
                    "upsample_initial_channel": 1024,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "num_mels": 64,
                    "sampling_rate": 16000,
                },
            },
        },
    },
}


def load_pipeline_from_original_AudioLDM2_ckpt(
    checkpoint_path: str,
    original_config_file: str = None,
    image_size: int = 1024,
    prediction_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "ddim",
    cross_attention_dim: Union[List, List[List]] = None,
    transformer_layers_per_block: int = None,
    device: str = None,
    from_safetensors: bool = False,
) -> AudioLDM2Pipeline:
    """
    Load an AudioLDM2 pipeline object from a `.ckpt`/`.safetensors` file and (ideally) a `.yaml` config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path (`str`): Path to `.ckpt` file.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            set to the AudioLDM2 base config.
        image_size (`int`, *optional*, defaults to 1024):
            The image size that the model was trained on.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. If `None`, will be automatically
            inferred by looking for a key in the config. For the default config, the prediction type is `'epsilon'`.
        scheduler_type (`str`, *optional*, defaults to 'ddim'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        cross_attention_dim (`list`, *optional*, defaults to `None`):
            The dimension of the cross-attention layers. If `None`, the cross-attention dimension will be
            automatically inferred. Set to `[768, 1024]` for the base model, or `[768, 1024, None]` for the large model.
        transformer_layers_per_block (`int`, *optional*, defaults to `None`):
            The number of transformer layers in each transformer block. If `None`, number of layers will be "
             "automatically inferred. Set to `1` for the base model, or `2` for the large model.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.
        return: An AudioLDM2Pipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """

    if from_safetensors:
        if not is_safetensors_available():
            raise ValueError(BACKENDS_MAPPING["safetensors"][1])

        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if original_config_file is None:
        original_config = DEFAULT_CONFIG
    else:
        original_config = yaml.safe_load(original_config_file)

    if image_size is not None:
        original_config["model"]["params"]["unet_config"]["params"]["image_size"] = image_size

    if cross_attention_dim is not None:
        original_config["model"]["params"]["unet_config"]["params"]["context_dim"] = cross_attention_dim

    if transformer_layers_per_block is not None:
        original_config["model"]["params"]["unet_config"]["params"]["transformer_depth"] = transformer_layers_per_block

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            prediction_type = "v_prediction"
    else:
        if prediction_type is None:
            prediction_type = "epsilon"

    num_train_timesteps = original_config["model"]["params"]["timesteps"]
    beta_start = original_config["model"]["params"]["linear_start"]
    beta_end = original_config["model"]["params"]["linear_end"]

    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # Convert the UNet2DModel
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet = AudioLDM2UNet2DConditionModel(**unet_config)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
    )

    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model
    vae_config = create_vae_diffusers_config(original_config, checkpoint=checkpoint, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the joint audio-text encoding model
    clap_config = ClapConfig.from_pretrained("laion/clap-htsat-unfused")
    clap_config.audio_config.update(
        {
            "patch_embeds_hidden_size": 128,
            "hidden_size": 1024,
            "depths": [2, 2, 12, 2],
        }
    )
    # AudioLDM2 uses the same tokenizer and feature extractor as the original CLAP model
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    clap_feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

    converted_clap_model = convert_open_clap_checkpoint(checkpoint)
    clap_model = ClapModel(clap_config)

    missing_keys, unexpected_keys = clap_model.load_state_dict(converted_clap_model, strict=False)
    # we expect not to have token_type_ids in our original state dict so let's ignore them
    missing_keys = list(set(missing_keys) - set(CLAP_EXPECTED_MISSING_KEYS))

    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys when loading CLAP model: {unexpected_keys}")

    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys when loading CLAP model: {missing_keys}")

    # Convert the vocoder model
    vocoder_config = create_transformers_vocoder_config(original_config)
    vocoder_config = SpeechT5HifiGanConfig(**vocoder_config)
    converted_vocoder_checkpoint = convert_hifigan_checkpoint(checkpoint, vocoder_config)

    vocoder = SpeechT5HifiGan(vocoder_config)
    vocoder.load_state_dict(converted_vocoder_checkpoint)

    # Convert the Flan-T5 encoder model: AudioLDM2 uses the same configuration and tokenizer as the original Flan-T5 large model
    t5_config = T5Config.from_pretrained("google/flan-t5-large")
    converted_t5_checkpoint = extract_sub_model(checkpoint, key_prefix="cond_stage_models.1.model.")

    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    # hard-coded in the original implementation (i.e. not retrievable from the config)
    t5_tokenizer.model_max_length = 128
    t5_model = T5EncoderModel(t5_config)
    t5_model.load_state_dict(converted_t5_checkpoint)

    # Convert the GPT2 encoder model: AudioLDM2 uses the same configuration as the original GPT2 base model
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_model = GPT2Model(gpt2_config)
    gpt2_model.config.max_new_tokens = original_config["model"]["params"]["cond_stage_config"][
        "crossattn_audiomae_generated"
    ]["params"]["sequence_gen_length"]

    converted_gpt2_checkpoint = extract_sub_model(checkpoint, key_prefix="cond_stage_models.0.model.")
    gpt2_model.load_state_dict(converted_gpt2_checkpoint)

    # Convert the extra embedding / projection layers
    projection_model = AudioLDM2ProjectionModel(clap_config.projection_dim, t5_config.d_model, gpt2_config.n_embd)

    converted_projection_checkpoint = convert_projection_checkpoint(checkpoint)
    projection_model.load_state_dict(converted_projection_checkpoint)

    # Instantiate the diffusers pipeline
    pipe = AudioLDM2Pipeline(
        vae=vae,
        text_encoder=clap_model,
        text_encoder_2=t5_model,
        projection_model=projection_model,
        language_model=gpt2_model,
        tokenizer=clap_tokenizer,
        tokenizer_2=t5_tokenizer,
        feature_extractor=clap_feature_extractor,
        unet=unet,
        scheduler=scheduler,
        vocoder=vocoder,
    )

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--cross_attention_dim",
        default=None,
        type=int,
        nargs="+",
        help="The dimension of the cross-attention layers. If `None`, the cross-attention dimension will be "
        "automatically inferred. Set to `768+1024` for the base model, or `768+1024+640` for the large model",
    )
    parser.add_argument(
        "--transformer_layers_per_block",
        default=None,
        type=int,
        help="The number of transformer layers in each transformer block. If `None`, number of layers will be "
        "automatically inferred. Set to `1` for the base model, or `2` for the large model.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="ddim",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--image_size",
        default=1048,
        type=int,
        help="The image size that the model was trained on.",
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help=("The prediction type that the model was trained on."),
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--from_safetensors",
        action="store_true",
        help="If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    args = parser.parse_args()

    pipe = load_pipeline_from_original_AudioLDM2_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        cross_attention_dim=args.cross_attention_dim,
        transformer_layers_per_block=args.transformer_layers_per_block,
        from_safetensors=args.from_safetensors,
        device=args.device,
    )
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
