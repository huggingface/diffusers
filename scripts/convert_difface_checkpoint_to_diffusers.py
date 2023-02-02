import argparse
import json

import torch
from copy import deepcopy
from collections import OrderedDict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / 'src'))
from diffusers import AutoencoderKL, ImprovedDDPMScheduler, UNet2DModel, VQModel, DifFacePipeline


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = new_item.replace("block.", "resnets.")
        new_item = new_item.replace("conv_shorcut", "conv1")
        new_item = new_item.replace("in_shortcut", "conv_shortcut")
        new_item = new_item.replace("temb_proj", "time_emb_proj")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0, in_mid=False):
    mapping = []
    for old_item in old_list:
        new_item = old_item

        # In `model.mid`, the layer is called `attn`.
        if not in_mid:
            new_item = new_item.replace("attn", "attentions")
        new_item = new_item.replace(".k.", ".key.")
        new_item = new_item.replace(".v.", ".value.")
        new_item = new_item.replace(".q.", ".query.")

        new_item = new_item.replace("proj_out", "proj_attn")
        new_item = new_item.replace("norm", "group_norm")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)
        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    if attention_paths_to_split is not None:
        if config is None:
            raise ValueError("Please specify the config if setting 'attention_paths_to_split' to 'True'.")

        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config.get("num_head_channels", 1) // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape).squeeze()
            checkpoint[path_map["key"]] = key.reshape(target_shape).squeeze()
            checkpoint[path_map["value"]] = value.reshape(target_shape).squeeze()

    for path in paths:
        new_path = path["new"]

        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        new_path = new_path.replace("down.", "down_blocks.")
        new_path = new_path.replace("up.", "up_blocks.")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        if "attentions" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]].squeeze()
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]

def split_qkv_weight(old_weight, config):
    chn = old_weight.shape[0] // 3
    target_shape = (-1, chn) if len(old_weight.shape) == 3 else (-1)

    num_heads = old_weight.shape[0] // config.get("attention_head_dim", 1) // 3

    old_weight = old_weight.reshape((num_heads, 3 * chn // num_heads) + old_weight.shape[1:])
    query, key, value = old_weight.split(chn // num_heads, dim=1)

    query = query.reshape(target_shape).squeeze()
    key = key.reshape(target_shape).squeeze()
    value = value.reshape(target_shape).squeeze()

    return query, key, value

def copy_weight(target_weight, source_weight):
    assert target_weight.shape == source_weight.shape or target_weight.numel() == source_weight.numel()
    target_weight.copy_(source_weight.view(target_weight.shape))

class ConvertDifFace:
    def __init__(self, target_checkpoint, source_checkpoint, config):
        self.num_copy = 0
        self.converted_checkpoint = OrderedDict()

        self.config = config
        self.source_checkpoint = source_checkpoint
        self.target_checkpoint = target_checkpoint     # only for checking dimension
        self.num_weight = len([key for key in source_checkpoint.keys()])
        self.num_down_blocks = len(
                [key for key in checkpoint.keys() if ("op." in key and 'input_blocks' in key)]
                ) // 2 + 1
        self.num_up_blocks = self.num_down_blocks

    def update_step(self, target_key, source_key=None, add_copy=True, source_value=None):
        if source_value is None:
            assert source_key is not None
            source_value = self.source_checkpoint[source_key]

        assert source_value.squeeze().shape ==  self.target_checkpoint[target_key].squeeze().shape
        target_shape = self.target_checkpoint[target_key].shape
        self.converted_checkpoint[target_key] = source_value.view(target_shape)
        self.target_checkpoint.pop(target_key)
        if add_copy:
            self.source_checkpoint.pop(source_key)
            self.num_copy += 1

    def check_convert(self):
        assert self.num_copy == self.num_weight
        assert len([key for key in self.source_checkpoint.keys()]) == 0
        assert len([key for key in self.target_checkpoint.keys()]) == 0

    def convert(self):
        self.convert_head()
        self.convert_downblock()
        self.convert_middle()
        self.convert_upblock()
        self.convert_tail()

        self.check_convert()

        return self.converted_checkpoint

    def convert_head(self):
        self.update_step("time_embedding.linear_1.weight", "time_embed.0.weight")
        self.update_step("time_embedding.linear_1.bias", "time_embed.0.bias")
        self.update_step("time_embedding.linear_2.weight", "time_embed.2.weight")
        self.update_step("time_embedding.linear_2.bias", "time_embed.2.bias")
        self.update_step("conv_in.weight", "input_blocks.0.0.weight")
        self.update_step("conv_in.bias", "input_blocks.0.0.bias")

    def convert_tail(self):
        self.update_step("conv_norm_out.weight", "out.0.weight")
        self.update_step("conv_norm_out.bias", "out.0.bias")
        self.update_step("conv_out.weight", "out.2.weight")
        self.update_step("conv_out.bias", "out.2.bias")

    def convert_downblock(self):
        block_id_source = 1
        for i in range(self.num_down_blocks):
            for j in range(self.config["layers_per_block"][i]):
                # resblock
                self.convert_resblock(i, block_id_source, j, ['down', 'in'])
                for mode in ['weight', 'bias']:
                    # downsample
                    if j+1 == self.config["layers_per_block"][i] and i + 1 < self.num_down_blocks:
                        self.update_step(
                            f"down_blocks.{i}.downsamplers.0.conv.{mode}",
                            f"input_blocks.{block_id_source+j+1}.0.op.{mode}",
                                )
                        if mode == 'bias':
                            block_id_source += 1

                print(f"Down Block: {i+1}, Level: {j+1}," + \
                      f" Att: {str(self.attn_flag):5s}," + \
                      f" Skip Conv: {str(self.skip_conv_flag):5s}")

            block_id_source += self.config["layers_per_block"][i]

    def convert_upblock(self):
        block_id_source = 0
        for i in range(self.num_up_blocks):
            for j in range(list(reversed(self.config["layers_per_block"]))[i]+1):
                self.convert_resblock(i, block_id_source, j, ['up', 'out'])
                for mode in ['weight', 'bias']:
                    # upsample
                    if j == list(reversed(self.config["layers_per_block"]))[i] and i + 1 < self.num_up_blocks:
                        up_index = '2' if self.attn_flag else '1'
                        self.update_step(
                            f"up_blocks.{i}.upsamplers.0.conv.{mode}",
                            f"output_blocks.{block_id_source+j}.{up_index}.conv.{mode}",
                                )

                print(f"Up Block: {i+1}, Level: {j+1}," + \
                      f" Att: {str(self.attn_flag):5s}," + \
                      f" Skip Conv: {str(self.skip_conv_flag):5s}")

            block_id_source += list(reversed(self.config["layers_per_block"]))[i] + 1

    def convert_middle(self):
        block_id_source = 0
        for i in range(2):
            for mode in ['weight', 'bias']:
                # norm1
                self.update_step(
                    f"mid_block.resnets.{i}.norm1.{mode}",
                    f"middle_block.{block_id_source}.in_layers.0.{mode}"
                        )
                # conv1
                self.update_step(
                    f"mid_block.resnets.{i}.conv1.{mode}",
                    f"middle_block.{block_id_source}.in_layers.2.{mode}"
                        )
                # embedding
                self.update_step(
                    f"mid_block.resnets.{i}.time_emb_proj.{mode}",
                    f"middle_block.{block_id_source}.emb_layers.1.{mode}"
                        )
                # norm2
                self.update_step(
                    f"mid_block.resnets.{i}.norm2.{mode}",
                    f"middle_block.{block_id_source}.out_layers.0.{mode}"
                        )
                # conv2
                self.update_step(
                    f"mid_block.resnets.{i}.conv2.{mode}",
                    f"middle_block.{block_id_source}.out_layers.3.{mode}"
                        )

                # attention: norm layer
                if i == 0:
                    attn_flag = False
                    # norm layer
                    self.update_step(
                        f"mid_block.attentions.{i}.group_norm.{mode}",
                        f"middle_block.{block_id_source+1}.norm.{mode}"
                            )
                    # linear layer
                    self.update_step(
                        f"mid_block.attentions.{i}.proj_attn.{mode}",
                        f"middle_block.{block_id_source+1}.proj_out.{mode}"
                            )
                    # qkv
                    query, key, value = split_qkv_weight(
                                self.source_checkpoint[f"middle_block.{block_id_source+1}.qkv.{mode}"],
                                self.config,
                                )
                    self.update_step(
                        f"mid_block.attentions.{i}.query.{mode}",
                        source_value=query,
                        add_copy=False,
                            )
                    self.update_step(
                        f"mid_block.attentions.{i}.key.{mode}",
                        source_value=key,
                        add_copy=False,
                            )
                    self.update_step(
                        f"mid_block.attentions.{i}.value.{mode}",
                        source_value=value,
                        add_copy=False,
                            )
                    self.source_checkpoint.pop(f"middle_block.{block_id_source+1}.qkv.{mode}")
                    self.num_copy += 1
                    attn_flag = True
                    if mode == 'bias':
                        block_id_source += 1

                print(f"Mid Block: {i+1}, {mode}, Att: {str(attn_flag):5s}")

            block_id_source += 1

    def convert_resblock(self, block_id_target, block_id_source, level_id, prefix=['down', 'in']):
        updown, inout = prefix
        for mode in ['weight', 'bias']:
            # norm1
            self.update_step(
                f"{updown}_blocks.{block_id_target}.resnets.{level_id}.norm1.{mode}",
                f"{inout}put_blocks.{block_id_source+level_id}.0.in_layers.0.{mode}"
                    )
            # conv1
            self.update_step(
                f"{updown}_blocks.{block_id_target}.resnets.{level_id}.conv1.{mode}",
                f"{inout}put_blocks.{block_id_source+level_id}.0.in_layers.2.{mode}"
                    )
            # embedding
            self.update_step(
                f"{updown}_blocks.{block_id_target}.resnets.{level_id}.time_emb_proj.{mode}",
                f"{inout}put_blocks.{block_id_source+level_id}.0.emb_layers.1.{mode}"
                    )
            # norm2
            self.update_step(
                f"{updown}_blocks.{block_id_target}.resnets.{level_id}.norm2.{mode}",
                f"{inout}put_blocks.{block_id_source+level_id}.0.out_layers.0.{mode}"
                    )
            # conv2
            self.update_step(
                f"{updown}_blocks.{block_id_target}.resnets.{level_id}.conv2.{mode}",
                f"{inout}put_blocks.{block_id_source+level_id}.0.out_layers.3.{mode}"
                    )
            # skip conv
            try:
                self.skip_conv_flag = False
                self.update_step(
                    f"{updown}_blocks.{block_id_target}.resnets.{level_id}.conv_shortcut.{mode}",
                    f"{inout}put_blocks.{block_id_source+level_id}.0.skip_connection.{mode}"
                        )
                self.skip_conv_flag = True
            except:
                pass

            # attention: norm layer
            try:
                self.attn_flag = False
                # norm layer
                self.update_step(
                    f"{updown}_blocks.{block_id_target}.attentions.{level_id}.group_norm.{mode}",
                    f"{inout}put_blocks.{block_id_source+level_id}.1.norm.{mode}"
                        )
                # linear layer
                self.update_step(
                    f"{updown}_blocks.{block_id_target}.attentions.{level_id}.proj_attn.{mode}",
                    f"{inout}put_blocks.{block_id_source+level_id}.1.proj_out.{mode}"
                        )
                # qkv
                query, key, value = split_qkv_weight(
                        self.source_checkpoint[f"{inout}put_blocks.{block_id_source+level_id}.1.qkv.{mode}"],
                        self.config,
                            )
                self.update_step(
                    f"{updown}_blocks.{block_id_target}.attentions.{level_id}.query.{mode}",
                    source_value=query,
                    add_copy=False
                        )
                self.update_step(
                    f"{updown}_blocks.{block_id_target}.attentions.{level_id}.key.{mode}",
                    source_value=key,
                    add_copy=False
                        )
                self.update_step(
                    f"{updown}_blocks.{block_id_target}.attentions.{level_id}.value.{mode}",
                    source_value=value,
                    add_copy=False,
                        )
                self.source_checkpoint.pop(
                    f"{inout}put_blocks.{block_id_source+level_id}.1.qkv.{mode}"
                        )
                self.num_copy += 1
                self.attn_flag = True
            except:
                pass

def convert_ddpm_checkpoint(new_checkpoint, checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    # new_checkpoint = {}
    converted_checkpoint = OrderedDict()
    num_copy = 0

    copy_weight(new_checkpoint["time_embedding.linear_1.weight"], checkpoint["time_embed.0.weight"])
    num_copy += 1
    copy_weight(new_checkpoint["time_embedding.linear_1.bias"], checkpoint["time_embed.0.bias"])
    num_copy += 1
    copy_weight(new_checkpoint["time_embedding.linear_2.weight"], checkpoint["time_embed.2.weight"])
    num_copy += 1
    copy_weight(new_checkpoint["time_embedding.linear_2.bias"], checkpoint["time_embed.2.bias"])
    num_copy += 1

    copy_weight(new_checkpoint["conv_norm_out.weight"], checkpoint["out.0.weight"])
    num_copy += 1
    copy_weight(new_checkpoint["conv_norm_out.bias"], checkpoint["out.0.bias"])
    num_copy += 1

    copy_weight(new_checkpoint["conv_in.weight"], checkpoint["input_blocks.0.0.weight"])
    num_copy += 1
    copy_weight(new_checkpoint["conv_in.bias"], checkpoint["input_blocks.0.0.bias"])
    num_copy += 1
    copy_weight(new_checkpoint["conv_out.weight"], checkpoint["out.2.weight"])
    num_copy += 1
    copy_weight(new_checkpoint["conv_out.bias"], checkpoint["out.2.bias"])
    num_copy += 1

    num_down_blocks = len([key for key in checkpoint.keys() if ("op." in key and 'input_blocks' in key)]) // 2 + 1
    num_up_blocks = num_down_blocks
    print(f"Down and up blocks: {num_down_blocks}")

    # down blocks
    block_id_source = 1
    for i in range(num_down_blocks):
        for j in range(config["layers_per_block"][i]):
            # resnet
            for mode in ['weight', 'bias']:
                # norm1
                copy_weight(
                    new_checkpoint[f"down_blocks.{i}.resnets.{j}.norm1.{mode}"],
                    checkpoint[f"input_blocks.{block_id_source+j}.0.in_layers.0.{mode}"]
                        )
                num_copy += 1
                # conv1
                copy_weight(
                    new_checkpoint[f"down_blocks.{i}.resnets.{j}.conv1.{mode}"],
                    checkpoint[f"input_blocks.{block_id_source+j}.0.in_layers.2.{mode}"]
                        )
                num_copy += 1
                # embedding
                copy_weight(
                    new_checkpoint[f"down_blocks.{i}.resnets.{j}.time_emb_proj.{mode}"],
                    checkpoint[f"input_blocks.{block_id_source+j}.0.emb_layers.1.{mode}"]
                        )
                num_copy += 1
                # norm2
                copy_weight(
                    new_checkpoint[f"down_blocks.{i}.resnets.{j}.norm2.{mode}"],
                    checkpoint[f"input_blocks.{block_id_source+j}.0.out_layers.0.{mode}"]
                        )
                num_copy += 1
                # conv2
                copy_weight(
                    new_checkpoint[f"down_blocks.{i}.resnets.{j}.conv2.{mode}"],
                    checkpoint[f"input_blocks.{block_id_source+j}.0.out_layers.3.{mode}"]
                        )
                num_copy += 1
                # skip conv
                try:
                    skip_conv_flag = False
                    copy_weight(
                        new_checkpoint[f"down_blocks.{i}.resnets.{j}.conv_shortcut.{mode}"],
                        checkpoint[f"input_blocks.{block_id_source+j}.0.skip_connection.{mode}"]
                            )
                    num_copy += 1
                    skip_conv_flag = True
                except:
                    pass

                # attention: norm layer
                try:
                    attn_flag = False
                    # norm layer
                    copy_weight(
                        new_checkpoint[f"down_blocks.{i}.attentions.{j}.group_norm.{mode}"],
                        checkpoint[f"input_blocks.{block_id_source+j}.1.norm.{mode}"]
                            )
                    num_copy += 1
                    # linear layer
                    copy_weight(
                        new_checkpoint[f"down_blocks.{i}.attentions.{j}.proj_attn.{mode}"],
                        checkpoint[f"input_blocks.{block_id_source+j}.1.proj_out.{mode}"]
                            )
                    num_copy += 1
                    # qkv
                    query, key, value = split_qkv_weight(
                                checkpoint[f"input_blocks.{block_id_source+j}.1.qkv.{mode}"],
                                config,
                                )
                    copy_weight(
                        new_checkpoint[f"down_blocks.{i}.attentions.{j}.query.{mode}"],
                        query,
                            )
                    copy_weight(
                        new_checkpoint[f"down_blocks.{i}.attentions.{j}.key.{mode}"],
                        key,
                            )
                    copy_weight(
                        new_checkpoint[f"down_blocks.{i}.attentions.{j}.value.{mode}"],
                        value,
                            )
                    num_copy += 1
                    attn_flag = True
                except:
                    pass

                # downsample
                if j+1 == config["layers_per_block"][i] and i + 1 < num_down_blocks:
                    copy_weight(
                        new_checkpoint[f"down_blocks.{i}.downsamplers.0.conv.{mode}"],
                        checkpoint[f"input_blocks.{block_id_source+j+1}.0.op.{mode}"],
                            )
                    num_copy += 1
                    if mode == 'bias':
                        block_id_source += 1

                print(f"Down Block: {i+1}, Level: {j+1}, {mode:6s}," + \
                      f" Att: {str(attn_flag):5s}," + \
                      f" Skip Conv: {str(skip_conv_flag):5s}")

        block_id_source += config["layers_per_block"][i]

    # middle blocks
    block_id_source = 0
    for i in range(2):
        for mode in ['weight', 'bias']:
            # norm1
            copy_weight(
                new_checkpoint[f"mid_block.resnets.{i}.norm1.{mode}"],
                checkpoint[f"middle_block.{block_id_source}.in_layers.0.{mode}"]
                    )
            num_copy += 1
            # conv1
            copy_weight(
                new_checkpoint[f"mid_block.resnets.{i}.conv1.{mode}"],
                checkpoint[f"middle_block.{block_id_source}.in_layers.2.{mode}"]
                    )
            num_copy += 1
            # embedding
            copy_weight(
                new_checkpoint[f"mid_block.resnets.{i}.time_emb_proj.{mode}"],
                checkpoint[f"middle_block.{block_id_source}.emb_layers.1.{mode}"]
                    )
            num_copy += 1
            # norm2
            copy_weight(
                new_checkpoint[f"mid_block.resnets.{i}.norm2.{mode}"],
                checkpoint[f"middle_block.{block_id_source}.out_layers.0.{mode}"]
                    )
            num_copy += 1
            # conv2
            copy_weight(
                new_checkpoint[f"mid_block.resnets.{i}.conv2.{mode}"],
                checkpoint[f"middle_block.{block_id_source}.out_layers.3.{mode}"]
                    )
            num_copy += 1

            # attention: norm layer
            if i == 0:
                attn_flag = False
                # norm layer
                copy_weight(
                    new_checkpoint[f"mid_block.attentions.{i}.group_norm.{mode}"],
                    checkpoint[f"middle_block.{block_id_source+1}.norm.{mode}"]
                        )
                num_copy += 1
                # linear layer
                copy_weight(
                    new_checkpoint[f"mid_block.attentions.{i}.proj_attn.{mode}"],
                    checkpoint[f"middle_block.{block_id_source+1}.proj_out.{mode}"]
                        )
                num_copy += 1
                # qkv
                query, key, value = split_qkv_weight(
                            checkpoint[f"middle_block.{block_id_source+1}.qkv.{mode}"],
                            config,
                            )
                copy_weight(
                    new_checkpoint[f"mid_block.attentions.{i}.query.{mode}"],
                    query,
                        )
                copy_weight(
                    new_checkpoint[f"mid_block.attentions.{i}.key.{mode}"],
                    key,
                        )
                copy_weight(
                    new_checkpoint[f"mid_block.attentions.{i}.value.{mode}"],
                    value,
                        )
                num_copy += 1
                attn_flag = True
                if mode == 'bias':
                    block_id_source += 1

            print(f"Mid Block: {i+1}, {mode}, Att: {str(attn_flag):5s}")

        block_id_source += 1

    # up blocks
    block_id_source = 0
    for i in range(num_up_blocks):
        for j in range(list(reversed(config["layers_per_block"]))[i]+1):
            # resnet
            for mode in ['weight', 'bias']:
                # norm1
                copy_weight(
                    new_checkpoint[f"up_blocks.{i}.resnets.{j}.norm1.{mode}"],
                    checkpoint[f"output_blocks.{block_id_source+j}.0.in_layers.0.{mode}"]
                        )
                num_copy += 1
                # conv1
                copy_weight(
                    new_checkpoint[f"up_blocks.{i}.resnets.{j}.conv1.{mode}"],
                    checkpoint[f"output_blocks.{block_id_source+j}.0.in_layers.2.{mode}"]
                        )
                num_copy += 1
                # embedding
                copy_weight(
                    new_checkpoint[f"up_blocks.{i}.resnets.{j}.time_emb_proj.{mode}"],
                    checkpoint[f"output_blocks.{block_id_source+j}.0.emb_layers.1.{mode}"]
                        )
                num_copy += 1
                # norm2
                copy_weight(
                    new_checkpoint[f"up_blocks.{i}.resnets.{j}.norm2.{mode}"],
                    checkpoint[f"output_blocks.{block_id_source+j}.0.out_layers.0.{mode}"]
                        )
                num_copy += 1
                # conv2
                copy_weight(
                    new_checkpoint[f"up_blocks.{i}.resnets.{j}.conv2.{mode}"],
                    checkpoint[f"output_blocks.{block_id_source+j}.0.out_layers.3.{mode}"]
                        )
                num_copy += 1
                # skip conv
                try:
                    skip_conv_flag = False
                    copy_weight(
                        new_checkpoint[f"up_blocks.{i}.resnets.{j}.conv_shortcut.{mode}"],
                        checkpoint[f"output_blocks.{block_id_source+j}.0.skip_connection.{mode}"]
                            )
                    num_copy += 1
                    skip_conv_flag = True
                except:
                    pass

                # attention: norm layer
                try:
                    attn_flag = False
                    # norm layer
                    copy_weight(
                        new_checkpoint[f"up_blocks.{i}.attentions.{j}.group_norm.{mode}"],
                        checkpoint[f"output_blocks.{block_id_source+j}.1.norm.{mode}"]
                            )
                    num_copy += 1
                    # linear layer
                    copy_weight(
                        new_checkpoint[f"up_blocks.{i}.attentions.{j}.proj_attn.{mode}"],
                        checkpoint[f"output_blocks.{block_id_source+j}.1.proj_out.{mode}"]
                            )
                    num_copy += 1
                    # qkv
                    query, key, value = split_qkv_weight(
                                checkpoint[f"output_blocks.{block_id_source+j}.1.qkv.{mode}"],
                                config,
                                )
                    copy_weight(
                        new_checkpoint[f"up_blocks.{i}.attentions.{j}.query.{mode}"],
                        query,
                            )
                    copy_weight(
                        new_checkpoint[f"up_blocks.{i}.attentions.{j}.key.{mode}"],
                        key,
                            )
                    copy_weight(
                        new_checkpoint[f"up_blocks.{i}.attentions.{j}.value.{mode}"],
                        value,
                            )
                    num_copy += 1
                    attn_flag = True
                except:
                    pass

                # upsample
                if j == list(reversed(config["layers_per_block"]))[i] and i + 1 < num_down_blocks:
                    up_index = '2' if attn_flag else '1'
                    copy_weight(
                        new_checkpoint[f"up_blocks.{i}.upsamplers.0.conv.{mode}"],
                        checkpoint[f"output_blocks.{block_id_source+j}.{up_index}.conv.{mode}"],
                            )
                    num_copy += 1

                print(f"Up Block: {i+1}, Level: {j+1}, {mode:6s}," + \
                      f" Att: {str(attn_flag):5s}," + \
                      f" Skip Conv: {str(skip_conv_flag):5s}")

        block_id_source += list(reversed(config["layers_per_block"]))[i] + 1

    assert num_copy ==  len(list(checkpoint.keys()))
    return new_checkpoint

def convert_vq_autoenc_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}

    new_checkpoint["encoder.conv_norm_out.weight"] = checkpoint["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = checkpoint["encoder.norm_out.bias"]

    new_checkpoint["encoder.conv_in.weight"] = checkpoint["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = checkpoint["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = checkpoint["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = checkpoint["encoder.conv_out.bias"]

    new_checkpoint["decoder.conv_norm_out.weight"] = checkpoint["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = checkpoint["decoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = checkpoint["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = checkpoint["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = checkpoint["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = checkpoint["decoder.conv_out.bias"]

    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in checkpoint if "down" in layer})
    down_blocks = {
        layer_id: [key for key in checkpoint if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in checkpoint if "up" in layer})
    up_blocks = {layer_id: [key for key in checkpoint if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)}

    for i in range(num_down_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)

        if any("downsample" in layer for layer in down_blocks[i]):
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = checkpoint[
                f"encoder.down.{i}.downsample.conv.weight"
            ]
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = checkpoint[
                f"encoder.down.{i}.downsample.conv.bias"
            ]

        if any("block" in layer for layer in down_blocks[i]):
            num_blocks = len(
                {".".join(shave_segments(layer, 3).split(".")[:3]) for layer in down_blocks[i] if "block" in layer}
            )
            blocks = {
                layer_id: [key for key in down_blocks[i] if f"block.{layer_id}" in key]
                for layer_id in range(num_blocks)
            }

            if num_blocks > 0:
                for j in range(config["layers_per_block"]):
                    paths = renew_resnet_paths(blocks[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint)

        if any("attn" in layer for layer in down_blocks[i]):
            num_attn = len(
                {".".join(shave_segments(layer, 3).split(".")[:3]) for layer in down_blocks[i] if "attn" in layer}
            )
            attns = {
                layer_id: [key for key in down_blocks[i] if f"attn.{layer_id}" in key]
                for layer_id in range(num_blocks)
            }

            if num_attn > 0:
                for j in range(config["layers_per_block"]):
                    paths = renew_attention_paths(attns[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint, config=config)

    mid_block_1_layers = [key for key in checkpoint if "mid.block_1" in key]
    mid_block_2_layers = [key for key in checkpoint if "mid.block_2" in key]
    mid_attn_1_layers = [key for key in checkpoint if "mid.attn_1" in key]

    # Mid new 2
    paths = renew_resnet_paths(mid_block_1_layers)
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        checkpoint,
        additional_replacements=[{"old": "mid.", "new": "mid_new_2."}, {"old": "block_1", "new": "resnets.0"}],
    )

    paths = renew_resnet_paths(mid_block_2_layers)
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        checkpoint,
        additional_replacements=[{"old": "mid.", "new": "mid_new_2."}, {"old": "block_2", "new": "resnets.1"}],
    )

    paths = renew_attention_paths(mid_attn_1_layers, in_mid=True)
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        checkpoint,
        additional_replacements=[{"old": "mid.", "new": "mid_new_2."}, {"old": "attn_1", "new": "attentions.0"}],
    )

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i

        if any("upsample" in layer for layer in up_blocks[i]):
            new_checkpoint[f"decoder.up_blocks.{block_id}.upsamplers.0.conv.weight"] = checkpoint[
                f"decoder.up.{i}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{block_id}.upsamplers.0.conv.bias"] = checkpoint[
                f"decoder.up.{i}.upsample.conv.bias"
            ]

        if any("block" in layer for layer in up_blocks[i]):
            num_blocks = len(
                {".".join(shave_segments(layer, 3).split(".")[:3]) for layer in up_blocks[i] if "block" in layer}
            )
            blocks = {
                layer_id: [key for key in up_blocks[i] if f"block.{layer_id}" in key] for layer_id in range(num_blocks)
            }

            if num_blocks > 0:
                for j in range(config["layers_per_block"] + 1):
                    replace_indices = {"old": f"up_blocks.{i}", "new": f"up_blocks.{block_id}"}
                    paths = renew_resnet_paths(blocks[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[replace_indices])

        if any("attn" in layer for layer in up_blocks[i]):
            num_attn = len(
                {".".join(shave_segments(layer, 3).split(".")[:3]) for layer in up_blocks[i] if "attn" in layer}
            )
            attns = {
                layer_id: [key for key in up_blocks[i] if f"attn.{layer_id}" in key] for layer_id in range(num_blocks)
            }

            if num_attn > 0:
                for j in range(config["layers_per_block"] + 1):
                    replace_indices = {"old": f"up_blocks.{i}", "new": f"up_blocks.{block_id}"}
                    paths = renew_attention_paths(attns[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint, additional_replacements=[replace_indices])

    new_checkpoint = {k.replace("mid_new_2", "mid_block"): v for k, v in new_checkpoint.items()}
    new_checkpoint["quant_conv.weight"] = checkpoint["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = checkpoint["quant_conv.bias"]
    if "quantize.embedding.weight" in checkpoint:
        new_checkpoint["quantize.embedding.weight"] = checkpoint["quantize.embedding.weight"]
    new_checkpoint["post_quant_conv.weight"] = checkpoint["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = checkpoint["post_quant_conv.bias"]

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
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    with open(args.config_file) as f:
        config = json.loads(f.read())

    if list(checkpoint.keys())[0].startswith('module'):
        checkpoint = OrderedDict({key[7:]:value for key, value in checkpoint.items()})

    model = UNet2DModel(**config)
    new_checkpoint = deepcopy(model.state_dict())
    # new_checkpoint_back = deepcopy(new_checkpoint)

    # unet case
    key_prefix_set = set(key.split(".")[0] for key in checkpoint.keys())
    if "encoder" in key_prefix_set and "decoder" in key_prefix_set:
        converted_checkpoint = convert_vq_autoenc_checkpoint(checkpoint, config)
    else:
        # converted_checkpoint = convert_ddpm_checkpoint(new_checkpoint, checkpoint, config)
        convertdifface = ConvertDifFace(new_checkpoint, checkpoint, config)
        converted_checkpoint = convertdifface.convert()

    if config["_class_name"] == "VQModel":
        model = VQModel(**config)
        model.load_state_dict(converted_checkpoint)
        model.save_pretrained(args.dump_path)
    elif config["_class_name"] == "AutoencoderKL":
        model = AutoencoderKL(**config)
        model.load_state_dict(converted_checkpoint)
        model.save_pretrained(args.dump_path)
    else:
        model.load_state_dict(converted_checkpoint, strict=True)

        scheduler = ImprovedDDPMScheduler.from_config("/".join(args.checkpoint_path.split("/")[:-1]))

        pipe = DifFacePipeline(unet=model, scheduler=scheduler)
        pipe.save_pretrained(args.dump_path)
