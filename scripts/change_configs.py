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
import os
import json
import torch
from diffusers import UNet2DModel, UNet2DConditionModel
from transformers.file_utils import has_file

do_only_config = False
do_only_weights = True
do_only_renaming = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_path",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to the output model."
    )

    args = parser.parse_args()

    config_parameters_to_change = {
        "image_size": "sample_size",
        "num_res_blocks": "layers_per_block",
        "block_channels": "block_out_channels",
        "down_blocks": "down_block_types",
        "up_blocks": "up_block_types",
        "downscale_freq_shift": "freq_shift",
        "resnet_num_groups": "norm_num_groups",
        "resnet_act_fn": "act_fn",
        "resnet_eps": "norm_eps",
        "num_head_channels": "attention_head_dim",
    }

    key_parameters_to_change = {
        "time_steps": "time_proj",
        "mid": "mid_block",
        "downsample_blocks": "down_blocks",
        "upsample_blocks": "up_blocks",
    }

    subfolder = "" if has_file(args.repo_path, "config.json") else "unet"

    with open(os.path.join(args.repo_path, subfolder, "config.json"), "r", encoding="utf-8") as reader:
        text = reader.read()
        config = json.loads(text)

    if do_only_config:
        for key in config_parameters_to_change.keys():
            config.pop(key, None)

    if has_file(args.repo_path, "config.json"):
        model = UNet2DModel(**config)
    else:
        class_name = UNet2DConditionModel if "ldm-text2im-large-256" in args.repo_path else UNet2DModel
        model = class_name(**config)

    if do_only_config:
        model.save_config(os.path.join(args.repo_path, subfolder))

    config = dict(model.config)

    if do_only_renaming:
        for key, value in config_parameters_to_change.items():
            if key in config:
                config[value] = config[key]
                del config[key]

        config["down_block_types"] = [k.replace("UNetRes", "") for k in config["down_block_types"]]
        config["up_block_types"] = [k.replace("UNetRes", "") for k in config["up_block_types"]]

    if do_only_weights:
        state_dict = torch.load(os.path.join(args.repo_path, subfolder, "diffusion_pytorch_model.bin"))

        new_state_dict = {}
        for param_key, param_value in state_dict.items():
            if param_key.endswith(".op.bias") or param_key.endswith(".op.weight"):
                continue
            has_changed = False
            for key, new_key in key_parameters_to_change.items():
                if not has_changed and param_key.split(".")[0] == key:
                    new_state_dict[".".join([new_key] + param_key.split(".")[1:])] = param_value
                    has_changed = True
            if not has_changed:
                new_state_dict[param_key] = param_value

        model.load_state_dict(new_state_dict)
        model.save_pretrained(os.path.join(args.repo_path, subfolder))
