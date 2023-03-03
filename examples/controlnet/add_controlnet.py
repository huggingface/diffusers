#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import os
from diffusers import UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Add controlnet to existing model by copying unet weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory where the model will be written.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--controlnet_only",
        action="store_true",
        help=(
            "Save only the controlnet weights."
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    device_map = 'cpu'

    try:
        config, unused_kwargs = ControlNetModel.load_config(
            args.pretrained_model_name_or_path,
            return_unused_kwargs=True,
            revision=args.revision,
            subfolder="controlnet",
            device_map=device_map,
        )
        print("Model already has controlnet. Doing nothing.")
        return
    except OSError:
        pass

    config, unused_kwargs = UNet2DConditionModel.load_config(
        args.pretrained_model_name_or_path,
        return_unused_kwargs=True,
        revision=args.revision,
        subfolder="unet",
        device_map=device_map,
    )

    config["_class_name"] = 'ControlNetModel'
    config.pop("conv_out_kernel", None)
    config.pop("out_channels", None)
    config.pop("up_block_types", None)
    config.pop("center_input_sample", None)
    config.pop("conv_in_kernel", None)
    config.pop("dual_cross_attention", None)
    config.pop("mid_block_type", None)
    config.pop("sample_size", None)
    config.pop("time_cond_proj_dim", None)
    config.pop("time_embedding_type", None)
    config.pop("timestep_post_act", None)

    controlnet = ControlNetModel.from_config(config, **unused_kwargs)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    unet = unet.state_dict()
    temp_state = controlnet.state_dict()

    for k in temp_state.keys():
        if k in unet:
            temp_state[k] = unet[k].clone()
        else:
            if 'controlnet' not in k:
                print('Not found in unet:', k)

    controlnet.load_state_dict(temp_state, strict=True)
    if args.controlnet_only:
        controlnet.save_pretrained(os.path.join(args.output_dir, 'controlnet'))
    else:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

    print('Done')

if __name__ == "__main__":
    args = parse_args()
    main(args)
