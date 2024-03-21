# coding=utf-8
# Copyright 2024, Haofan Wang, Qixun Wang, All rights reserved.
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

"""Conversion script for the LoRA's safetensors checkpoints."""

import argparse

import torch
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline


def convert(base_model_path, checkpoint_path, LORA_PREFIX_UNET, LORA_PREFIX_TEXT_ENCODER, alpha):
    # load base model
    pipeline = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32)

    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_path", default=None, type=str, required=True, help="Path to the base model in diffusers format."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--lora_prefix_unet", default="lora_unet", type=str, help="The prefix of UNet weight in safetensors"
    )
    parser.add_argument(
        "--lora_prefix_text_encoder",
        default="lora_te",
        type=str,
        help="The prefix of text encoder weight in safetensors",
    )
    parser.add_argument("--alpha", default=0.75, type=float, help="The merging ratio in W = W0 + alpha * deltaW")
    parser.add_argument(
        "--to_safetensors", action="store_true", help="Whether to store pipeline in safetensors format or not."
    )
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")

    args = parser.parse_args()

    base_model_path = args.base_model_path
    checkpoint_path = args.checkpoint_path
    dump_path = args.dump_path
    lora_prefix_unet = args.lora_prefix_unet
    lora_prefix_text_encoder = args.lora_prefix_text_encoder
    alpha = args.alpha

    pipe = convert(base_model_path, checkpoint_path, lora_prefix_unet, lora_prefix_text_encoder, alpha)

    pipe = pipe.to(args.device)
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
