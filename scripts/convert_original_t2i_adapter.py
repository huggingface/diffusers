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
"""
Conversion script for the T2I-Adapter checkpoints.
"""

import argparse

import torch

from diffusers import T2IAdapter


def convert_adapter(src_state, in_channels):
    original_body_length = max([int(x.split(".")[1]) for x in src_state.keys() if "body." in x]) + 1

    assert original_body_length == 8

    # (0, 1) -> channels 1
    assert src_state["body.0.block1.weight"].shape == (320, 320, 3, 3)

    # (2, 3) -> channels 2
    assert src_state["body.2.in_conv.weight"].shape == (640, 320, 1, 1)

    # (4, 5) -> channels 3
    assert src_state["body.4.in_conv.weight"].shape == (1280, 640, 1, 1)

    # (6, 7) -> channels 4
    assert src_state["body.6.block1.weight"].shape == (1280, 1280, 3, 3)

    res_state = {
        "adapter.conv_in.weight": src_state.pop("conv_in.weight"),
        "adapter.conv_in.bias": src_state.pop("conv_in.bias"),
        # 0.resnets.0
        "adapter.body.0.resnets.0.block1.weight": src_state.pop("body.0.block1.weight"),
        "adapter.body.0.resnets.0.block1.bias": src_state.pop("body.0.block1.bias"),
        "adapter.body.0.resnets.0.block2.weight": src_state.pop("body.0.block2.weight"),
        "adapter.body.0.resnets.0.block2.bias": src_state.pop("body.0.block2.bias"),
        # 0.resnets.1
        "adapter.body.0.resnets.1.block1.weight": src_state.pop("body.1.block1.weight"),
        "adapter.body.0.resnets.1.block1.bias": src_state.pop("body.1.block1.bias"),
        "adapter.body.0.resnets.1.block2.weight": src_state.pop("body.1.block2.weight"),
        "adapter.body.0.resnets.1.block2.bias": src_state.pop("body.1.block2.bias"),
        # 1
        "adapter.body.1.in_conv.weight": src_state.pop("body.2.in_conv.weight"),
        "adapter.body.1.in_conv.bias": src_state.pop("body.2.in_conv.bias"),
        # 1.resnets.0
        "adapter.body.1.resnets.0.block1.weight": src_state.pop("body.2.block1.weight"),
        "adapter.body.1.resnets.0.block1.bias": src_state.pop("body.2.block1.bias"),
        "adapter.body.1.resnets.0.block2.weight": src_state.pop("body.2.block2.weight"),
        "adapter.body.1.resnets.0.block2.bias": src_state.pop("body.2.block2.bias"),
        # 1.resnets.1
        "adapter.body.1.resnets.1.block1.weight": src_state.pop("body.3.block1.weight"),
        "adapter.body.1.resnets.1.block1.bias": src_state.pop("body.3.block1.bias"),
        "adapter.body.1.resnets.1.block2.weight": src_state.pop("body.3.block2.weight"),
        "adapter.body.1.resnets.1.block2.bias": src_state.pop("body.3.block2.bias"),
        # 2
        "adapter.body.2.in_conv.weight": src_state.pop("body.4.in_conv.weight"),
        "adapter.body.2.in_conv.bias": src_state.pop("body.4.in_conv.bias"),
        # 2.resnets.0
        "adapter.body.2.resnets.0.block1.weight": src_state.pop("body.4.block1.weight"),
        "adapter.body.2.resnets.0.block1.bias": src_state.pop("body.4.block1.bias"),
        "adapter.body.2.resnets.0.block2.weight": src_state.pop("body.4.block2.weight"),
        "adapter.body.2.resnets.0.block2.bias": src_state.pop("body.4.block2.bias"),
        # 2.resnets.1
        "adapter.body.2.resnets.1.block1.weight": src_state.pop("body.5.block1.weight"),
        "adapter.body.2.resnets.1.block1.bias": src_state.pop("body.5.block1.bias"),
        "adapter.body.2.resnets.1.block2.weight": src_state.pop("body.5.block2.weight"),
        "adapter.body.2.resnets.1.block2.bias": src_state.pop("body.5.block2.bias"),
        # 3.resnets.0
        "adapter.body.3.resnets.0.block1.weight": src_state.pop("body.6.block1.weight"),
        "adapter.body.3.resnets.0.block1.bias": src_state.pop("body.6.block1.bias"),
        "adapter.body.3.resnets.0.block2.weight": src_state.pop("body.6.block2.weight"),
        "adapter.body.3.resnets.0.block2.bias": src_state.pop("body.6.block2.bias"),
        # 3.resnets.1
        "adapter.body.3.resnets.1.block1.weight": src_state.pop("body.7.block1.weight"),
        "adapter.body.3.resnets.1.block1.bias": src_state.pop("body.7.block1.bias"),
        "adapter.body.3.resnets.1.block2.weight": src_state.pop("body.7.block2.weight"),
        "adapter.body.3.resnets.1.block2.bias": src_state.pop("body.7.block2.bias"),
    }

    assert len(src_state) == 0

    adapter = T2IAdapter(in_channels=in_channels, adapter_type="full_adapter")

    adapter.load_state_dict(res_state)

    return adapter


def convert_light_adapter(src_state):
    original_body_length = max([int(x.split(".")[1]) for x in src_state.keys() if "body." in x]) + 1

    assert original_body_length == 4

    res_state = {
        # body.0.in_conv
        "adapter.body.0.in_conv.weight": src_state.pop("body.0.in_conv.weight"),
        "adapter.body.0.in_conv.bias": src_state.pop("body.0.in_conv.bias"),
        # body.0.resnets.0
        "adapter.body.0.resnets.0.block1.weight": src_state.pop("body.0.body.0.block1.weight"),
        "adapter.body.0.resnets.0.block1.bias": src_state.pop("body.0.body.0.block1.bias"),
        "adapter.body.0.resnets.0.block2.weight": src_state.pop("body.0.body.0.block2.weight"),
        "adapter.body.0.resnets.0.block2.bias": src_state.pop("body.0.body.0.block2.bias"),
        # body.0.resnets.1
        "adapter.body.0.resnets.1.block1.weight": src_state.pop("body.0.body.1.block1.weight"),
        "adapter.body.0.resnets.1.block1.bias": src_state.pop("body.0.body.1.block1.bias"),
        "adapter.body.0.resnets.1.block2.weight": src_state.pop("body.0.body.1.block2.weight"),
        "adapter.body.0.resnets.1.block2.bias": src_state.pop("body.0.body.1.block2.bias"),
        # body.0.resnets.2
        "adapter.body.0.resnets.2.block1.weight": src_state.pop("body.0.body.2.block1.weight"),
        "adapter.body.0.resnets.2.block1.bias": src_state.pop("body.0.body.2.block1.bias"),
        "adapter.body.0.resnets.2.block2.weight": src_state.pop("body.0.body.2.block2.weight"),
        "adapter.body.0.resnets.2.block2.bias": src_state.pop("body.0.body.2.block2.bias"),
        # body.0.resnets.3
        "adapter.body.0.resnets.3.block1.weight": src_state.pop("body.0.body.3.block1.weight"),
        "adapter.body.0.resnets.3.block1.bias": src_state.pop("body.0.body.3.block1.bias"),
        "adapter.body.0.resnets.3.block2.weight": src_state.pop("body.0.body.3.block2.weight"),
        "adapter.body.0.resnets.3.block2.bias": src_state.pop("body.0.body.3.block2.bias"),
        # body.0.out_conv
        "adapter.body.0.out_conv.weight": src_state.pop("body.0.out_conv.weight"),
        "adapter.body.0.out_conv.bias": src_state.pop("body.0.out_conv.bias"),
        # body.1.in_conv
        "adapter.body.1.in_conv.weight": src_state.pop("body.1.in_conv.weight"),
        "adapter.body.1.in_conv.bias": src_state.pop("body.1.in_conv.bias"),
        # body.1.resnets.0
        "adapter.body.1.resnets.0.block1.weight": src_state.pop("body.1.body.0.block1.weight"),
        "adapter.body.1.resnets.0.block1.bias": src_state.pop("body.1.body.0.block1.bias"),
        "adapter.body.1.resnets.0.block2.weight": src_state.pop("body.1.body.0.block2.weight"),
        "adapter.body.1.resnets.0.block2.bias": src_state.pop("body.1.body.0.block2.bias"),
        # body.1.resnets.1
        "adapter.body.1.resnets.1.block1.weight": src_state.pop("body.1.body.1.block1.weight"),
        "adapter.body.1.resnets.1.block1.bias": src_state.pop("body.1.body.1.block1.bias"),
        "adapter.body.1.resnets.1.block2.weight": src_state.pop("body.1.body.1.block2.weight"),
        "adapter.body.1.resnets.1.block2.bias": src_state.pop("body.1.body.1.block2.bias"),
        # body.1.body.2
        "adapter.body.1.resnets.2.block1.weight": src_state.pop("body.1.body.2.block1.weight"),
        "adapter.body.1.resnets.2.block1.bias": src_state.pop("body.1.body.2.block1.bias"),
        "adapter.body.1.resnets.2.block2.weight": src_state.pop("body.1.body.2.block2.weight"),
        "adapter.body.1.resnets.2.block2.bias": src_state.pop("body.1.body.2.block2.bias"),
        # body.1.body.3
        "adapter.body.1.resnets.3.block1.weight": src_state.pop("body.1.body.3.block1.weight"),
        "adapter.body.1.resnets.3.block1.bias": src_state.pop("body.1.body.3.block1.bias"),
        "adapter.body.1.resnets.3.block2.weight": src_state.pop("body.1.body.3.block2.weight"),
        "adapter.body.1.resnets.3.block2.bias": src_state.pop("body.1.body.3.block2.bias"),
        # body.1.out_conv
        "adapter.body.1.out_conv.weight": src_state.pop("body.1.out_conv.weight"),
        "adapter.body.1.out_conv.bias": src_state.pop("body.1.out_conv.bias"),
        # body.2.in_conv
        "adapter.body.2.in_conv.weight": src_state.pop("body.2.in_conv.weight"),
        "adapter.body.2.in_conv.bias": src_state.pop("body.2.in_conv.bias"),
        # body.2.body.0
        "adapter.body.2.resnets.0.block1.weight": src_state.pop("body.2.body.0.block1.weight"),
        "adapter.body.2.resnets.0.block1.bias": src_state.pop("body.2.body.0.block1.bias"),
        "adapter.body.2.resnets.0.block2.weight": src_state.pop("body.2.body.0.block2.weight"),
        "adapter.body.2.resnets.0.block2.bias": src_state.pop("body.2.body.0.block2.bias"),
        # body.2.body.1
        "adapter.body.2.resnets.1.block1.weight": src_state.pop("body.2.body.1.block1.weight"),
        "adapter.body.2.resnets.1.block1.bias": src_state.pop("body.2.body.1.block1.bias"),
        "adapter.body.2.resnets.1.block2.weight": src_state.pop("body.2.body.1.block2.weight"),
        "adapter.body.2.resnets.1.block2.bias": src_state.pop("body.2.body.1.block2.bias"),
        # body.2.body.2
        "adapter.body.2.resnets.2.block1.weight": src_state.pop("body.2.body.2.block1.weight"),
        "adapter.body.2.resnets.2.block1.bias": src_state.pop("body.2.body.2.block1.bias"),
        "adapter.body.2.resnets.2.block2.weight": src_state.pop("body.2.body.2.block2.weight"),
        "adapter.body.2.resnets.2.block2.bias": src_state.pop("body.2.body.2.block2.bias"),
        # body.2.body.3
        "adapter.body.2.resnets.3.block1.weight": src_state.pop("body.2.body.3.block1.weight"),
        "adapter.body.2.resnets.3.block1.bias": src_state.pop("body.2.body.3.block1.bias"),
        "adapter.body.2.resnets.3.block2.weight": src_state.pop("body.2.body.3.block2.weight"),
        "adapter.body.2.resnets.3.block2.bias": src_state.pop("body.2.body.3.block2.bias"),
        # body.2.out_conv
        "adapter.body.2.out_conv.weight": src_state.pop("body.2.out_conv.weight"),
        "adapter.body.2.out_conv.bias": src_state.pop("body.2.out_conv.bias"),
        # body.3.in_conv
        "adapter.body.3.in_conv.weight": src_state.pop("body.3.in_conv.weight"),
        "adapter.body.3.in_conv.bias": src_state.pop("body.3.in_conv.bias"),
        # body.3.body.0
        "adapter.body.3.resnets.0.block1.weight": src_state.pop("body.3.body.0.block1.weight"),
        "adapter.body.3.resnets.0.block1.bias": src_state.pop("body.3.body.0.block1.bias"),
        "adapter.body.3.resnets.0.block2.weight": src_state.pop("body.3.body.0.block2.weight"),
        "adapter.body.3.resnets.0.block2.bias": src_state.pop("body.3.body.0.block2.bias"),
        # body.3.body.1
        "adapter.body.3.resnets.1.block1.weight": src_state.pop("body.3.body.1.block1.weight"),
        "adapter.body.3.resnets.1.block1.bias": src_state.pop("body.3.body.1.block1.bias"),
        "adapter.body.3.resnets.1.block2.weight": src_state.pop("body.3.body.1.block2.weight"),
        "adapter.body.3.resnets.1.block2.bias": src_state.pop("body.3.body.1.block2.bias"),
        # body.3.body.2
        "adapter.body.3.resnets.2.block1.weight": src_state.pop("body.3.body.2.block1.weight"),
        "adapter.body.3.resnets.2.block1.bias": src_state.pop("body.3.body.2.block1.bias"),
        "adapter.body.3.resnets.2.block2.weight": src_state.pop("body.3.body.2.block2.weight"),
        "adapter.body.3.resnets.2.block2.bias": src_state.pop("body.3.body.2.block2.bias"),
        # body.3.body.3
        "adapter.body.3.resnets.3.block1.weight": src_state.pop("body.3.body.3.block1.weight"),
        "adapter.body.3.resnets.3.block1.bias": src_state.pop("body.3.body.3.block1.bias"),
        "adapter.body.3.resnets.3.block2.weight": src_state.pop("body.3.body.3.block2.weight"),
        "adapter.body.3.resnets.3.block2.bias": src_state.pop("body.3.body.3.block2.bias"),
        # body.3.out_conv
        "adapter.body.3.out_conv.weight": src_state.pop("body.3.out_conv.weight"),
        "adapter.body.3.out_conv.bias": src_state.pop("body.3.out_conv.bias"),
    }

    assert len(src_state) == 0

    adapter = T2IAdapter(in_channels=3, channels=[320, 640, 1280], num_res_blocks=4, adapter_type="light_adapter")

    adapter.load_state_dict(res_state)

    return adapter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--output_path", default=None, type=str, required=True, help="Path to the store the result checkpoint."
    )
    parser.add_argument(
        "--is_adapter_light",
        action="store_true",
        help="Is checkpoint come from Adapter-Light architecture. ex: color-adapter",
    )
    parser.add_argument("--in_channels", required=False, type=int, help="Input channels for non-light adapter")

    args = parser.parse_args()
    src_state = torch.load(args.checkpoint_path)

    if args.is_adapter_light:
        adapter = convert_light_adapter(src_state)
    else:
        if args.in_channels is None:
            raise ValueError("set `--in_channels=<n>`")
        adapter = convert_adapter(src_state, args.in_channels)

    adapter.save_pretrained(args.output_path)
