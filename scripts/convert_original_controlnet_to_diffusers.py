# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
""" Conversion script for stable diffusion checkpoints which _only_ contain a contrlnet. """

import argparse

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--original_config_file",
        type=str,
        required=True,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--image_size",
        default=512,
        type=int,
        help=(
            "The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2"
            " Base. Use 768 for Stable Diffusion v2."
        ),
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
        "--upcast_attention",
        action="store_true",
        help=(
            "Whether the attention computation should always be upcasted. This is necessary when running stable"
            " diffusion 2.1."
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

    controlnet = download_controlnet_from_original_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        extract_ema=args.extract_ema,
        num_in_channels=args.num_in_channels,
        upcast_attention=args.upcast_attention,
        from_safetensors=args.from_safetensors,
        device=args.device,
    )

    controlnet.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
