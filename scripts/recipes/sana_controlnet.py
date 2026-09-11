#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights

from diffusers import (
    SanaControlNetModel,
)
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext


def main(args):
    file_path = args.orig_ckpt_path

    all_state_dict = torch.load(file_path, weights_only=True)
    state_dict = all_state_dict.pop("state_dict")
    interpolation_scale = {512: None, 1024: None, 2048: 1.0, 4096: 2.0}

    # ControlNet
    with CTX():
        controlnet = SanaControlNetModel(
            num_attention_heads=model_kwargs[args.model_type]["num_attention_heads"],
            attention_head_dim=model_kwargs[args.model_type]["attention_head_dim"],
            num_layers=model_kwargs[args.model_type]["num_layers"],
            num_cross_attention_heads=model_kwargs[args.model_type]["num_cross_attention_heads"],
            cross_attention_head_dim=model_kwargs[args.model_type]["cross_attention_head_dim"],
            cross_attention_dim=model_kwargs[args.model_type]["cross_attention_dim"],
            caption_channels=2304,
            sample_size=args.image_size // 32,
            interpolation_scale=interpolation_scale[args.image_size],
        )

    converted_state_dict = convert_component_checkpoint(state_dict, dict(controlnet.config), "SanaControlNetModel")
    controlnet.load_state_dict(converted_state_dict, strict=True, assign=True)

    num_model_params = sum(p.numel() for p in controlnet.parameters())
    print(f"Total number of controlnet parameters: {num_model_params}")

    controlnet = controlnet.to(weight_dtype)

    print(f"Saving Sana ControlNet in Diffusers format in {args.dump_path}.")
    controlnet.save_pretrained(args.dump_path)


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

VARIANT_MAPPING = {
    "fp32": None,
    "fp16": "fp16",
    "bf16": "bf16",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--orig_ckpt_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
        choices=[512, 1024, 2048, 4096],
        required=False,
        help="Image size of pretrained model, 512, 1024, 2048 or 4096.",
    )
    parser.add_argument(
        "--model_type",
        default="SanaMS_1600M_P1_ControlNet_D7",
        type=str,
        choices=["SanaMS_1600M_P1_ControlNet_D7", "SanaMS_600M_P1_ControlNet_D7"],
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--dtype", default="fp16", type=str, choices=["fp32", "fp16", "bf16"], help="Weight dtype.")

    args = parser.parse_args()

    model_kwargs = {
        "SanaMS_1600M_P1_ControlNet_D7": {
            "num_attention_heads": 70,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "num_layers": 7,
        },
        "SanaMS_600M_P1_ControlNet_D7": {
            "num_attention_heads": 36,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 16,
            "cross_attention_head_dim": 72,
            "cross_attention_dim": 1152,
            "num_layers": 7,
        },
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = DTYPE_MAPPING[args.dtype]
    variant = VARIANT_MAPPING[args.dtype]

    main(args)
