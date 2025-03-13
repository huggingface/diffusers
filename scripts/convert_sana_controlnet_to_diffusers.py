#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights

from diffusers import (
    SanaControlNetModel,
)
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext


def main(args):
    file_path = args.orig_ckpt_path

    all_state_dict = torch.load(file_path, weights_only=True)
    state_dict = all_state_dict.pop("state_dict")
    converted_state_dict = {}

    # Patch embeddings.
    converted_state_dict["patch_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["patch_embed.proj.bias"] = state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    converted_state_dict["caption_projection.linear_1.weight"] = state_dict.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = state_dict.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = state_dict.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = state_dict.pop("y_embedder.y_proj.fc2.bias")

    # AdaLN-single LN
    converted_state_dict["time_embed.emb.timestep_embedder.linear_1.weight"] = state_dict.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["time_embed.emb.timestep_embedder.linear_1.bias"] = state_dict.pop("t_embedder.mlp.0.bias")
    converted_state_dict["time_embed.emb.timestep_embedder.linear_2.weight"] = state_dict.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["time_embed.emb.timestep_embedder.linear_2.bias"] = state_dict.pop("t_embedder.mlp.2.bias")

    # Shared norm.
    converted_state_dict["time_embed.linear.weight"] = state_dict.pop("t_block.1.weight")
    converted_state_dict["time_embed.linear.bias"] = state_dict.pop("t_block.1.bias")

    # y norm
    converted_state_dict["caption_norm.weight"] = state_dict.pop("attention_y_norm.weight")

    # Positional embedding interpolation scale.
    interpolation_scale = {512: None, 1024: None, 2048: 1.0, 4096: 2.0}

    # ControlNet Input Projection.
    converted_state_dict["input_block.weight"] = state_dict.pop("controlnet.0.before_proj.weight")
    converted_state_dict["input_block.bias"] = state_dict.pop("controlnet.0.before_proj.bias")

    for depth in range(7):
        # Transformer blocks.
        converted_state_dict[f"transformer_blocks.{depth}.scale_shift_table"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.scale_shift_table"
        )

        # Linear Attention is all you need 🤘
        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.attn.qkv.weight"), 3, dim=0)
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        # Projection.
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_inverted.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.inverted_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_inverted.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.inverted_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_depth.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.depth_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_depth.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.depth_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_point.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.mlp.point_conv.conv.weight"
        )

        # Cross-attention.
        q = state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(
            state_dict.pop(f"controlnet.{depth}.copied_block.cross_attn.kv_linear.bias"), 2, dim=0
        )

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.weight"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.cross_attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.bias"] = state_dict.pop(
            f"controlnet.{depth}.copied_block.cross_attn.proj.bias"
        )

        # ControlNet After Projection
        converted_state_dict[f"controlnet_blocks.{depth}.weight"] = state_dict.pop(
            f"controlnet.{depth}.after_proj.weight"
        )
        converted_state_dict[f"controlnet_blocks.{depth}.bias"] = state_dict.pop(f"controlnet.{depth}.after_proj.bias")

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

    if is_accelerate_available():
        load_model_dict_into_meta(controlnet, converted_state_dict)
    else:
        controlnet.load_state_dict(converted_state_dict, strict=True, assign=True)

    num_model_params = sum(p.numel() for p in controlnet.parameters())
    print(f"Total number of controlnet parameters: {num_model_params}")

    controlnet = controlnet.to(weight_dtype)
    controlnet.load_state_dict(converted_state_dict, strict=True)

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
