#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import (
    AutoencoderKLWan,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaVideoPipeline,
    SanaVideoTransformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext

ckpt_ids = ["Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"]
# https://github.com/NVlabs/Sana/blob/main/inference_video_scripts/inference_sana_video.py


def main(args):
    cache_dir_path = os.path.expanduser("~/.cache/huggingface/hub")

    if args.orig_ckpt_path is None or args.orig_ckpt_path in ckpt_ids:
        ckpt_id = args.orig_ckpt_path or ckpt_ids[0]
        snapshot_download(
            repo_id=f"{'/'.join(ckpt_id.split('/')[:2])}",
            cache_dir=cache_dir_path,
            repo_type="model",
        )
        file_path = hf_hub_download(
            repo_id=f"{'/'.join(ckpt_id.split('/')[:2])}",
            filename=f"{'/'.join(ckpt_id.split('/')[2:])}",
            cache_dir=cache_dir_path,
            repo_type="model",
        )
    else:
        file_path = args.orig_ckpt_path

    print(colored(f"Loading checkpoint from {file_path}", "green", attrs=["bold"]))
    all_state_dict = torch.load(file_path, weights_only=True)
    state_dict = all_state_dict.pop("state_dict")
    converted_state_dict = {}

    # Patch embeddings.
    converted_state_dict["patch_embedding.weight"] = state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["patch_embedding.bias"] = state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    converted_state_dict["caption_projection.linear_1.weight"] = state_dict.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = state_dict.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = state_dict.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = state_dict.pop("y_embedder.y_proj.fc2.bias")

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

    # scheduler
    flow_shift = 8.0

    # model config
    layer_num = 20
    # Positional embedding interpolation scale.
    qk_norm = True

    # sample size
    if args.video_size == 480:
        sample_size = 30  # Wan-VAE: 8xp2 downsample factor
        patch_size = (1, 2, 2)
    elif args.video_size == 720:
        sample_size = 22  # Wan-VAE: 32xp1 downsample factor
        patch_size = (1, 1, 1)
    else:
        raise ValueError(f"Video size {args.video_size} is not supported.")

    for depth in range(layer_num):
        # Transformer blocks.
        converted_state_dict[f"transformer_blocks.{depth}.scale_shift_table"] = state_dict.pop(
            f"blocks.{depth}.scale_shift_table"
        )

        # Linear Attention is all you need 🤘
        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.weight"), 3, dim=0)
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        if qk_norm is not None:
            # Add Q/K normalization for self-attention (attn1) - needed for Sana-Sprint and Sana-1.5
            converted_state_dict[f"transformer_blocks.{depth}.attn1.norm_q.weight"] = state_dict.pop(
                f"blocks.{depth}.attn.q_norm.weight"
            )
            converted_state_dict[f"transformer_blocks.{depth}.attn1.norm_k.weight"] = state_dict.pop(
                f"blocks.{depth}.attn.k_norm.weight"
            )
        # Projection.
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_inverted.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.inverted_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_inverted.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.inverted_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_depth.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.depth_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_depth.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.depth_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_point.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.point_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.conv_temp.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.t_conv.weight"
        )

        # Cross-attention.
        q = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.bias"), 2, dim=0)

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.bias"] = v_bias
        if qk_norm is not None:
            # Add Q/K normalization for cross-attention (attn2) - needed for Sana-Sprint and Sana-1.5
            converted_state_dict[f"transformer_blocks.{depth}.attn2.norm_q.weight"] = state_dict.pop(
                f"blocks.{depth}.cross_attn.q_norm.weight"
            )
            converted_state_dict[f"transformer_blocks.{depth}.attn2.norm_k.weight"] = state_dict.pop(
                f"blocks.{depth}.cross_attn.k_norm.weight"
            )

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.bias"
        )

    # Final block.
    converted_state_dict["proj_out.weight"] = state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = state_dict.pop("final_layer.linear.bias")
    converted_state_dict["scale_shift_table"] = state_dict.pop("final_layer.scale_shift_table")

    # Transformer
    with CTX():
        transformer_kwargs = {
            "in_channels": 16,
            "out_channels": 16,
            "num_attention_heads": 20,
            "attention_head_dim": 112,
            "num_layers": 20,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "caption_channels": 2304,
            "mlp_ratio": 3.0,
            "attention_bias": False,
            "sample_size": sample_size,
            "patch_size": patch_size,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 1024,
        }

        transformer = SanaVideoTransformer3DModel(**transformer_kwargs)

    transformer.load_state_dict(converted_state_dict, strict=True, assign=True)

    try:
        state_dict.pop("y_embedder.y_embedding")
        state_dict.pop("pos_embed")
        state_dict.pop("logvar_linear.weight")
        state_dict.pop("logvar_linear.bias")
    except KeyError:
        print("y_embedder.y_embedding or pos_embed not found in the state_dict")

    assert len(state_dict) == 0, f"State dict is not empty, {state_dict.keys()}"

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    transformer = transformer.to(weight_dtype)

    if not args.save_full_pipeline:
        print(
            colored(
                f"Only saving transformer model of {args.model_type}. "
                f"Set --save_full_pipeline to save the whole Pipeline",
                "green",
                attrs=["bold"],
            )
        )
        transformer.save_pretrained(
            os.path.join(args.dump_path, "transformer"), safe_serialization=True, max_shard_size="5GB"
        )
    else:
        print(colored(f"Saving the whole Pipeline containing {args.model_type}", "green", attrs=["bold"]))
        # VAE
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32
        )

        # Text Encoder
        text_encoder_model_path = "Efficient-Large-Model/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_path)
        tokenizer.padding_side = "right"
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_model_path, torch_dtype=torch.bfloat16
        ).get_decoder()

        # Choose the appropriate pipeline and scheduler based on model type
        # Original Sana scheduler
        if args.scheduler_type == "flow-dpm_solver":
            scheduler = DPMSolverMultistepScheduler(
                flow_shift=flow_shift,
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
            )
        elif args.scheduler_type == "flow-euler":
            scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        elif args.scheduler_type == "uni-pc":
            scheduler = UniPCMultistepScheduler(
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                num_train_timesteps=1000,
                flow_shift=flow_shift,
            )
        else:
            raise ValueError(f"Scheduler type {args.scheduler_type} is not supported")

        pipe = SanaVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        pipe.save_pretrained(args.dump_path, safe_serialization=True, max_shard_size="5GB")


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--orig_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--video_size",
        default=480,
        type=int,
        choices=[480, 720],
        required=False,
        help="Video size of pretrained model, 480 or 720.",
    )
    parser.add_argument(
        "--model_type",
        default="SanaVideo",
        type=str,
        choices=[
            "SanaVideo",
        ],
    )
    parser.add_argument(
        "--scheduler_type",
        default="flow-dpm_solver",
        type=str,
        choices=["flow-dpm_solver", "flow-euler", "uni-pc"],
        help="Scheduler type to use.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--save_full_pipeline", action="store_true", help="save all the pipeline elements in one.")
    parser.add_argument("--dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="Weight dtype.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = DTYPE_MAPPING[args.dtype]

    main(args)
