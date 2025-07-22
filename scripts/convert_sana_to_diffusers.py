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
    AutoencoderDC,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaSprintPipeline,
    SanaTransformer2DModel,
    SCMScheduler,
)
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext

ckpt_ids = [
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px/checkpoints/Sana_Sprint_0.6B_1024px.pth"
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"
    "Efficient-Large-Model/SANA1.5_4.8B_1024px/checkpoints/SANA1.5_4.8B_1024px.pth",
    "Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth",
    "Efficient-Large-Model/Sana_1600M_4Kpx_BF16/checkpoints/Sana_1600M_4Kpx_BF16.pth",
    "Efficient-Large-Model/Sana_1600M_2Kpx_BF16/checkpoints/Sana_1600M_2Kpx_BF16.pth",
    "Efficient-Large-Model/Sana_1600M_1024px_MultiLing/checkpoints/Sana_1600M_1024px_MultiLing.pth",
    "Efficient-Large-Model/Sana_1600M_1024px_BF16/checkpoints/Sana_1600M_1024px_BF16.pth",
    "Efficient-Large-Model/Sana_1600M_512px_MultiLing/checkpoints/Sana_1600M_512px_MultiLing.pth",
    "Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth",
    "Efficient-Large-Model/Sana_1600M_512px/checkpoints/Sana_1600M_512px.pth",
    "Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth",
    "Efficient-Large-Model/Sana_600M_512px/checkpoints/Sana_600M_512px_MultiLing.pth",
]
# https://github.com/NVlabs/Sana/blob/main/scripts/inference.py


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
    converted_state_dict["patch_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["patch_embed.proj.bias"] = state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    converted_state_dict["caption_projection.linear_1.weight"] = state_dict.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = state_dict.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = state_dict.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = state_dict.pop("y_embedder.y_proj.fc2.bias")

    # Handle different time embedding structure based on model type

    if args.model_type in ["SanaSprint_1600M_P1_D20", "SanaSprint_600M_P1_D28"]:
        # For Sana Sprint, the time embedding structure is different
        converted_state_dict["time_embed.timestep_embedder.linear_1.weight"] = state_dict.pop(
            "t_embedder.mlp.0.weight"
        )
        converted_state_dict["time_embed.timestep_embedder.linear_1.bias"] = state_dict.pop("t_embedder.mlp.0.bias")
        converted_state_dict["time_embed.timestep_embedder.linear_2.weight"] = state_dict.pop(
            "t_embedder.mlp.2.weight"
        )
        converted_state_dict["time_embed.timestep_embedder.linear_2.bias"] = state_dict.pop("t_embedder.mlp.2.bias")

        # Guidance embedder for Sana Sprint
        converted_state_dict["time_embed.guidance_embedder.linear_1.weight"] = state_dict.pop(
            "cfg_embedder.mlp.0.weight"
        )
        converted_state_dict["time_embed.guidance_embedder.linear_1.bias"] = state_dict.pop("cfg_embedder.mlp.0.bias")
        converted_state_dict["time_embed.guidance_embedder.linear_2.weight"] = state_dict.pop(
            "cfg_embedder.mlp.2.weight"
        )
        converted_state_dict["time_embed.guidance_embedder.linear_2.bias"] = state_dict.pop("cfg_embedder.mlp.2.bias")
    else:
        # Original Sana time embedding structure
        converted_state_dict["time_embed.emb.timestep_embedder.linear_1.weight"] = state_dict.pop(
            "t_embedder.mlp.0.weight"
        )
        converted_state_dict["time_embed.emb.timestep_embedder.linear_1.bias"] = state_dict.pop(
            "t_embedder.mlp.0.bias"
        )
        converted_state_dict["time_embed.emb.timestep_embedder.linear_2.weight"] = state_dict.pop(
            "t_embedder.mlp.2.weight"
        )
        converted_state_dict["time_embed.emb.timestep_embedder.linear_2.bias"] = state_dict.pop(
            "t_embedder.mlp.2.bias"
        )

    # Shared norm.
    converted_state_dict["time_embed.linear.weight"] = state_dict.pop("t_block.1.weight")
    converted_state_dict["time_embed.linear.bias"] = state_dict.pop("t_block.1.bias")

    # y norm
    converted_state_dict["caption_norm.weight"] = state_dict.pop("attention_y_norm.weight")

    # scheduler
    if args.image_size == 4096:
        flow_shift = 6.0
    else:
        flow_shift = 3.0

    # model config
    if args.model_type in ["SanaMS_1600M_P1_D20", "SanaSprint_1600M_P1_D20", "SanaMS1.5_1600M_P1_D20"]:
        layer_num = 20
    elif args.model_type in ["SanaMS_600M_P1_D28", "SanaSprint_600M_P1_D28"]:
        layer_num = 28
    elif args.model_type == "SanaMS_4800M_P1_D60":
        layer_num = 60
    else:
        raise ValueError(f"{args.model_type} is not supported.")
    # Positional embedding interpolation scale.
    interpolation_scale = {512: None, 1024: None, 2048: 1.0, 4096: 2.0}
    qk_norm = (
        "rms_norm_across_heads"
        if args.model_type
        in ["SanaMS1.5_1600M_P1_D20", "SanaMS1.5_4800M_P1_D60", "SanaSprint_600M_P1_D28", "SanaSprint_1600M_P1_D20"]
        else None
    )

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
            "in_channels": 32,
            "out_channels": 32,
            "num_attention_heads": model_kwargs[args.model_type]["num_attention_heads"],
            "attention_head_dim": model_kwargs[args.model_type]["attention_head_dim"],
            "num_layers": model_kwargs[args.model_type]["num_layers"],
            "num_cross_attention_heads": model_kwargs[args.model_type]["num_cross_attention_heads"],
            "cross_attention_head_dim": model_kwargs[args.model_type]["cross_attention_head_dim"],
            "cross_attention_dim": model_kwargs[args.model_type]["cross_attention_dim"],
            "caption_channels": 2304,
            "mlp_ratio": 2.5,
            "attention_bias": False,
            "sample_size": args.image_size // 32,
            "patch_size": 1,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "interpolation_scale": interpolation_scale[args.image_size],
        }

        # Add qk_norm parameter for Sana Sprint
        if args.model_type in [
            "SanaMS1.5_1600M_P1_D20",
            "SanaMS1.5_4800M_P1_D60",
            "SanaSprint_600M_P1_D28",
            "SanaSprint_1600M_P1_D20",
        ]:
            transformer_kwargs["qk_norm"] = "rms_norm_across_heads"
        if args.model_type in ["SanaSprint_1600M_P1_D20", "SanaSprint_600M_P1_D28"]:
            transformer_kwargs["guidance_embeds"] = True

        transformer = SanaTransformer2DModel(**transformer_kwargs)

    if is_accelerate_available():
        load_model_dict_into_meta(transformer, converted_state_dict)
    else:
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
        ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", torch_dtype=torch.float32)

        # Text Encoder
        text_encoder_model_path = "Efficient-Large-Model/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_path)
        tokenizer.padding_side = "right"
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_model_path, torch_dtype=torch.bfloat16
        ).get_decoder()

        # Choose the appropriate pipeline and scheduler based on model type
        if args.model_type in ["SanaSprint_1600M_P1_D20", "SanaSprint_600M_P1_D28"]:
            # Force SCM Scheduler for Sana Sprint regardless of scheduler_type
            if args.scheduler_type != "scm":
                print(
                    colored(
                        f"Warning: Overriding scheduler_type '{args.scheduler_type}' to 'scm' for SanaSprint model",
                        "yellow",
                        attrs=["bold"],
                    )
                )

            # SCM Scheduler for Sana Sprint
            scheduler_config = {
                "prediction_type": "trigflow",
                "sigma_data": 0.5,
            }
            scheduler = SCMScheduler(**scheduler_config)
            pipe = SanaSprintPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                vae=ae,
                scheduler=scheduler,
            )
        else:
            # Original Sana scheduler
            if args.scheduler_type == "flow-dpm_solver":
                scheduler = DPMSolverMultistepScheduler(
                    flow_shift=flow_shift,
                    use_flow_sigmas=True,
                    prediction_type="flow_prediction",
                )
            elif args.scheduler_type == "flow-euler":
                scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
            else:
                raise ValueError(f"Scheduler type {args.scheduler_type} is not supported")

            pipe = SanaPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                vae=ae,
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
        "--image_size",
        default=1024,
        type=int,
        choices=[512, 1024, 2048, 4096],
        required=False,
        help="Image size of pretrained model, 512, 1024, 2048 or 4096.",
    )
    parser.add_argument(
        "--model_type",
        default="SanaMS_1600M_P1_D20",
        type=str,
        choices=[
            "SanaMS_1600M_P1_D20",
            "SanaMS_600M_P1_D28",
            "SanaMS1.5_1600M_P1_D20",
            "SanaMS1.5_4800M_P1_D60",
            "SanaSprint_1600M_P1_D20",
            "SanaSprint_600M_P1_D28",
        ],
    )
    parser.add_argument(
        "--scheduler_type",
        default="flow-dpm_solver",
        type=str,
        choices=["flow-dpm_solver", "flow-euler", "scm"],
        help="Scheduler type to use. Use 'scm' for Sana Sprint models.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--save_full_pipeline", action="store_true", help="save all the pipeline elements in one.")
    parser.add_argument("--dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="Weight dtype.")

    args = parser.parse_args()

    model_kwargs = {
        "SanaMS_1600M_P1_D20": {
            "num_attention_heads": 70,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "num_layers": 20,
        },
        "SanaMS_600M_P1_D28": {
            "num_attention_heads": 36,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 16,
            "cross_attention_head_dim": 72,
            "cross_attention_dim": 1152,
            "num_layers": 28,
        },
        "SanaMS1.5_1600M_P1_D20": {
            "num_attention_heads": 70,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "num_layers": 20,
        },
        "SanaMS1.5_4800M_P1_D60": {
            "num_attention_heads": 70,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "num_layers": 60,
        },
        "SanaSprint_600M_P1_D28": {
            "num_attention_heads": 36,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 16,
            "cross_attention_head_dim": 72,
            "cross_attention_dim": 1152,
            "num_layers": 28,
        },
        "SanaSprint_1600M_P1_D20": {
            "num_attention_heads": 70,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "num_layers": 20,
        },
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = DTYPE_MAPPING[args.dtype]

    main(args)
