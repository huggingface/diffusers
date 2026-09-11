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
    AutoencoderKLLTX2Video,
    AutoencoderKLWan,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaVideoPipeline,
    SanaVideoTransformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext

ckpt_ids = [
    "Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth",
    "Efficient-Large-Model/SANA-Video_2B_720p/checkpoints/SANA_Video_2B_720p_LTXVAE.pth",
]
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
    # scheduler
    flow_shift = 8.0
    if args.task == "i2v":
        assert args.scheduler_type == "flow-euler", "Scheduler type must be flow-euler for i2v task."

    # model config
    # Positional embedding interpolation scale.

    # sample size
    if args.video_size == 480:
        sample_size = 30  # Wan-VAE: 8xp2 downsample factor
        patch_size = (1, 2, 2)
        in_channels = 16
        out_channels = 16
    elif args.video_size == 720:
        sample_size = 22  # DC-AE-V: 32xp1 downsample factor
        patch_size = (1, 1, 1)
        in_channels = 32
        out_channels = 32
    else:
        raise ValueError(f"Video size {args.video_size} is not supported.")

    if args.vae_type == "ltx2":
        sample_size = 22
        patch_size = (1, 1, 1)
        in_channels = 128
        out_channels = 128

    # Transformer
    with CTX():
        transformer_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
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

    converted_state_dict = convert_component_checkpoint(
        state_dict, dict(transformer.config), "SanaVideoTransformer3DModel"
    )
    transformer.load_state_dict(converted_state_dict, strict=True, assign=True)

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
        if args.vae_type == "ltx2":
            vae_path = args.vae_path or "Lightricks/LTX-2"
            vae = AutoencoderKLLTX2Video.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.float32)
        else:
            vae_path = args.vae_path or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            vae = AutoencoderKLWan.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.float32)

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
    parser.add_argument(
        "--vae_type",
        default="wan",
        type=str,
        choices=["wan", "ltx2"],
        help="VAE type to use for saving full pipeline (ltx2 uses patchify 1x1x1).",
    )
    parser.add_argument(
        "--vae_path",
        default=None,
        type=str,
        required=False,
        help="Optional VAE path or repo id. If not set, a default is used per VAE type.",
    )
    parser.add_argument(
        "--task", default="t2v", type=str, required=True, choices=["t2v", "i2v"], help="Task to convert, t2v or i2v."
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--save_full_pipeline", action="store_true", help="save all the pipeline elements in one.")
    parser.add_argument("--dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="Weight dtype.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = DTYPE_MAPPING[args.dtype]

    main(args)
