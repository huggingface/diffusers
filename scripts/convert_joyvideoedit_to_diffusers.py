"""Convert JoyVideoEdit (JoyAI-Video-Edit) checkpoints to diffusers format.

Converts the transformer and/or the VAE. The transformer checkpoint is a raw `.pth` state dict whose double-block
attention keys need remapping under `.attn.`. The VAE checkpoint uses diffusers-format safetensors and is re-saved with
the `AutoencoderKLJoyVideoEdit` configuration.

Usage:

```bash
python scripts/convert_joyvideoedit_to_diffusers.py \
    --transformer_ckpt_path /path/to/joyai_video_edit_dit_0804.pth \
    --vae_dir /path/to/JoyAI-Video-Edit/vae \
    --output_path /path/to/output
```
"""

import argparse
import json
import os

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from diffusers import (
    AutoencoderKLJoyVideoEdit,
    FlowMatchEulerDiscreteScheduler,
    JoyVideoEditPipeline,
    JoyVideoEditTransformer3DModel,
)


TRANSFORMER_CONFIG = {
    "patch_size": [1, 1, 1],
    "in_channels": 64,
    "out_channels": 64,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "text_dim": 4096,
    "num_layers": 40,
    "rope_dim_list": [16, 56, 56],
    "theta": 256,
    "chunk_size": 1,
    "local_window_size": 3,
    "global_sink_chunk": True,
    "source_id_rope_dim": 128,
    "source_id_rope_theta": 256.0,
}


def convert_transformer(ckpt_path: str) -> JoyVideoEditTransformer3DModel:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    original_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    attn_suffixes = (
        "img_attn_qkv.",
        "img_attn_q_norm.",
        "img_attn_k_norm.",
        "img_attn_proj.",
        "txt_attn_qkv.",
        "txt_attn_q_norm.",
        "txt_attn_k_norm.",
        "txt_attn_proj.",
    )
    remapped = {}
    for key, value in original_state_dict.items():
        new_key = key
        if key.startswith("double_blocks."):
            for suffix in attn_suffixes:
                if "." + suffix in key and ".attn." + suffix not in key:
                    new_key = key.replace("." + suffix, ".attn." + suffix)
                    break
        remapped[new_key] = value

    with init_empty_weights():
        transformer = JoyVideoEditTransformer3DModel(**TRANSFORMER_CONFIG)
    transformer.load_state_dict(remapped, strict=True, assign=True)
    return transformer


def convert_vae(vae_dir: str) -> AutoencoderKLJoyVideoEdit:
    with open(os.path.join(vae_dir, "config.json")) as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    state_dict = load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"))

    with init_empty_weights():
        vae = AutoencoderKLJoyVideoEdit(**config)
    vae.load_state_dict(state_dict, strict=True, assign=True)
    return vae


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_args():
    parser = argparse.ArgumentParser(description="Convert JoyVideoEdit checkpoints to diffusers format")
    parser.add_argument(
        "--transformer_ckpt_path",
        type=str,
        default=None,
        help="Path to the transformer checkpoint (e.g. joyai_video_edit_dit_0804.pth)",
    )
    parser.add_argument(
        "--vae_dir",
        type=str,
        default=None,
        help="Path to the VAE directory (with config.json + diffusion_pytorch_model.safetensors)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=(
            "Output directory. Saves a complete pipeline when both checkpoints are provided, or an individual "
            "transformer/ or vae/ subdirectory otherwise."
        ),
    )
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAPPING), default="bf16", help="Torch dtype")
    return parser.parse_args()


def set_model_dtype(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    torch.nn.Module.to(model, dtype=dtype)
    keep_in_fp32_modules = getattr(model, "_keep_in_fp32_modules", None) or []
    for module_name, module in model.named_modules():
        if any(pattern in module_name.split(".") for pattern in keep_in_fp32_modules):
            torch.nn.Module.to(module, dtype=torch.float32)
    return model


def save_joyvideoedit_pipeline(
    transformer: JoyVideoEditTransformer3DModel,
    vae: AutoencoderKLJoyVideoEdit,
    output_path: str,
) -> None:
    pipeline = JoyVideoEditPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        processor=None,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipeline.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")


if __name__ == "__main__":
    args = get_args()
    dtype = DTYPE_MAPPING[args.dtype]

    transformer = None
    vae = None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path)
        transformer = set_model_dtype(transformer, dtype)

    if args.vae_dir is not None:
        vae = convert_vae(args.vae_dir)
        vae = set_model_dtype(vae, dtype)

    if transformer is not None and vae is not None:
        save_joyvideoedit_pipeline(transformer, vae, args.output_path)
    elif transformer is not None:
        transformer.save_pretrained(
            os.path.join(args.output_path, "transformer"), safe_serialization=True, max_shard_size="5GB"
        )
    elif vae is not None:
        vae.save_pretrained(os.path.join(args.output_path, "vae"), safe_serialization=True)
    else:
        raise ValueError("Provide at least one of `--transformer_ckpt_path` or `--vae_dir`.")
