"""Convert an original Mage-Flow HF repo layout into diffusers-format weights.

Example
-------

    python scripts/convert_mage_flow_to_diffusers.py \
        --input_dir  path/to/Mage-Flow-Base \
        --output_dir path/to/Mage-Flow-Base-diffusers \
        --dtype bfloat16
"""

import argparse
import json
import os
import shutil
from typing import Dict

import safetensors.torch
import torch


# ---------------------------------------------------------------------------
# Transformer conversion
# ---------------------------------------------------------------------------

# Top-level 1:1 renames. Anything not matched here (transformer_blocks.*, the
# time_text_embed / norm_out / proj_out subtrees) is passed through unchanged.
TRANSFORMER_TOP_LEVEL_RENAMES: Dict[str, str] = {
    "img_in.weight": "x_embedder.weight",
    "img_in.bias": "x_embedder.bias",
    "txt_norm.weight": "context_embedder_norm.weight",
    "txt_in.weight": "context_embedder.weight",
    "txt_in.bias": "context_embedder.bias",
}


TRANSFORMER_DIFFUSERS_CONFIG = {
    "_class_name": "MageFlowTransformer2DModel",
    "_diffusers_version": "0.37.0",
    "in_channels": 128,
    "out_channels": 128,
    "context_in_dim": 2560,
    "hidden_size": 3072,
    "num_attention_heads": 24,
    "num_layers": 12,
    "axes_dim": [16, 56, 56],
    "patch_size": 1,
}


def convert_transformer_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key in TRANSFORMER_TOP_LEVEL_RENAMES:
            new_key = TRANSFORMER_TOP_LEVEL_RENAMES[key]
        else:
            # Pass-through: transformer_blocks.*, time_text_embed.*, norm_out.*, proj_out.*
            new_key = key
        if new_key in new_state_dict:
            raise ValueError(f"Duplicate destination key while converting transformer: {new_key}")
        new_state_dict[new_key] = tensor
    return new_state_dict


# ---------------------------------------------------------------------------
# VAE conversion
# ---------------------------------------------------------------------------

VAE_ENCODER_PREFIX = "student.dconv_encoder."
VAE_DECODER_PREFIX = "pipeline."

# Sub-trees of the original decoder ("pipeline.*") that belong to the Flux2
# encoder and must be dropped instead of being mapped into the diffusers
# decoder namespace.
VAE_DECODER_EXCLUDE_PREFIXES = (
    "pipeline.y_embedder.encoder.",
    "pipeline.y_embedder.bottleneck.",
)


VAE_DIFFUSERS_CONFIG = {
    "_class_name": "AutoencoderMageVAE",
    "_diffusers_version": "0.37.0",
    "latent_channels": 128,
    "downsample_factor": 16,
    "encoder_hidden_size": 384,
    "encoder_num_blocks": 21,
    "encoder_patch_size": 16,
    "encoder_head_size": 768,
    "encoder_num_head_blocks": 2,
    "decoder_hidden_size": 384,
    "decoder_hidden_size_x": 32,
    "decoder_num_blocks": 24,
    "decoder_num_cond_blocks": 21,
    "decoder_bottleneck_dim": 128,
    "decoder_patch_size": 16,
    "sample_posterior": False,
}


def convert_vae_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith(VAE_ENCODER_PREFIX):
            new_key = "encoder." + key[len(VAE_ENCODER_PREFIX):]
        elif key.startswith(VAE_DECODER_PREFIX):
            if any(key.startswith(p) for p in VAE_DECODER_EXCLUDE_PREFIXES):
                continue
            # pipeline.y_embedder.decoder.* naturally maps to
            # decoder.y_embedder.decoder.* under this rule, matching the spec.
            new_key = "decoder." + key[len(VAE_DECODER_PREFIX):]
        else:
            raise ValueError(f"Unexpected VAE key with no known prefix: {key}")
        if new_key in new_state_dict:
            raise ValueError(f"Duplicate destination key while converting VAE: {new_key}")
        new_state_dict[new_key] = tensor
    return new_state_dict


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def cast_state_dict(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        # Leave integer / bool buffers alone (e.g. num_batches_tracked).
        if tensor.is_floating_point():
            out[key] = tensor.to(dtype)
        else:
            out[key] = tensor
    return out


def save_component(
    state_dict: Dict[str, torch.Tensor],
    config: Dict,
    output_component_dir: str,
) -> None:
    os.makedirs(output_component_dir, exist_ok=True)
    safetensors.torch.save_file(
        state_dict,
        os.path.join(output_component_dir, "diffusion_pytorch_model.safetensors"),
    )
    with open(os.path.join(output_component_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def copy_scheduler(input_dir: str, output_dir: str) -> None:
    src = os.path.join(input_dir, "scheduler", "scheduler_config.json")
    dst_dir = os.path.join(output_dir, "scheduler")
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src, os.path.join(dst_dir, "scheduler_config.json"))


def copy_text_encoder(input_dir: str, output_dir: str, symlink: bool) -> None:
    src = os.path.join(input_dir, "text_encoder")
    dst = os.path.join(output_dir, "text_encoder")
    if os.path.lexists(dst):
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copytree(src, dst)


MODEL_INDEX = {
    "_class_name": "MageFlowPipeline",
    "_diffusers_version": "0.37.0",
    "transformer": ["diffusers", "MageFlowTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderMageVAE"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "tokenizer": ["transformers", "AutoTokenizer"],
}


def write_model_index(output_dir: str) -> None:
    with open(os.path.join(output_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(MODEL_INDEX, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Original Mage-Flow HF repo path.")
    parser.add_argument("--output_dir", required=True, help="Destination diffusers-format directory.")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_MAP.keys()), help="Output tensor dtype.")
    parser.add_argument(
        "--text_encoder_mode",
        default="symlink",
        choices=["symlink", "copy"],
        help="How to include the text_encoder directory in the output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Transformer ---
    print("[transformer] loading original weights ...")
    transformer_sd = safetensors.torch.load_file(
        os.path.join(args.input_dir, "transformer", "diffusion_pytorch_model.safetensors")
    )
    print(f"[transformer] converting {len(transformer_sd)} tensors ...")
    transformer_sd = convert_transformer_state_dict(transformer_sd)
    transformer_sd = cast_state_dict(transformer_sd, dtype)
    save_component(
        transformer_sd,
        TRANSFORMER_DIFFUSERS_CONFIG,
        os.path.join(args.output_dir, "transformer"),
    )
    print(f"[transformer] wrote {len(transformer_sd)} tensors to {args.output_dir}/transformer")
    del transformer_sd

    # --- VAE ---
    print("[vae] loading original weights ...")
    vae_sd = safetensors.torch.load_file(
        os.path.join(args.input_dir, "vae", "diffusion_pytorch_model.safetensors")
    )
    print(f"[vae] converting {len(vae_sd)} tensors ...")
    vae_sd = convert_vae_state_dict(vae_sd)
    vae_sd = cast_state_dict(vae_sd, dtype)
    save_component(
        vae_sd,
        VAE_DIFFUSERS_CONFIG,
        os.path.join(args.output_dir, "vae"),
    )
    print(f"[vae] wrote {len(vae_sd)} tensors to {args.output_dir}/vae")
    del vae_sd

    # --- Scheduler ---
    print("[scheduler] copying config ...")
    copy_scheduler(args.input_dir, args.output_dir)

    # --- Text encoder ---
    print(f"[text_encoder] {args.text_encoder_mode} ...")
    copy_text_encoder(args.input_dir, args.output_dir, symlink=(args.text_encoder_mode == "symlink"))

    # --- model_index.json ---
    write_model_index(args.output_dir)
    print(f"Done. Diffusers-format repo written to: {args.output_dir}")


if __name__ == "__main__":
    main()
