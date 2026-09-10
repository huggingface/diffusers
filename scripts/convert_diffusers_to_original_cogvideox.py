#!/usr/bin/env python3
"""
Script for converting a HuggingFace Diffusers CogVideoX pipeline
back to the original CogVideoX checkpoint format.

This reverses the transformations done by convert_cogvideox_to_diffusers.py.
"""

import argparse
import os
from typing import Any, Dict

import torch
from safetensors.torch import save_file as safetensors_save_file


# ============================#
# Transformer Key Rename Map  #
# ============================#

# Reverse of TRANSFORMER_KEYS_RENAME_DICT
TRANSFORMER_KEYS_REVERSE_MAP = {
    "norm_final": "transformer.final_layernorm",
    "transformer_blocks": "transformer",
    "attn1": "attention",
    "ff.net": "mlp",
    "0.proj": "dense_h_to_4h",
    "2": "dense_4h_to_h",
    "to_out.0": "dense",
    "norm1.norm": "input_layernorm",
    "norm2.norm": "post_attn1_layernorm",
    "time_embedding.linear_1": "time_embed.0",
    "time_embedding.linear_2": "time_embed.2",
    "ofs_embedding.linear_1": "ofs_embed.0",
    "ofs_embedding.linear_2": "ofs_embed.2",
    "patch_embed": "mixins.patch_embed",
    "norm_out.norm": "mixins.final_layer.norm_final",
    "proj_out": "mixins.final_layer.linear",
    "norm_out.linear": "mixins.final_layer.adaLN_modulation.1",
}

# Keys that need special handling
TRANSFORMER_KEYS_REVERSE_MAP_SPECIAL = {
    "patch_embed.pos_embedding": "mixins.pos_embed.pos_embedding",
}


def merge_qkv_to_query_key_value(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge to_q, to_k, to_v back into query_key_value (reverse of chunk)."""
    keys_to_merge = {}
    for key in list(state_dict.keys()):
        if ".to_q." in key:
            base = key.replace(".to_q.", ".query_key_value.")
            keys_to_merge.setdefault(base, {})["q"] = key
        elif ".to_k." in key:
            base = key.replace(".to_k.", ".query_key_value.")
            keys_to_merge.setdefault(base, {})["k"] = key
        elif ".to_v." in key:
            base = key.replace(".to_v.", ".query_key_value.")
            keys_to_merge.setdefault(base, {})["v"] = key

    for merged_key, parts in keys_to_merge.items():
        if len(parts) == 3:
            q = state_dict.pop(parts["q"])
            k = state_dict.pop(parts["k"])
            v = state_dict.pop(parts["v"])
            state_dict[merged_key] = torch.cat([q, k, v], dim=0)
    return state_dict


def merge_norm_qk_to_layernorm_list(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge norm_q and norm_k back into query_layernorm_list / key_layernorm_list."""
    norm_q_keys = {}
    norm_k_keys = {}

    for key in list(state_dict.keys()):
        if "attn1.norm_q." in key:
            parts = key.split(".")
            layer_id = parts[1]
            wb = parts[-1]
            norm_q_keys.setdefault(layer_id, {})[wb] = key
        elif "attn1.norm_k." in key:
            parts = key.split(".")
            layer_id = parts[1]
            wb = parts[-1]
            norm_k_keys.setdefault(layer_id, {})[wb] = key

    for layer_id, wb_map in norm_q_keys.items():
        for wb, key in wb_map.items():
            new_key = f"transformer.layers.{layer_id}.query_layernorm_list.{wb}"
            state_dict[new_key] = state_dict.pop(key)

    for layer_id, wb_map in norm_k_keys.items():
        for wb, key in wb_map.items():
            new_key = f"transformer.layers.{layer_id}.key_layernorm_list.{wb}"
            state_dict[new_key] = state_dict.pop(key)

    return state_dict


def merge_norm1_norm2_to_adaln(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge norm1.linear and norm2.linear back into adaLN_layer.adaLN_modulations."""
    norm1_keys = {}
    norm2_keys = {}

    for key in list(state_dict.keys()):
        if "norm1.linear." in key:
            parts = key.split(".")
            layer_id = parts[1]
            wb = parts[-1]
            norm1_keys.setdefault(layer_id, {})[wb] = key
        elif "norm2.linear." in key:
            parts = key.split(".")
            layer_id = parts[1]
            wb = parts[-1]
            norm2_keys.setdefault(layer_id, {})[wb] = key

    for layer_id in norm1_keys:
        if layer_id in norm2_keys:
            wb = list(norm1_keys[layer_id].keys())[0]
            wb2 = list(norm2_keys[layer_id].keys())[0]
            if wb != wb2:
                continue
            w1 = state_dict.pop(norm1_keys[layer_id][wb])
            w2 = state_dict.pop(norm2_keys[layer_id][wb])
            c1, c2, c3, c4, c5, c6 = w1.chunk(6, dim=0)
            c7, c8, c9, c10, c11, c12 = w2.chunk(6, dim=0)
            merged = torch.cat([c1, c2, c3, c7, c8, c9, c4, c5, c6, c10, c11, c12], dim=0)
            new_key = f"transformer.layers.{layer_id}.adaln_layer.adaLN_modulations.{wb}"
            state_dict[new_key] = merged

    return state_dict


def reverse_transformer_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply reverse key renaming for transformer."""
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_key = key
        for hf_name, og_name in TRANSFORMER_KEYS_REVERSE_MAP_SPECIAL.items():
            new_key = new_key.replace(hf_name, og_name)
        for hf_name, og_name in sorted(TRANSFORMER_KEYS_REVERSE_MAP.items(), key=lambda x: -len(x[0])):
            new_key = new_key.replace(hf_name, og_name)
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def convert_transformer_to_original(
    hf_pipeline_path: str,
    dtype: torch.dtype,
    output_path: str,
) -> str:
    """Load HF transformer, reverse key mappings, save as original format."""
    from diffusers import CogVideoXTransformer3DModel

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        hf_pipeline_path, subfolder="transformer", torch_dtype=dtype
    )
    state_dict = transformer.state_dict()

    state_dict = merge_qkv_to_query_key_value(state_dict)
    state_dict = merge_norm_qk_to_layernorm_list(state_dict)
    state_dict = merge_norm1_norm2_to_adaln(state_dict)
    state_dict = reverse_transformer_keys(state_dict)

    prefixed_state_dict = {}
    for key, value in state_dict.items():
        prefixed_state_dict[f"model.diffusion_model.{key}"] = value.contiguous()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    safetensors_save_file(prefixed_state_dict, output_path, metadata={"format": "pt"})
    print(f"Saved transformer checkpoint to {output_path}")
    return output_path


# ====================#
# VAE Key Rename Map  #
# ====================#

VAE_KEYS_REVERSE_MAP = [
    ("resnets.", "block."),
    ("down_blocks.", "down."),
    ("downsamplers.0", "downsample"),
    ("upsamplers.0", "upsample"),
    ("conv_shortcut", "nin_shortcut"),
    ("encoder.mid_block.resnets.0", "encoder.mid.block_1"),
    ("encoder.mid_block.resnets.1", "encoder.mid.block_2"),
    ("decoder.mid_block.resnets.0", "decoder.mid.block_1"),
    ("decoder.mid_block.resnets.1", "decoder.mid.block_2"),
]


def reverse_up_blocks(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse the up_blocks index transformation."""
    for key in list(state_dict.keys()):
        if key.startswith("decoder.up_blocks."):
            parts = key.split(".")
            old_idx = int(parts[2])
            new_idx = 4 - 1 - old_idx
            parts[2] = str(new_idx)
            new_key = ".".join(parts)
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def reverse_vae_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply reverse key renaming for VAE."""
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_key = key
        for hf_name, og_name in sorted(VAE_KEYS_REVERSE_MAP, key=lambda x: -len(x[0])):
            new_key = new_key.replace(hf_name, og_name)
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def convert_vae_to_original(
    hf_pipeline_path: str,
    dtype: torch.dtype,
    output_path: str,
) -> str:
    """Load HF VAE, reverse key mappings, save as original format."""
    from diffusers import AutoencoderKLCogVideoX

    vae = AutoencoderKLCogVideoX.from_pretrained(hf_pipeline_path, subfolder="vae", torch_dtype=dtype)
    state_dict = vae.state_dict()

    state_dict = reverse_up_blocks(state_dict)
    state_dict = reverse_vae_keys(state_dict)

    final_state_dict = {"model": state_dict}

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(final_state_dict, output_path)
    print(f"Saved VAE checkpoint to {output_path}")
    return output_path


# =============#
# Main Entry   #
# =============#


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert a HF Diffusers CogVideoX pipeline back to original CogVideoX format."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the HF Diffusers pipeline directory (parent of subfolders like 'transformer', 'vae').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory path where converted checkpoints will be saved.",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Whether to save in fp16."
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="Whether to save in bf16."
    )
    parser.add_argument(
        "--convert_transformer",
        action="store_true",
        default=True,
        help="Convert the transformer (default: True).",
    )
    parser.add_argument(
        "--convert_vae",
        action="store_true",
        default=True,
        help="Convert the VAE (default: True).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.fp16 and args.bf16:
        raise ValueError("You cannot pass both --fp16 and --bf16 at the same time.")

    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    os.makedirs(args.output_path, exist_ok=True)

    if args.convert_transformer:
        transformer_out = os.path.join(args.output_path, "transformer.safetensors")
        convert_transformer_to_original(args.checkpoint_path, dtype, transformer_out)

    if args.convert_vae:
        vae_out = os.path.join(args.output_path, "vae.pt")
        convert_vae_to_original(args.checkpoint_path, torch.float32, vae_out)

    print(f"\nDone! Converted checkpoints saved to {args.output_path}")
    print(f"  - Transformer: {os.path.join(args.output_path, 'transformer.safetensors')}")
    print(f"  - VAE: {os.path.join(args.output_path, 'vae.pt')}")
    print("\nNote: The text encoder (T5) is not converted -- it can be loaded directly")
    print("from HuggingFace as 'google/t5-v1_1-xxl'.")
