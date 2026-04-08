"""Convert CogVideoX models from diffusers format back to the original format.

This script reverses the conversion performed by `convert_cogvideox_to_diffusers.py`,
enabling users to export diffusers-format CogVideoX models (transformer and VAE) back
to the original checkpoint format used by the CogVideo codebase.

Usage examples:
    # Convert transformer only
    python scripts/convert_cogvideox_to_original.py \
        --diffusers_model_path THUDM/CogVideoX-2b \
        --output_path ./cogvideox_original/transformer.pt \
        --component transformer

    # Convert VAE only
    python scripts/convert_cogvideox_to_original.py \
        --diffusers_model_path THUDM/CogVideoX-2b \
        --output_path ./cogvideox_original/vae.pt \
        --component vae

    # Use fp16 precision
    python scripts/convert_cogvideox_to_original.py \
        --diffusers_model_path THUDM/CogVideoX-2b \
        --output_path ./cogvideox_original/transformer.pt \
        --component transformer --fp16
"""

import argparse
from typing import Any, Dict

import torch

from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel


# Reverse of TRANSFORMER_KEYS_RENAME_DICT from convert_cogvideox_to_diffusers.py.
# Note: ordering matters — longer/more-specific targets must be matched first to avoid
# partial replacements by shorter keys (e.g. "norm1.norm" before "norm1").
TRANSFORMER_KEYS_RENAME_DICT_REVERSE = {
    "norm_final": "transformer.final_layernorm",
    "attn1": "attention",
    "ff.net": "mlp",
    "0.proj": "dense_h_to_4h",
    "norm1.norm": "input_layernorm",
    "norm2.norm": "post_attn1_layernorm",
    "to_out.0": "dense",
    "time_embedding.linear_1": "time_embed.0",
    "time_embedding.linear_2": "time_embed.2",
    "ofs_embedding.linear_1": "ofs_embed.0",
    "ofs_embedding.linear_2": "ofs_embed.2",
    "patch_embed.pos_embedding": "mixins.pos_embed.pos_embedding",
    "norm_out.norm": "mixins.final_layer.norm_final",
    "proj_out": "mixins.final_layer.linear",
    "norm_out.linear": "mixins.final_layer.adaLN_modulation.1",
    "patch_embed": "mixins.patch_embed",
    "transformer_blocks": "transformer",
}

# Reverse of VAE_KEYS_RENAME_DICT from convert_cogvideox_to_diffusers.py.
VAE_KEYS_RENAME_DICT_REVERSE = {
    "resnets.": "block.",
    "down_blocks.": "down.",
    "downsamplers.0": "downsample",
    "upsamplers.0": "upsample",
    "conv_shortcut": "nin_shortcut",
    "encoder.mid_block.resnets.0": "encoder.mid.block_1",
    "encoder.mid_block.resnets.1": "encoder.mid.block_2",
    "decoder.mid_block.resnets.0": "decoder.mid.block_1",
    "decoder.mid_block.resnets.1": "decoder.mid.block_2",
}


def merge_qkv_inplace(state_dict: Dict[str, Any], prefix: str = ""):
    """Merge separate to_q, to_k, to_v weights back into a single query_key_value tensor."""
    keys_to_delete = []
    qkv_groups: Dict[str, Dict[str, torch.Tensor]] = {}

    for key in list(state_dict.keys()):
        for suffix in (".to_q.", ".to_k.", ".to_v."):
            if suffix in key:
                # Build the base key that groups q/k/v together
                base = key.replace(suffix, ".PLACEHOLDER.")
                qkv_groups.setdefault(base, {})[suffix] = key
                break

    for base, parts in qkv_groups.items():
        if len(parts) != 3:
            continue
        q_key = parts[".to_q."]
        k_key = parts[".to_k."]
        v_key = parts[".to_v."]
        merged = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)

        new_key = q_key.replace(".to_q.", ".query_key_value.")
        state_dict[new_key] = merged
        keys_to_delete.extend([q_key, k_key, v_key])

    for k in keys_to_delete:
        state_dict.pop(k, None)


def unmerge_adaln_norm_inplace(state_dict: Dict[str, Any]):
    """Reverse the adaln norm split: merge norm1.linear and norm2.linear back into adaln_layer.adaLN_modulations."""
    norm1_keys = {}
    norm2_keys = {}

    for key in list(state_dict.keys()):
        if ".norm1.linear." in key:
            layer_id = key.split("transformer_blocks.")[1].split(".")[0]
            wb = key.split(".")[-1]  # weight or bias
            norm1_keys[(layer_id, wb)] = key
        elif ".norm2.linear." in key:
            layer_id = key.split("transformer_blocks.")[1].split(".")[0]
            wb = key.split(".")[-1]
            norm2_keys[(layer_id, wb)] = key

    for (layer_id, wb), norm1_key in norm1_keys.items():
        norm2_key = norm2_keys.get((layer_id, wb))
        if norm2_key is None:
            continue

        # Forward split: chunks[0:3]+chunks[6:9] -> norm1, chunks[3:6]+chunks[9:12] -> norm2
        # Reverse: interleave them back to the original 12-chunk order
        norm1_val = state_dict.pop(norm1_key)
        norm2_val = state_dict.pop(norm2_key)

        chunk_size = norm1_val.shape[0] // 6
        n1_chunks = norm1_val.chunk(6, dim=0)  # originally [0,1,2,6,7,8]
        n2_chunks = norm2_val.chunk(6, dim=0)  # originally [3,4,5,9,10,11]

        # Reconstruct original order: 0,1,2,3,4,5,6,7,8,9,10,11
        original = torch.cat(
            [
                n1_chunks[0], n1_chunks[1], n1_chunks[2],  # chunks 0-2
                n2_chunks[0], n2_chunks[1], n2_chunks[2],  # chunks 3-5
                n1_chunks[3], n1_chunks[4], n1_chunks[5],  # chunks 6-8
                n2_chunks[3], n2_chunks[4], n2_chunks[5],  # chunks 9-11
            ],
            dim=0,
        )

        adaln_key = f"transformer_blocks.{layer_id}.adaln_layer.adaLN_modulations.{wb}"
        state_dict[adaln_key] = original


def unmerge_qk_layernorm_inplace(state_dict: Dict[str, Any]):
    """Reverse the query/key layernorm reassignment."""
    keys_to_process = []
    for key in list(state_dict.keys()):
        if ".attn1.norm_q." in key or ".attn1.norm_k." in key:
            keys_to_process.append(key)

    for key in keys_to_process:
        weight_or_bias = key.split(".")[-1]
        layer_id = key.split("transformer_blocks.")[1].split(".")[0]

        if ".norm_q." in key:
            new_key = f"transformer_blocks.{layer_id}.query_layernorm_list.{layer_id}.{weight_or_bias}"
        else:
            new_key = f"transformer_blocks.{layer_id}.key_layernorm_list.{layer_id}.{weight_or_bias}"

        state_dict[new_key] = state_dict.pop(key)


def replace_up_blocks_reverse_inplace(state_dict: Dict[str, Any]):
    """Reverse the up_blocks index reversal back to the original 'up.' prefix with inverted indices."""
    for key in list(state_dict.keys()):
        if ".up_blocks." not in key:
            continue
        key_split = key.split(".")
        idx = key_split.index("up_blocks")
        layer_index = int(key_split[idx + 1])
        replace_layer_index = 4 - 1 - layer_index

        key_split[idx] = "up"
        key_split[idx + 1] = str(replace_layer_index)
        new_key = ".".join(key_split)
        state_dict[new_key] = state_dict.pop(key)


def convert_transformer_to_original(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a diffusers CogVideoXTransformer3DModel state dict to original format."""
    # Step 1: Handle special keys first (reverse the special remap operations)
    merge_qkv_inplace(state_dict)
    unmerge_adaln_norm_inplace(state_dict)
    unmerge_qk_layernorm_inplace(state_dict)

    # Step 2: Reverse the simple key renames.
    # We must apply longer replacements first to avoid partial matches.
    sorted_renames = sorted(TRANSFORMER_KEYS_RENAME_DICT_REVERSE.items(), key=lambda x: -len(x[0]))

    for key in list(state_dict.keys()):
        new_key = key
        for diffusers_pattern, original_pattern in sorted_renames:
            new_key = new_key.replace(diffusers_pattern, original_pattern)
        # Restore ".layers" which was stripped as ".layers" -> ""
        # In the forward conversion, ".layers" was removed from keys like "transformer.layers.X..."
        # After reversing "transformer_blocks" -> "transformer", we need to re-insert ".layers"
        if new_key.startswith("transformer.") and not new_key.startswith("transformer.final_layernorm"):
            parts = new_key.split(".", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                new_key = f"transformer.layers.{parts[1]}" + ("." + parts[2] if len(parts) > 2 else "")

        # Add back the prefix that was stripped during forward conversion
        new_key = "model.diffusion_model." + new_key
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)
        elif not key.startswith("model.diffusion_model."):
            state_dict["model.diffusion_model." + key] = state_dict.pop(key)

    return state_dict


def convert_vae_to_original(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a diffusers AutoencoderKLCogVideoX state dict to original format."""
    # Step 1: Handle up_blocks index reversal
    replace_up_blocks_reverse_inplace(state_dict)

    # Step 2: Reverse simple key renames (longer patterns first)
    sorted_renames = sorted(VAE_KEYS_RENAME_DICT_REVERSE.items(), key=lambda x: -len(x[0]))

    for key in list(state_dict.keys()):
        new_key = key
        for diffusers_pattern, original_pattern in sorted_renames:
            new_key = new_key.replace(diffusers_pattern, original_pattern)
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)

    return state_dict


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert CogVideoX models from diffusers format to the original CogVideo checkpoint format."
    )
    parser.add_argument(
        "--diffusers_model_path",
        type=str,
        required=True,
        help="Path to a local diffusers CogVideoX pipeline directory, or a Hugging Face Hub model ID.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted original-format checkpoint (.pt).",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["transformer", "vae"],
        required=True,
        help="Which component to convert: 'transformer' or 'vae'.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Save the checkpoint in fp16 precision.")
    parser.add_argument("--bf16", action="store_true", default=False, help="Save the checkpoint in bf16 precision.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.fp16 and args.bf16:
        raise ValueError("You cannot pass both --fp16 and --bf16 at the same time.")

    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    if args.component == "transformer":
        print(f"Loading CogVideoXTransformer3DModel from {args.diffusers_model_path} ...")
        model = CogVideoXTransformer3DModel.from_pretrained(args.diffusers_model_path, subfolder="transformer")
        state_dict = {k: v.to(dtype) for k, v in model.state_dict().items()}
        print("Converting transformer state dict to original format ...")
        state_dict = convert_transformer_to_original(state_dict)
    elif args.component == "vae":
        print(f"Loading AutoencoderKLCogVideoX from {args.diffusers_model_path} ...")
        model = AutoencoderKLCogVideoX.from_pretrained(args.diffusers_model_path, subfolder="vae")
        state_dict = {k: v.to(dtype) for k, v in model.state_dict().items()}
        print("Converting VAE state dict to original format ...")
        state_dict = convert_vae_to_original(state_dict)

    print(f"Saving converted checkpoint to {args.output_path} ...")
    torch.save(state_dict, args.output_path)
    print("Done!")
