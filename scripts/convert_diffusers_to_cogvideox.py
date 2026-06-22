# Script for converting a HF Diffusers saved CogVideoX pipeline back to the original checkpoint format.
# *Only* converts the Transformer and VAE.
# Does not convert optimizer state or any other thing.
#
# Reference forward conversion: scripts/convert_cogvideox_to_diffusers.py

import argparse
import os
import os.path as osp
from typing import Dict

import torch
from safetensors.torch import load_file


# ==========================#
# Transformer Reverse Ops  #
# ==========================#

# (original-format-key-segment, diffusers-format-key-segment)
# Applied in order — most specific replacements first to avoid substring conflicts.
# Note: "dense_4h_to_h" → "2" is handled separately (post-processing) to avoid
# accidentally replacing "2" in other contexts like "linear_2" or "norm2".
TRANSFORMER_REVERSE_RENAME = [
    # Global mappings with specific patterns (must come before shorter matches)
    ("time_embed.0", "time_embedding.linear_1"),
    ("time_embed.2", "time_embedding.linear_2"),
    ("ofs_embed.0", "ofs_embedding.linear_1"),
    ("ofs_embed.2", "ofs_embedding.linear_2"),
    ("transformer.final_layernorm", "norm_final"),  # before norm_out -> mixins.final_layer.norm_final
    ("mixins.pos_embed.pos_embedding", "patch_embed.pos_embedding"),  # before generic patch_embed
    ("mixins.final_layer.norm_final", "norm_out.norm"),
    ("mixins.final_layer.linear", "proj_out"),
    ("mixins.final_layer.adaLN_modulation.1", "norm_out.linear"),
    # Per-layer mappings (go inside transformer_blocks.N.*)
    ("attention", "attn1"),
    ("mlp", "ff.net"),
    ("input_layernorm", "norm1.norm"),
    ("post_attention_layernorm", "norm2.norm"),
    ("dense_h_to_4h", "0.proj"),
    ("dense", "to_out.0"),
    # Global: less specific (applied last)
    ("transformer.layers", "transformer_blocks"),
    ("mixins.patch_embed", "patch_embed"),
]


def reverse_special_transformer_ops(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Reverse the special operations done during forward conversion:

    1. to_q / to_k / to_v → query_key_value (concatenate along dim 0)
    2. norm_q / norm_k → query_layernorm_list / key_layernorm_list
    3. norm1.linear / norm2.linear → adaln_layer.adaLN_modulations (concatenate)
    """
    new_state_dict = {}
    processed_keys = set()
    keys = list(state_dict.keys())

    # ---- 1. Merge to_q, to_k, to_v back into query_key_value ----
    # Find unique layer prefixes that have to_q
    qkv_groups = {}
    for key in keys:
        # Match: transformer_blocks.N.attn1.to_q.weight (or .bias)
        for suffix in [".weight", ".bias"]:
            if key.endswith(f"to_q{suffix}"):
                prefix = key[: -len(f"to_q{suffix}")]
                qkv_groups.setdefault(prefix, {})
                qkv_groups[prefix]["q" + suffix] = key
            elif key.endswith(f"to_k{suffix}"):
                prefix = key[: -len(f"to_k{suffix}")]
                qkv_groups.setdefault(prefix, {})
                qkv_groups[prefix]["k" + suffix] = key
            elif key.endswith(f"to_v{suffix}"):
                prefix = key[: -len(f"to_v{suffix}")]
                qkv_groups.setdefault(prefix, {})
                qkv_groups[prefix]["v" + suffix] = key

    for prefix, parts in qkv_groups.items():
        has_qkv_weights = "q.weight" in parts and "k.weight" in parts and "v.weight" in parts
        has_qkv_biases = "q.bias" in parts and "k.bias" in parts and "v.bias" in parts
        if has_qkv_weights:
            qkv_weight = torch.cat(
                [state_dict[parts["q.weight"]], state_dict[parts["k.weight"]], state_dict[parts["v.weight"]]], dim=0
            )
            new_key = prefix + "query_key_value.weight"
            new_state_dict[new_key] = qkv_weight
            processed_keys.add(parts["q.weight"])
            processed_keys.add(parts["k.weight"])
            processed_keys.add(parts["v.weight"])
        if has_qkv_biases:
            qkv_bias = torch.cat(
                [state_dict[parts["q.bias"]], state_dict[parts["k.bias"]], state_dict[parts["v.bias"]]], dim=0
            )
            new_key = prefix + "query_key_value.bias"
            new_state_dict[new_key] = qkv_bias
            processed_keys.add(parts["q.bias"])
            processed_keys.add(parts["k.bias"])
            processed_keys.add(parts["v.bias"])

    # ---- 2. Merge norm_q / norm_k back into query/key layernorm lists ----
    norm_groups = {}
    for key in keys:
        # Match: transformer_blocks.N.attn1.norm_q.weight (or .bias)
        # or:    transformer_blocks.N.attn1.norm_k.weight
        for suffix in [".weight", ".bias"]:
            if ".norm_q" + suffix in key:
                prefix = key[: key.index(".norm_q" + suffix)]
                norm_groups.setdefault(prefix, {})
                norm_groups[prefix]["q" + suffix] = key
            elif ".norm_k" + suffix in key:
                prefix = key[: key.index(".norm_k" + suffix)]
                norm_groups.setdefault(prefix, {})
                norm_groups[prefix]["k" + suffix] = key

    for prefix, parts in norm_groups.items():
        if "q.weight" in parts:
            new_state_dict[prefix + ".query_layernorm_list.weight"] = state_dict[parts["q.weight"]]
            processed_keys.add(parts["q.weight"])
        if "q.bias" in parts:
            new_state_dict[prefix + ".query_layernorm_list.bias"] = state_dict[parts["q.bias"]]
            processed_keys.add(parts["q.bias"])
        if "k.weight" in parts:
            new_state_dict[prefix + ".key_layernorm_list.weight"] = state_dict[parts["k.weight"]]
            processed_keys.add(parts["k.weight"])
        if "k.bias" in parts:
            new_state_dict[prefix + ".key_layernorm_list.bias"] = state_dict[parts["k.bias"]]
            processed_keys.add(parts["k.bias"])

    # ---- 3. Merge norm1.linear / norm2.linear back into adaln_layer.adaLN_modulations ----
    adaln_groups = {}
    for key in keys:
        # Match: transformer_blocks.N.norm1.linear.weight (or .bias)
        for suffix in [".weight", ".bias"]:
            if ".norm1.linear" + suffix in key:
                layer_prefix = key[: key.index(".norm1.linear" + suffix)]
                adaln_groups.setdefault(layer_prefix, {})
                adaln_groups[layer_prefix]["norm1" + suffix] = key
            elif ".norm2.linear" + suffix in key:
                layer_prefix = key[: key.index(".norm2.linear" + suffix)]
                adaln_groups.setdefault(layer_prefix, {})
                adaln_groups[layer_prefix]["norm2" + suffix] = key

    for layer_prefix, parts in adaln_groups.items():
        if "norm1.weight" in parts and "norm2.weight" in parts:
            norm1_w = state_dict[parts["norm1.weight"]]
            norm2_w = state_dict[parts["norm2.weight"]]
            # Original: weights_or_biases = state_dict[key].chunk(12, dim=0)
            # norm1 = cat(weights[0:3] + weights[6:9])
            # norm2 = cat(weights[3:6] + weights[9:12])
            # Reverse: interleave norm1 and norm2 back into original 12 chunks
            n1_chunks = norm1_w.chunk(6, dim=0)
            n2_chunks = norm2_w.chunk(6, dim=0)
            combined = torch.cat(
                [
                    n1_chunks[0],
                    n1_chunks[1],
                    n1_chunks[2],
                    n2_chunks[0],
                    n2_chunks[1],
                    n2_chunks[2],
                    n1_chunks[3],
                    n1_chunks[4],
                    n1_chunks[5],
                    n2_chunks[3],
                    n2_chunks[4],
                    n2_chunks[5],
                ],
                dim=0,
            )
            new_state_dict[layer_prefix + ".adaln_layer.adaLN_modulations.weight"] = combined
            processed_keys.add(parts["norm1.weight"])
            processed_keys.add(parts["norm2.weight"])
        if "norm1.bias" in parts and "norm2.bias" in parts:
            norm1_b = state_dict[parts["norm1.bias"]]
            norm2_b = state_dict[parts["norm2.bias"]]
            n1_chunks = norm1_b.chunk(6, dim=0)
            n2_chunks = norm2_b.chunk(6, dim=0)
            combined = torch.cat(
                [
                    n1_chunks[0],
                    n1_chunks[1],
                    n1_chunks[2],
                    n2_chunks[0],
                    n2_chunks[1],
                    n2_chunks[2],
                    n1_chunks[3],
                    n1_chunks[4],
                    n1_chunks[5],
                    n2_chunks[3],
                    n2_chunks[4],
                    n2_chunks[5],
                ],
                dim=0,
            )
            new_state_dict[layer_prefix + ".adaln_layer.adaLN_modulations.bias"] = combined
            processed_keys.add(parts["norm1.bias"])
            processed_keys.add(parts["norm2.bias"])

    # ---- Pass through all unprocessed keys ----
    for key in keys:
        if key not in processed_keys:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def convert_transformer_state_dict(transformer_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert diffusers transformer state dict back to CogVideoX original format."""
    # Step 1: Reverse special operations
    state_dict = reverse_special_transformer_ops(transformer_state_dict)

    # Step 2: Apply key renames (more specific first, to avoid substring conflicts)
    new_state_dict = {}
    for key, tensor in state_dict.items():
        new_key = key
        for orig_seg, diff_seg in TRANSFORMER_REVERSE_RENAME:
            new_key = new_key.replace(diff_seg, orig_seg)
        new_state_dict[new_key] = tensor

    # Step 2b: Handle dense_4h_to_h (replaces "2" in ff.net context only).
    # By now "ff.net" has been replaced by "mlp", so we safely target ".mlp.2."
    fixed_state_dict = {}
    for key, tensor in new_state_dict.items():
        # Replace the ".mlp.2." segment (or ending with ".mlp.2")
        fixed_key = key.replace(".mlp.2.", ".mlp.dense_4h_to_h.")
        # Also handle the case where ".mlp.2" is at the end before .weight/.bias
        if fixed_key == key:
            fixed_key = key.replace(".mlp.2.weight", ".mlp.dense_4h_to_h.weight")
            fixed_key = fixed_key.replace(".mlp.2.bias", ".mlp.dense_4h_to_h.bias")
        fixed_state_dict[fixed_key] = tensor
    new_state_dict = fixed_state_dict

    # Step 3: Add original prefix
    final_state_dict = {"model.diffusion_model." + k: v for k, v in new_state_dict.items()}

    return final_state_dict


# ====================#
# VAE Reverse Ops     #
# ====================#

# (original-format-key-segment, diffusers-format-key-segment)
# Applied in order — most specific replacements first.
VAE_REVERSE_RENAME = [
    # Mid-block specific (must come before generic resnets->block, down_blocks->down)
    ("encoder.mid.block_1", "encoder.mid_block.resnets.0"),
    ("encoder.mid.block_2", "encoder.mid_block.resnets.1"),
    ("decoder.mid.block_1", "decoder.mid_block.resnets.0"),
    ("decoder.mid.block_2", "decoder.mid_block.resnets.1"),
    # Generic
    ("block.", "resnets."),
    ("down.", "down_blocks."),
    ("up.", "up_blocks."),
    ("downsample", "downsamplers.0"),
    ("upsample", "upsamplers.0"),
    ("nin_shortcut", "conv_shortcut"),
]


def reverse_special_vae_ops(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Reverse VAE special operations:
    - 'up.' layer indices were reversed in forward conversion → reverse them back
    - 'loss' key was removed → nothing to do
    """
    new_state_dict = {}
    processed_keys = set()

    for key in list(state_dict.keys()):
        if "up_blocks" not in key:
            continue
        # In forward: original 'up.N.' had N reversed: new_N = 4 - 1 - N
        # So in reverse: original_N = 4 - 1 - new_N (same formula, it's self-inverse!)
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "up_blocks":
                try:
                    layer_idx = int(parts[i + 1])
                    reversed_idx = 4 - 1 - layer_idx  # 3 - layer_idx
                    parts[i + 1] = str(reversed_idx)
                except (ValueError, IndexError):
                    pass
        new_key = ".".join(parts)
        new_state_dict[new_key] = state_dict[key]
        processed_keys.add(key)

    for key in state_dict:
        if key not in processed_keys:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def convert_vae_state_dict(vae_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert diffusers VAE state dict back to CogVideoX original format."""
    # Step 1: Reverse special operations (up_blocks index reversal)
    state_dict = reverse_special_vae_ops(vae_state_dict)

    # Step 2: Apply key renames
    new_state_dict = {}
    for key, tensor in state_dict.items():
        new_key = key
        for orig_seg, diff_seg in VAE_REVERSE_RENAME:
            new_key = new_key.replace(diff_seg, orig_seg)
        new_state_dict[new_key] = tensor

    return new_state_dict


# ==============#
# Main Script   #
# ==============#


def load_state_dict_from_folder(folder_path: str) -> Dict[str, torch.Tensor]:
    """Load state dict from a diffusers model folder (try safetensors first, then pytorch)."""
    safetensors_path = osp.join(folder_path, "diffusion_pytorch_model.safetensors")
    bin_path = osp.join(folder_path, "diffusion_pytorch_model.bin")

    if osp.exists(safetensors_path):
        return load_file(safetensors_path, device="cpu")
    elif osp.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model file found in {folder_path} (tried .safetensors and .bin)")


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert a HF Diffusers CogVideoX pipeline back to the original checkpoint format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the diffusers CogVideoX pipeline (directory with model_index.json).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Whether to save the model weights in fp16.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Whether to save the model weights in bf16.",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        default=False,
        help="Save weights using safetensors format (default uses .bin).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.fp16 and args.bf16:
        raise ValueError("You cannot pass both --fp16 and --bf16 at the same time.")

    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    assert osp.isdir(args.model_path), f"Model path {args.model_path} is not a valid directory."

    transformer_path = osp.join(args.model_path, "transformer")
    vae_path = osp.join(args.model_path, "vae")

    combined_state_dict = {}

    # Convert transformer
    if osp.isdir(transformer_path):
        print("Converting transformer...")
        transformer_sd = load_state_dict_from_folder(transformer_path)
        transformer_sd = convert_transformer_state_dict(transformer_sd)
        combined_state_dict.update(transformer_sd)
        print(f"  Converted {len(transformer_sd)} transformer keys.")
    else:
        print("No transformer/ directory found — skipping transformer conversion.")

    # Convert VAE
    if osp.isdir(vae_path):
        print("Converting VAE...")
        vae_sd = load_state_dict_from_folder(vae_path)
        vae_sd = convert_vae_state_dict(vae_sd)
        combined_state_dict.update(vae_sd)
        print(f"  Converted {len(vae_sd)} VAE keys.")
    else:
        print("No vae/ directory found — skipping VAE conversion.")

    if not combined_state_dict:
        raise RuntimeError("No model components found to convert.")

    # Apply dtype
    if dtype != torch.float32:
        combined_state_dict = {k: v.to(dtype) for k, v in combined_state_dict.items()}

    # Save
    os.makedirs(osp.dirname(args.output_path) or ".", exist_ok=True)

    if args.use_safetensors:
        from safetensors.torch import save_file

        save_file(combined_state_dict, args.output_path)
        print(f"Saved safetensors checkpoint to {args.output_path}")
    else:
        checkpoint = {"state_dict": combined_state_dict}
        torch.save(checkpoint, args.output_path)
        print(f"Saved PyTorch checkpoint to {args.output_path}")

    print("Done!")
