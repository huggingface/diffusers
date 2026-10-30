import argparse
from typing import Any, Dict

import torch
from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX


# ============================================================
# Transformer reverse mappings
# ============================================================

TRANSFORMER_KEYS_REVERSE = {
    "norm_final": "transformer.final_layernorm",
    "transformer_blocks": "transformer.layers",
    "attn1": "attention",
    "ff.net": "mlp",
    "0.proj": "dense_h_to_4h",
    "2": "dense_4h_to_h",
    "to_out.0": "dense",
    "norm1.norm": "input_layernorm",
    "norm2.norm": "post_attention_layernorm",
    "time_embedding.linear_1": "time_embed.0",
    "time_embedding.linear_2": "time_embed.2",
    "ofs_embedding.linear_1": "ofs_embed.0",
    "ofs_embedding.linear_2": "ofs_embed.2",
    "patch_embed": "mixins.patch_embed",
    "norm_out.norm": "mixins.final_layer.norm_final",
    "proj_out": "mixins.final_layer.linear",
    "norm_out.linear": "mixins.final_layer.adaLN_modulation.1",
    "patch_embed.pos_embedding": "mixins.pos_embed.pos_embedding",
}


# ============================================================
# VAE reverse mappings
# ============================================================

VAE_KEYS_REVERSE = {
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


def reverse_qkv(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge to_q, to_k, to_v back into query_key_value"""
    new_state_dict = dict(state_dict)
    processed = set()

    for key in list(state_dict.keys()):
        if ".attn1.to_q.weight" not in key:
            continue

        parts = key.split(".")
        layer_id = parts[1]
        if layer_id in processed:
            continue
        processed.add(layer_id)

        prefix = f"transformer_blocks.{layer_id}.attn1"

        q = state_dict[f"{prefix}.to_q.weight"]
        k = state_dict[f"{prefix}.to_k.weight"]
        v = state_dict[f"{prefix}.to_v.weight"]
        qkv = torch.cat([q, k, v], dim=0)

        new_state_dict[f"transformer.layers.{layer_id}.attention.query_key_value.weight"] = qkv

        if f"{prefix}.to_q.bias" in state_dict:
            qb = state_dict[f"{prefix}.to_q.bias"]
            kb = state_dict[f"{prefix}.to_k.bias"]
            vb = state_dict[f"{prefix}.to_v.bias"]
            new_state_dict[f"transformer.layers.{layer_id}.attention.query_key_value.bias"] = torch.cat(
                [qb, kb, vb], dim=0
            )
            del new_state_dict[f"{prefix}.to_q.bias"]
            del new_state_dict[f"{prefix}.to_k.bias"]
            del new_state_dict[f"{prefix}.to_v.bias"]

        del new_state_dict[f"{prefix}.to_q.weight"]
        del new_state_dict[f"{prefix}.to_k.weight"]
        del new_state_dict[f"{prefix}.to_v.weight"]

    return new_state_dict


def reverse_adaln(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Correct reverse of AdaLN split.
    Forward did:
        original (12 chunks) → 
        norm1 = cat(0:3 + 6:9)
        norm2 = cat(3:6 + 9:12)
    """
    new_state_dict = dict(state_dict)
    processed = set()

    for key in list(state_dict.keys()):
        if ".norm1.linear.weight" not in key:
            continue

        parts = key.split(".")
        layer_id = parts[1]
        if layer_id in processed:
            continue
        processed.add(layer_id)

        norm1_w = state_dict.get(f"transformer_blocks.{layer_id}.norm1.linear.weight")
        norm2_w = state_dict.get(f"transformer_blocks.{layer_id}.norm2.linear.weight")
        norm1_b = state_dict.get(f"transformer_blocks.{layer_id}.norm1.linear.bias")
        norm2_b = state_dict.get(f"transformer_blocks.{layer_id}.norm2.linear.bias")

        if norm1_w is None or norm2_w is None:
            continue

        n1 = norm1_w.chunk(6, dim=0)
        n2 = norm2_w.chunk(6, dim=0)

        reconstructed_w = torch.cat([
            n1[0], n1[1], n1[2],   # 0,1,2
            n2[0], n2[1], n2[2],   # 3,4,5
            n1[3], n1[4], n1[5],   # 6,7,8
            n2[3], n2[4], n2[5],   # 9,10,11
        ], dim=0)

        new_state_dict[f"transformer.layers.{layer_id}.adaln_layer.adaLN_modulations.1.weight"] = reconstructed_w

        if norm1_b is not None and norm2_b is not None:
            n1b = norm1_b.chunk(6, dim=0)
            n2b = norm2_b.chunk(6, dim=0)
            reconstructed_b = torch.cat([
                n1b[0], n1b[1], n1b[2],
                n2b[0], n2b[1], n2b[2],
                n1b[3], n1b[4], n1b[5],
                n2b[3], n2b[4], n2b[5],
            ], dim=0)
            new_state_dict[f"transformer.layers.{layer_id}.adaln_layer.adaLN_modulations.1.bias"] = reconstructed_b

            del new_state_dict[f"transformer_blocks.{layer_id}.norm1.linear.bias"]
            del new_state_dict[f"transformer_blocks.{layer_id}.norm2.linear.bias"]

        del new_state_dict[f"transformer_blocks.{layer_id}.norm1.linear.weight"]
        del new_state_dict[f"transformer_blocks.{layer_id}.norm2.linear.weight"]

    return new_state_dict


def reverse_rename_transformer(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for d_key, o_key in TRANSFORMER_KEYS_REVERSE.items():
            if d_key in new_key:
                new_key = new_key.replace(d_key, o_key)
        new_state_dict[new_key] = value
    return new_state_dict


def reverse_up_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse the up_blocks index flipping done in the forward script"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "up_blocks." in key:
            parts = key.split(".")
            try:
                layer_index = int(parts[2])
                original_index = 3 - layer_index
                parts[1] = "up"
                parts[2] = str(original_index)
                new_key = ".".join(parts)
                new_state_dict[new_key] = value
            except (IndexError, ValueError):
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def reverse_rename_vae(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for d_key, o_key in VAE_KEYS_REVERSE.items():
            if d_key in new_key:
                new_key = new_key.replace(d_key, o_key)
        new_state_dict[new_key] = value
    return new_state_dict


def convert_transformer(model_path: str, output_path: str):
    print("→ Loading Diffusers transformer...")
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.float32
    )
    state_dict = transformer.state_dict()
    print(f"  Loaded {len(state_dict)} keys")

    print("→ Reversing QKV...")
    state_dict = reverse_qkv(state_dict)

    print("→ Reversing AdaLN...")
    state_dict = reverse_adaln(state_dict)

    print("→ Reversing key names...")
    state_dict = reverse_rename_transformer(state_dict)

    final_state_dict = {f"model.diffusion_model.{k}": v for k, v in state_dict.items()}

    print(f"→ Saving transformer to {output_path}")
    torch.save(final_state_dict, output_path)
    print("✓ Transformer conversion completed")


def convert_vae(model_path: str, output_path: str):
    print("→ Loading Diffusers VAE...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    )
    state_dict = vae.state_dict()
    print(f"  Loaded {len(state_dict)} keys")

    print("→ Reversing up_blocks keys...")
    state_dict = reverse_up_keys(state_dict)

    print("→ Reversing VAE key names...")
    state_dict = reverse_rename_vae(state_dict)

    print(f"→ Saving VAE to {output_path}")
    torch.save(state_dict, output_path)
    print("✓ VAE conversion completed")


def get_args():
    parser = argparse.ArgumentParser(description="Convert Diffusers CogVideoX → Original format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Diffusers CogVideoX model folder")
    parser.add_argument("--transformer_output", type=str, default=None, help="Output path for transformer .pt")
    parser.add_argument("--vae_output", type=str, default=None, help="Output path for VAE .pt")
    parser.add_argument("--transformer_only", action="store_true", help="Only convert transformer")
    parser.add_argument("--vae_only", action="store_true", help="Only convert VAE")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("=" * 60)
    print("Diffusers → Original CogVideoX Converter")
    print("=" * 60)

    if args.transformer_only or (not args.transformer_only and not args.vae_only):
        if args.transformer_output is None:
            raise ValueError("Please provide --transformer_output")
        convert_transformer(args.model_path, args.transformer_output)

    if args.vae_only or (not args.transformer_only and not args.vae_only):
        if args.vae_output is None:
            raise ValueError("Please provide --vae_output")
        convert_vae(args.model_path, args.vae_output)

    print("\nAll done!")