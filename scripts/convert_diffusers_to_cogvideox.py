import argparse
from typing import Any, Dict

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from scripts.convert_cogvideox_to_diffusers import TRANSFORMER_KEYS_RENAME_DICT, VAE_KEYS_RENAME_DICT, get_transformer_init_kwargs


def reassemble_query_key_value_inplace(key: str, state_dict: Dict[str, Any]):
    """Combines separate q/k/v tensors back into single tensor."""
    base_key = key.replace("to_q", "query_key_value")
    q_tensor = state_dict.pop(key)
    k_tensor = state_dict.pop(key.replace("to_q", "to_k"))
    v_tensor = state_dict.pop(key.replace("to_q", "to_v"))

    combined = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)
    state_dict[base_key] = combined


def reassemble_query_key_layernorm_inplace(key: str, state_dict: Dict[str, Any]):
    """Converts diffusers layernorm format back to CogVideoX format."""
    layer_id = key.split(".")[1]
    weight_or_bias = key.split(".")[-1]

    if "norm_q" in key:
        new_key = f"query_layernorm_list.{layer_id}.{weight_or_bias}"
    elif "norm_k" in key:
        new_key = f"key_layernorm_list.{layer_id}.{weight_or_bias}"

    state_dict[new_key] = state_dict.pop(key)


def reassemble_adaln_norm_inplace(key: str, state_dict: Dict[str, Any]):
    """Reconstructs AdaLN modulations from separated norms."""
    layer_id = key.split(".")[1]
    weight_or_bias = key.split(".")[-1]

    norm1_weights = state_dict.pop(
        f"transformer_blocks.{layer_id}.norm1.linear.{weight_or_bias}")
    norm2_weights = state_dict.pop(
        f"transformer_blocks.{layer_id}.norm2.linear.{weight_or_bias}")

    norm1_chunks = torch.chunk(norm1_weights, chunks=6, dim=0)
    norm2_chunks = torch.chunk(norm2_weights, chunks=6, dim=0)

    combined = torch.cat(
        norm1_chunks[:3] + norm2_chunks[:3] + norm1_chunks[3:] + norm2_chunks[3:])
    state_dict[f"adaln_layer.adaLN_modulations.{layer_id}.{weight_or_bias}"] = combined


def restore_position_embeddings_inplace(key: str, state_dict: Dict[str, Any]):
    """Restores position embeddings and related tensors."""
    if "pos_embedding" in key:
        new_key = "position_embedding"
        state_dict[new_key] = state_dict.pop(key)


def replace_down_keys_inplace(key: str, state_dict: Dict[str, Any]):
    """Reverses the up/down block key replacements."""
    key_split = key.split(".")
    layer_index = int(key_split[2])
    replace_layer_index = 4 - 1 - layer_index

    key_split[1] = "down"
    key_split[2] = str(replace_layer_index)
    new_key = ".".join(key_split)

    state_dict[new_key] = state_dict.pop(key)


REVERSE_TRANSFORMER_KEYS_RENAME_DICT = {
    v: k for k, v in TRANSFORMER_KEYS_RENAME_DICT.items()}

REVERSE_TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "to_q": reassemble_query_key_value_inplace,
    "norm_q": reassemble_query_key_layernorm_inplace,
    "norm_k": reassemble_query_key_layernorm_inplace,
    "norm1.linear": reassemble_adaln_norm_inplace,
    "pos_embedding": restore_position_embeddings_inplace,
}

REVERSE_VAE_KEYS_RENAME_DICT = {v: k for k, v in VAE_KEYS_RENAME_DICT.items()}

REVERSE_VAE_SPECIAL_KEYS_REMAP = {
    "down_blocks": replace_down_keys_inplace,
}


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract state dict from nested dictionary structure."""
    state_dict = saved_dict
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "diffusion_model" in state_dict:
            state_dict = state_dict["diffusion_model"]
    return state_dict


def convert_transformer_to_cogvideox(transformer, num_layers, num_attention_heads, use_rotary_positional_embeddings, i2v, dtype, init_kwargs):
    state_dict = transformer.state_dict()
    new_state_dict = {}
    model_prefix = "model.diffusion_model."

    qkv_keys = [k for k in state_dict.keys() if "to_q" in k]
    for key in qkv_keys:
        layer_id = key.split(".")[1]
        base_path = f"transformer.layers.{layer_id}"

        q = state_dict[key]
        k = state_dict[key.replace("to_q", "to_k")]
        v = state_dict[key.replace("to_q", "to_v")]
        combined = torch.cat([q, k, v], dim=0)
        new_state_dict[f"{model_prefix}{base_path}.attention.query_key_value"] = combined.to(
            dtype)

        out_key_base = key.replace("to_q", "to_out.0")
        weight_key = f"{model_prefix}{base_path}.attention.to_out.0.weight"
        bias_key = f"{model_prefix}{base_path}.attention.to_out.0.bias"
        new_state_dict[weight_key] = state_dict[out_key_base].to(dtype)
        new_state_dict[bias_key] = state_dict[out_key_base.replace(
            "weight", "bias")].to(dtype)

    for key, value in state_dict.items():
        if any(x in key for x in ["to_q", "to_k", "to_v", "to_out"]):
            continue

        if "transformer_blocks" in key:
            layer_id = key.split(".")[1]
            new_key = None

            if "norm1.norm" in key:
                new_key = key.replace(f"transformer_blocks.{layer_id}.norm1.norm",
                                      f"transformer.layers.{layer_id}.input_layernorm")
            elif "norm2.norm" in key:
                new_key = key.replace(f"transformer_blocks.{layer_id}.norm2.norm",
                                      f"transformer.layers.{layer_id}.post_attn1_layernorm")
            elif "ff.net.0.proj" in key:
                new_key = key.replace(f"transformer_blocks.{layer_id}.ff.net.0.proj",
                                      f"transformer.layers.{layer_id}.mlp.dense_h_to_4h")
            elif "ff.net.2" in key:
                new_key = key.replace(f"transformer_blocks.{layer_id}.ff.net.2",
                                      f"transformer.layers.{layer_id}.mlp.dense_4h_to_h")

            if new_key:
                new_state_dict[model_prefix + new_key] = value.to(dtype)

    return new_state_dict


def convert_vae_to_cogvideox(
    vae: AutoencoderKLCogVideoX,
    scaling_factor: float,
    version: str,
):
    """Converts diffusers VAE to CogVideoX format."""
    state_dict = vae.state_dict()

    new_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key
        for diffusers_key, cogvideo_key in VAE_KEYS_RENAME_DICT.items():
            cleaned_key = cleaned_key.replace(diffusers_key, cogvideo_key)
        new_state_dict[cleaned_key] = value

    for key in list(new_state_dict.keys()):
        if "up_blocks" in key:
            parts = key.split(".")
            layer_idx = int(parts[2])
            new_idx = 4 - 1 - layer_idx
            parts[1] = "up"
            parts[2] = str(new_idx)
            new_key = ".".join(parts)
            new_state_dict[new_key] = new_state_dict.pop(key)

    return new_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusers_path", type=str, required=True,
        help="Path to diffusers model directory"
    )
    parser.add_argument(
        "--output_transformer_path", type=str, required=True,
        help="Path to save converted transformer checkpoint"
    )
    parser.add_argument(
        "--output_vae_path", type=str, required=True,
        help="Path to save converted VAE checkpoint"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False,
        help="Load and save in fp16 precision"
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False,
        help="Load and save in bf16 precision"
    )
    parser.add_argument(
        "--num_layers", type=int, default=30,
        help="Number of transformer layers (30 for 2B, 42 for 5B)"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=30,
        help="Number of attention heads (30 for 2B, 48 for 5B)"
    )
    parser.add_argument(
        "--use_rotary_positional_embeddings", action="store_true", default=False,
        help="Use rotary positional embeddings (False for 2B, True for 5B)"
    )
    parser.add_argument(
        "--scaling_factor", type=float, default=1.15258426,
        help="VAE scaling factor (1.15258426 for 2B, 0.7 for 5B)"
    )
    parser.add_argument(
        "--i2v", action="store_true", default=False,
        help="Convert Image-to-Video version"
    )
    parser.add_argument(
        "--version", choices=["1.0", "1.5"], default="1.0",
        help="CogVideoX version"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both fp16 and bf16")

    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32

    pipeline_cls = CogVideoXImageToVideoPipeline if args.i2v else CogVideoXPipeline
    pipe = pipeline_cls.from_pretrained(args.diffusers_path)

    init_kwargs = get_transformer_init_kwargs(args.version)
    transformer_state_dict = convert_transformer_to_cogvideox(
        pipe.transformer,
        args.num_layers,
        args.num_attention_heads,
        args.use_rotary_positional_embeddings,
        args.i2v,
        dtype,
        init_kwargs,
    )

    vae_state_dict = convert_vae_to_cogvideox(
        pipe.vae,
        args.scaling_factor,
        args.version,
    )

    torch.save(
        transformer_state_dict,
        args.output_transformer_path,
        _use_new_zipfile_serialization=False
    )

    torch.save(
        vae_state_dict,
        args.output_vae_path,
        _use_new_zipfile_serialization=False
    )
