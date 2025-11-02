import argparse
import os
import shutil
import tempfile
from typing import Any, Dict

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLMagi1,
    FlowMatchEulerDiscreteScheduler,
    Magi1ImageToVideoPipeline,
    Magi1Pipeline,
    Magi1Transformer3DModel,
    Magi1VideoToVideoPipeline,
)


# Simple top-level mappings for MAGI-1 transformer conversion
SIMPLE_TRANSFORMER_MAPPINGS = [
    ("x_embedder.weight", "patch_embedding.weight"),
    ("t_embedder.mlp.0.weight", "condition_embedder.time_embedder.linear_1.weight"),
    ("t_embedder.mlp.0.bias", "condition_embedder.time_embedder.linear_1.bias"),
    ("t_embedder.mlp.2.weight", "condition_embedder.time_embedder.linear_2.weight"),
    ("t_embedder.mlp.2.bias", "condition_embedder.time_embedder.linear_2.bias"),
    ("y_embedder.y_proj_xattn.0.weight", "condition_embedder.text_embedder.y_proj_xattn.0.weight"),
    ("y_embedder.y_proj_xattn.0.bias", "condition_embedder.text_embedder.y_proj_xattn.0.bias"),
    ("y_embedder.y_proj_adaln.0.weight", "condition_embedder.text_embedder.y_proj_adaln.weight"),
    ("y_embedder.y_proj_adaln.0.bias", "condition_embedder.text_embedder.y_proj_adaln.bias"),
    ("videodit_blocks.final_layernorm.weight", "norm_out.weight"),
    ("videodit_blocks.final_layernorm.bias", "norm_out.bias"),
    ("final_linear.linear.weight", "proj_out.weight"),
    ("rope.bands", "rope.bands"),
]


def get_transformer_config(model_type: str) -> Dict[str, Any]:
    """
    Get transformer configuration for different MAGI-1 model variants.

    Args:
        model_type: Model type identifier (e.g., "MAGI-1-T2V-4.5B-distill")

    Returns:
        Dictionary containing model_id, repo_path, and diffusers_config
    """
    if model_type == "MAGI-1-T2V-4.5B-distill" or model_type == "4.5B_distill":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "4.5B_distill",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 34,
                "num_attention_heads": 24,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 12288,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-T2V-24B-distill" or model_type == "24B_distill":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "24B_distill",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 48,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 16384,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-T2V-4.5B" or model_type == "4.5B":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "4.5B_base",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 34,
                "num_attention_heads": 24,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 12288,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-T2V-24B" or model_type == "24B":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "24B_base",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 48,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 16384,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-I2V-4.5B-distill":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "4.5B_distill",  # Placeholder - update when I2V weights are released
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 34,
                "num_attention_heads": 24,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 12288,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-I2V-4.5B":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "4.5B_base",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 34,
                "num_attention_heads": 24,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 12288,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-I2V-24B-distill":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "24B_distill",  # Placeholder - update when I2V weights are released
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 48,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 16384,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-I2V-24B":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "24B_base",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 48,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 16384,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-V2V-4.5B-distill":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "4.5B_distill",  # Placeholder - update when V2V weights are released
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 34,
                "num_attention_heads": 24,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 12288,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-V2V-4.5B":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "4.5B_base",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 34,
                "num_attention_heads": 24,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 12288,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-V2V-24B-distill":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "24B_distill",  # Placeholder - update when V2V weights are released
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 48,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 16384,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    elif model_type == "MAGI-1-V2V-24B":
        return {
            "model_id": "sand-ai/MAGI-1",
            "repo_path": "24B_base",
            "diffusers_config": {
                "in_channels": 16,
                "out_channels": 16,
                "num_layers": 48,
                "num_attention_heads": 32,
                "num_kv_heads": 8,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 16384,
                "patch_size": (1, 2, 2),
                "eps": 1e-6,
            },
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def convert_magi1_transformer(model_type):
    """
    Convert MAGI-1 transformer for a specific model type.

    Args:
        model_type: The model type (e.g., "MAGI-1-T2V-4.5B-distill", "MAGI-1-T2V-24B-distill", etc.)

    Returns:
        The converted transformer model.
    """
    config = get_transformer_config(model_type)
    model_id = config["model_id"]
    repo_path = config["repo_path"]
    diffusers_config = config["diffusers_config"]

    temp_dir = tempfile.mkdtemp()
    transformer_ckpt_dir = os.path.join(temp_dir, "transformer_checkpoint")
    os.makedirs(transformer_ckpt_dir, exist_ok=True)

    # Determine checkpoint path based on model type (distill vs base)
    if "distill" in model_type.lower():
        weight_subpath = "inference_weight.distill"
    else:
        weight_subpath = "inference_weight"

    checkpoint_files = []
    last_exception = None

    # Try to download both shards
    for shard_index in [1, 2]:
        try:
            shard_filename = f"model-{shard_index:05d}-of-00002.safetensors"
            checkpoint_path = f"ckpt/magi/{repo_path}/{weight_subpath}/{shard_filename}"
            print(f"Attempting to download: {model_id}/{checkpoint_path}")
            shard_path = hf_hub_download(model_id, checkpoint_path)
            checkpoint_files.append(shard_path)
            print(f"Successfully downloaded shard {shard_index}")
        except Exception as e:
            last_exception = e
            print(f"Failed to download shard {shard_index}: {e}")
            break

    if not checkpoint_files:
        error_msg = f"No checkpoint files found for model type: {model_type}\n"
        error_msg += f"Tried path: {model_id}/ckpt/magi/{repo_path}/{weight_subpath}/\n"
        if last_exception:
            error_msg += f"Last error: {last_exception}"
        raise ValueError(error_msg)

    for i, shard_path in enumerate(checkpoint_files):
        dest_path = os.path.join(transformer_ckpt_dir, f"model-{i + 1:05d}-of-{len(checkpoint_files):05d}.safetensors")
        shutil.copy2(shard_path, dest_path)

    transformer = convert_magi1_transformer_checkpoint(transformer_ckpt_dir, diffusers_config=diffusers_config)

    return transformer


def convert_magi1_vae():
    """
    Convert MAGI-1 VAE checkpoint to diffusers format.

    Uses init_empty_weights() for memory-efficient loading of large models,
    avoiding OOM errors during conversion.

    Returns:
        AutoencoderKLMagi1: The converted VAE model.
    """
    vae_ckpt_path = hf_hub_download("sand-ai/MAGI-1", "ckpt/vae/diffusion_pytorch_model.safetensors")
    checkpoint = load_file(vae_ckpt_path)

    config = {
        "patch_size": (4, 8, 8),
        "num_attention_heads": 16,
        "attention_head_dim": 64,
        "z_dim": 16,
        "height": 256,
        "width": 256,
        "num_frames": 16,
        "ffn_dim": 4 * 1024,
        "num_layers": 24,
        "eps": 1e-6,
        # Add compression ratios explicitly for pipeline compatibility
        "temporal_compression_ratio": 4,  # patch_size[0]
        "spatial_compression_ratio": 8,  # patch_size[1] or patch_size[2]
    }

    with init_empty_weights():
        vae = AutoencoderKLMagi1.from_config(config)

    converted_state_dict = convert_vae_state_dict(checkpoint)

    vae.load_state_dict(converted_state_dict, strict=True, assign=True)

    return vae


def convert_vae_state_dict(checkpoint):
    """
    Convert MAGI-1 VAE state dict to diffusers format.

    Maps the keys from the MAGI-1 VAE state dict to the diffusers VAE state dict.
    """
    state_dict = {}

    state_dict["encoder.patch_embedding.weight"] = checkpoint["encoder.patch_embed.proj.weight"]
    state_dict["encoder.patch_embedding.bias"] = checkpoint["encoder.patch_embed.proj.bias"]

    state_dict["encoder.pos_embed"] = checkpoint["encoder.pos_embed"]

    state_dict["encoder.cls_token"] = checkpoint["encoder.cls_token"]

    for i in range(24):
        qkv_weight = checkpoint[f"encoder.blocks.{i}.attn.qkv.weight"]
        qkv_bias = checkpoint[f"encoder.blocks.{i}.attn.qkv.bias"]

        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

        state_dict[f"encoder.blocks.{i}.attn.to_q.weight"] = q_weight
        state_dict[f"encoder.blocks.{i}.attn.to_q.bias"] = q_bias
        state_dict[f"encoder.blocks.{i}.attn.to_k.weight"] = k_weight
        state_dict[f"encoder.blocks.{i}.attn.to_k.bias"] = k_bias
        state_dict[f"encoder.blocks.{i}.attn.to_v.weight"] = v_weight
        state_dict[f"encoder.blocks.{i}.attn.to_v.bias"] = v_bias

        state_dict[f"encoder.blocks.{i}.attn.to_out.0.weight"] = checkpoint[f"encoder.blocks.{i}.attn.proj.weight"]
        state_dict[f"encoder.blocks.{i}.attn.to_out.0.bias"] = checkpoint[f"encoder.blocks.{i}.attn.proj.bias"]

        state_dict[f"encoder.blocks.{i}.norm.weight"] = checkpoint[f"encoder.blocks.{i}.norm2.weight"]
        state_dict[f"encoder.blocks.{i}.norm.bias"] = checkpoint[f"encoder.blocks.{i}.norm2.bias"]

        state_dict[f"encoder.blocks.{i}.proj_out.net.0.proj.weight"] = checkpoint[f"encoder.blocks.{i}.mlp.fc1.weight"]
        state_dict[f"encoder.blocks.{i}.proj_out.net.0.proj.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc1.bias"]
        state_dict[f"encoder.blocks.{i}.proj_out.net.2.weight"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.weight"]

        state_dict[f"encoder.blocks.{i}.proj_out.net.2.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.bias"]

    state_dict["encoder.norm_out.weight"] = checkpoint["encoder.norm.weight"]
    state_dict["encoder.norm_out.bias"] = checkpoint["encoder.norm.bias"]

    state_dict["encoder.linear_out.weight"] = checkpoint["encoder.last_layer.weight"]
    state_dict["encoder.linear_out.bias"] = checkpoint["encoder.last_layer.bias"]

    state_dict["decoder.proj_in.weight"] = checkpoint["decoder.proj_in.weight"]
    state_dict["decoder.proj_in.bias"] = checkpoint["decoder.proj_in.bias"]

    state_dict["decoder.pos_embed"] = checkpoint["decoder.pos_embed"]

    state_dict["decoder.cls_token"] = checkpoint["decoder.cls_token"]

    for i in range(24):
        qkv_weight = checkpoint[f"decoder.blocks.{i}.attn.qkv.weight"]
        qkv_bias = checkpoint[f"decoder.blocks.{i}.attn.qkv.bias"]

        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

        state_dict[f"decoder.blocks.{i}.attn.to_q.weight"] = q_weight
        state_dict[f"decoder.blocks.{i}.attn.to_q.bias"] = q_bias
        state_dict[f"decoder.blocks.{i}.attn.to_k.weight"] = k_weight
        state_dict[f"decoder.blocks.{i}.attn.to_k.bias"] = k_bias
        state_dict[f"decoder.blocks.{i}.attn.to_v.weight"] = v_weight
        state_dict[f"decoder.blocks.{i}.attn.to_v.bias"] = v_bias

        state_dict[f"decoder.blocks.{i}.attn.to_out.0.weight"] = checkpoint[f"decoder.blocks.{i}.attn.proj.weight"]
        state_dict[f"decoder.blocks.{i}.attn.to_out.0.bias"] = checkpoint[f"decoder.blocks.{i}.attn.proj.bias"]

        state_dict[f"decoder.blocks.{i}.norm.weight"] = checkpoint[f"decoder.blocks.{i}.norm2.weight"]
        state_dict[f"decoder.blocks.{i}.norm.bias"] = checkpoint[f"decoder.blocks.{i}.norm2.bias"]

        state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.weight"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.bias"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.2.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.weight"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.2.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.bias"]

    state_dict["decoder.norm_out.weight"] = checkpoint["decoder.norm.weight"]
    state_dict["decoder.norm_out.bias"] = checkpoint["decoder.norm.bias"]

    state_dict["decoder.conv_out.weight"] = checkpoint["decoder.last_layer.weight"]
    state_dict["decoder.conv_out.bias"] = checkpoint["decoder.last_layer.bias"]

    return state_dict


def load_magi1_transformer_checkpoint(checkpoint_path):
    """
    Load a MAGI-1 transformer checkpoint.

    Args:
        checkpoint_path: Path to the MAGI-1 transformer checkpoint.

    Returns:
        The loaded checkpoint state dict.
    """
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    elif os.path.isdir(checkpoint_path):
        safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
        if safetensors_files:
            state_dict = {}
            for safetensors_file in sorted(safetensors_files):
                file_path = os.path.join(checkpoint_path, safetensors_file)
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        else:
            checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt") or f.endswith(".pth")]
            if not checkpoint_files:
                raise ValueError(f"No checkpoint files found in {checkpoint_path}")

            checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[0])
            checkpoint_data = torch.load(checkpoint_file, map_location="cpu")

            if isinstance(checkpoint_data, dict):
                if "model" in checkpoint_data:
                    state_dict = checkpoint_data["model"]
                elif "state_dict" in checkpoint_data:
                    state_dict = checkpoint_data["state_dict"]
                else:
                    state_dict = checkpoint_data
            else:
                state_dict = checkpoint_data
    else:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint_data, dict):
            if "model" in checkpoint_data:
                state_dict = checkpoint_data["model"]
            elif "state_dict" in checkpoint_data:
                state_dict = checkpoint_data["state_dict"]
            else:
                state_dict = checkpoint_data
        else:
            state_dict = checkpoint_data

    return state_dict


def convert_magi1_transformer_checkpoint(checkpoint_path, diffusers_config):
    """
    Convert a MAGI-1 transformer checkpoint to a diffusers Magi1Transformer3DModel.

    Uses init_empty_weights() for memory-efficient loading to avoid OOM errors
    with large models (4.5B-24B parameters). Follows diffusers best practices
    with strict=True and assign=True for direct tensor assignment.

    Preserves original mixed precision from checkpoint:
    - F32 for embeddings and output layers (numerical stability)
    - BF16 for attention and MLP layers (memory efficiency)

    Args:
        checkpoint_path: Path to the MAGI-1 transformer checkpoint.
        diffusers_config: Diffusers config dict (from get_transformer_config).

    Returns:
        A diffusers Magi1Transformer3DModel model with original mixed precision.
    """
    checkpoint = load_magi1_transformer_checkpoint(checkpoint_path)

    with init_empty_weights():
        transformer = Magi1Transformer3DModel.from_config(diffusers_config)

    converted_state_dict = convert_transformer_state_dict(checkpoint, transformer)

    # Use assign=True to preserve original mixed precision dtypes from checkpoint
    # Original MAGI-1 uses F32 for embeddings/output layers and BF16 for attention/MLP
    transformer.load_state_dict(converted_state_dict, strict=True, assign=True)

    # Note: dtype parameter is intentionally NOT applied to preserve mixed precision
    # If you need uniform dtype, convert after loading the pipeline

    return transformer


def convert_transformer_state_dict(checkpoint, transformer):
    """
    Convert MAGI-1 transformer state dict to diffusers format.

    Uses explicit key mappings for clarity and correctness.
    """
    converted_state_dict = {}

    # Simple top-level mappings
    for src, dst in SIMPLE_TRANSFORMER_MAPPINGS:
        if src in checkpoint:
            converted_state_dict[dst] = checkpoint[src]

    # Determine number of layers
    num_layers = transformer.config.num_layers

    # Per-layer mappings
    for i in range(num_layers):
        layer_prefix = f"videodit_blocks.layers.{i}"
        block_prefix = f"blocks.{i}"

        # Self-attention (attn1)
        converted_state_dict[f"{block_prefix}.norm1.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.layer_norm.weight"
        ]
        converted_state_dict[f"{block_prefix}.norm1.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.layer_norm.bias"
        ]
        converted_state_dict[f"{block_prefix}.attn1.to_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.q.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.to_k.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.k.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.to_v.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.v.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.norm_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.norm_q.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm.bias"
        ]
        converted_state_dict[f"{block_prefix}.attn1.norm_k.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.norm_k.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm.bias"
        ]

        # Cross-attention (attn2)
        converted_state_dict[f"{block_prefix}.attn2.to_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.qx.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn2.norm_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm_xattn.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn2.norm_q.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm_xattn.bias"
        ]
        converted_state_dict[f"{block_prefix}.attn2.norm_k.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm_xattn.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn2.norm_k.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm_xattn.bias"
        ]

        # Split KV for cross-attention
        kv = checkpoint[f"{layer_prefix}.self_attention.linear_kv_xattn.weight"]
        k, v = kv.chunk(2, dim=0)
        converted_state_dict[f"{block_prefix}.attn2.to_k.weight"] = k
        converted_state_dict[f"{block_prefix}.attn2.to_v.weight"] = v

        # Combined projection for both attentions
        converted_state_dict[f"{block_prefix}.attn_proj.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_proj.weight"
        ]
        converted_state_dict[f"{block_prefix}.norm2.weight"] = checkpoint[f"{layer_prefix}.self_attn_post_norm.weight"]
        converted_state_dict[f"{block_prefix}.norm2.bias"] = checkpoint[f"{layer_prefix}.self_attn_post_norm.bias"]

        # MLP
        converted_state_dict[f"{block_prefix}.norm3.weight"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.weight"]
        converted_state_dict[f"{block_prefix}.norm3.bias"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.bias"]
        converted_state_dict[f"{block_prefix}.ffn.net.0.proj.weight"] = checkpoint[
            f"{layer_prefix}.mlp.linear_fc1.weight"
        ]
        converted_state_dict[f"{block_prefix}.ffn.net.2.weight"] = checkpoint[f"{layer_prefix}.mlp.linear_fc2.weight"]
        converted_state_dict[f"{block_prefix}.norm4.weight"] = checkpoint[f"{layer_prefix}.mlp_post_norm.weight"]
        converted_state_dict[f"{block_prefix}.norm4.bias"] = checkpoint[f"{layer_prefix}.mlp_post_norm.bias"]

        # Ada LayerNorm modulation
        converted_state_dict[f"{block_prefix}.ada_modulate_layer.1.weight"] = checkpoint[
            f"{layer_prefix}.ada_modulate_layer.proj.0.weight"
        ]
        converted_state_dict[f"{block_prefix}.ada_modulate_layer.1.bias"] = checkpoint[
            f"{layer_prefix}.ada_modulate_layer.proj.0.bias"
        ]

    return converted_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            "MAGI-1-T2V-4.5B-distill",
            "MAGI-1-T2V-24B-distill",
            "MAGI-1-T2V-4.5B",
            "MAGI-1-T2V-24B",
            "MAGI-1-I2V-4.5B-distill",
            "MAGI-1-I2V-24B-distill",
            "MAGI-1-I2V-4.5B",
            "MAGI-1-I2V-24B",
            "MAGI-1-V2V-4.5B-distill",
            "MAGI-1-V2V-24B-distill",
            "MAGI-1-V2V-4.5B",
            "MAGI-1-V2V-24B",
        ],
        help="Model type to convert",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for converted pipeline")
    parser.add_argument(
        "--dtype", default="bf16", choices=["fp32", "fp16", "bf16", "none"], help="Data type for conversion"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None, help="Hugging Face Hub repo ID to push the converted model to"
    )
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

if __name__ == "__main__":
    args = get_args()

    # Convert transformer
    transformer = convert_magi1_transformer(args.model_type)

    # Convert VAE
    vae = convert_magi1_vae()

    # Load text encoder and tokenizer
    # Apply dtype to text encoder if specified
    if args.dtype != "none":
        text_encoder_dtype = DTYPE_MAPPING[args.dtype]
    else:
        text_encoder_dtype = torch.bfloat16

    text_encoder = T5EncoderModel.from_pretrained(
        "sand-ai/MAGI-1", subfolder="ckpt/t5/t5-v1_1-xxl", torch_dtype=text_encoder_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained("sand-ai/MAGI-1", subfolder="ckpt/t5/t5-v1_1-xxl")

    # Create scheduler with SD3-style shift
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    # Note: Transformer preserves original mixed precision (F32 embeddings + BF16 layers)
    # VAE and text encoder use the specified dtype

    # Determine pipeline class based on model type
    if "I2V" in args.model_type:
        pipe = Magi1ImageToVideoPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
    elif "V2V" in args.model_type:
        pipe = Magi1VideoToVideoPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
    else:  # T2V
        pipe = Magi1Pipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

    # Save complete pipeline
    pipe.save_pretrained(
        args.output_path, repo_id=args.repo_id, push_to_hub=True, safe_serialization=True, max_shard_size="5GB"
    )
