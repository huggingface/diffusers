#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert MAGI-1 checkpoints to diffusers format.

This script converts MAGI-1 transformer checkpoints from the original sharded format
to the diffusers format. It follows diffusers' design philosophy with modular components,
config-driven model creation, and proper error handling.
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

from diffusers import Magi1Pipeline, Magi1Transformer3DModel
from diffusers.models.autoencoders import AutoencoderKLMagi1
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging as diffusers_logging


# Set up logging
logger = logging.getLogger(__name__)
diffusers_logging.set_verbosity_info()


# Mapping from original MAGI-1 transformer keys to diffusers format
# Following the same pattern as WAN to ensure consistency
TRANSFORMER_KEYS_RENAME_DICT = {
    # Time embedding
    "t_embedder.mlp.0": "condition_embedder.time_embedder.linear_1",
    "t_embedder.mlp.2": "condition_embedder.time_embedder.linear_2",

    # Text embedding - AdaLN projection
    "y_embedder.y_proj_adaln.0": "condition_embedder.text_embedder.linear_1",
    "y_embedder.y_proj_adaln.2": "condition_embedder.text_embedder.linear_2",

    # Time projection
    "condition_embedder.time_embedder.linear_2": "condition_embedder.time_proj", # Duplicate mapping handled later

    # Final output components
    "videodit_blocks.final_layernorm": "norm_out",
    "final_linear.linear": "proj_out",

    # AdaLN modulation
    "ada_modulate_layer.proj.0": "scale_shift_table",

    # Self-attention mappings
    "self_attention.linear_qkv.q": "attn1.to_q",
    "self_attention.linear_qkv.k": "attn1.to_k",
    "self_attention.linear_qkv.v": "attn1.to_v",
    "self_attention.linear_proj": "attn1.to_out.0",
    "self_attention.q_layernorm": "attn1.norm_q",
    "self_attention.k_layernorm": "attn1.norm_k",

    # Cross-attention mappings
    "self_attention.linear_qkv.qx": "attn2.to_q",  # Cross-attention query
    "self_attention.linear_kv_xattn": "attn2.to_kv",  # Will be split into to_k and to_v
    "self_attention.q_layernorm_xattn": "attn2.norm_q",
    "self_attention.k_layernorm_xattn": "attn2.norm_k",

    # Feed-forward network
    "mlp.linear_fc1": "ffn.net.0.proj",
    "mlp.linear_fc2": "ffn.net.2",
    "mlp.layer_norm": "norm3",

    # Normalization layers
    "self_attn_post_norm": "norm2",

    # Image conditioning (for I2V models)
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
}

# Special handling for keys that need custom processing
TRANSFORMER_SPECIAL_KEYS_REMAP = {}




def convert_magi_transformer(model_type):
    """
    Convert MAGI-1 transformer for a specific model type.

    Args:
        model_type: The model type (e.g., "MAGI-1-T2V-4.5B-distill", "MAGI-1-T2V-24B-distill", etc.)

    Returns:
        The converted transformer model.
    """
    # Map model_type to the actual path in the repo
    # MAGI-1-T2V-4.5B-distill -> 4.5B_distill
    # MAGI-1-T2V-24B-distill -> 24B_distill
    model_type_mapping = {
        "MAGI-1-T2V-4.5B-distill": "4.5B_distill",
        "MAGI-1-T2V-24B-distill": "24B_distill",
        "MAGI-1-T2V-4.5B": "4.5B",
        "MAGI-1-T2V-24B": "24B",
        "4.5B_distill": "4.5B_distill",
        "24B_distill": "24B_distill",
        "4.5B": "4.5B",
        "24B": "24B",
    }

    repo_path = model_type_mapping.get(model_type, model_type)

    # Download the transformer checkpoint directory from HuggingFace Hub
    # The checkpoint is sharded into multiple files in inference_weight.distill directory

    # Create a temporary directory to store the sharded checkpoint files
    temp_dir = tempfile.mkdtemp()
    transformer_ckpt_dir = os.path.join(temp_dir, "transformer_checkpoint")
    os.makedirs(transformer_ckpt_dir, exist_ok=True)

    # Download all sharded checkpoint files
    # For 4.5B_distill, there are 2 shards: model-00001-of-00002.safetensors and model-00002-of-00002.safetensors
    checkpoint_files = []
    shard_index = 1
    while True:
        try:
            if shard_index == 1:
                # Try to download the first shard to determine total number of shards
                shard_filename = f"model-{shard_index:05d}-of-00002.safetensors"
                shard_path = hf_hub_download(
                    "sand-ai/MAGI-1",
                    f"ckpt/magi/{repo_path}/inference_weight.distill/{shard_filename}"
                )
                checkpoint_files.append(shard_path)
                print(f"Downloaded {shard_filename}")
                shard_index += 1
            elif shard_index == 2:
                # Download the second shard
                shard_filename = f"model-{shard_index:05d}-of-00002.safetensors"
                shard_path = hf_hub_download(
                    "sand-ai/MAGI-1",
                    f"ckpt/magi/{repo_path}/inference_weight.distill/{shard_filename}"
                )
                checkpoint_files.append(shard_path)
                print(f"Downloaded {shard_filename}")
                break
            else:
                break
        except Exception as e:
            print(f"No more shards found or error downloading shard {shard_index}: {e}")
            break

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found for model type: {model_type}")

    # Copy files to the temporary directory with consistent naming
    for i, shard_path in enumerate(checkpoint_files):
        dest_path = os.path.join(transformer_ckpt_dir, f"model-{i+1:05d}-of-{len(checkpoint_files):05d}.safetensors")
        shutil.copy2(shard_path, dest_path)

    # Convert the transformer checkpoint
    transformer = convert_magi_transformer_checkpoint(transformer_ckpt_dir)

    return transformer


def convert_magi_vae():
    vae_ckpt_path = hf_hub_download("sand-ai/MAGI-1", "ckpt/vae/diffusion_pytorch_model.safetensors")
    checkpoint = load_file(vae_ckpt_path)

    config = {
        "patch_size": (4, 8, 8),  # Based on encoder.patch_embed.proj.weight shape [1024, 3, 4, 8, 8]
        "num_attention_heads": 16,  # Typical for 1024 hidden size
        "attention_head_dim": 64,  # 1024 / 16 = 64
        "z_dim": 16,  # Based on decoder.proj_in.weight shape [1024, 16]
        "height": 256,  # Typical video frame height
        "width": 256,  # Typical video frame width
        "num_frames": 16,  # Typical number of frames
        "ffn_dim": 4 * 1024,  # Typical FFN expansion
        "num_layers": 24,  # 24 transformer blocks in encoder/decoder
        "eps": 1e-6,
    }

    # Create the diffusers VAE model
    vae = AutoencoderKLMagi1(
        patch_size=config["patch_size"],
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        z_dim=config["z_dim"],
        height=config["height"],
        width=config["width"],
        num_frames=config["num_frames"],
        ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"],
        eps=config["eps"],
    )

    # Convert and load the state dict
    converted_state_dict = convert_vae_state_dict(checkpoint)

    # Load the state dict
    missing_keys, unexpected_keys = vae.load_state_dict(converted_state_dict, strict=True)

    # if missing_keys:
    #     print(f"Missing keys in VAE: {missing_keys}")
    # if unexpected_keys:
    #     print(f"Unexpected keys in VAE: {unexpected_keys}")

    return vae


def convert_vae_state_dict(checkpoint):
    """
    Convert MAGI-1 VAE state dict to diffusers format.

    Maps the keys from the MAGI-1 VAE state dict to the diffusers VAE state dict.
    """
    state_dict = {}

    # Encoder mappings
    # Patch embedding (3D conv for video frames)
    if "encoder.patch_embed.proj.weight" in checkpoint:
        state_dict["encoder.patch_embedding.weight"] = checkpoint["encoder.patch_embed.proj.weight"]
        state_dict["encoder.patch_embedding.bias"] = checkpoint["encoder.patch_embed.proj.bias"]

    # Position embeddings
    if "encoder.pos_embed" in checkpoint:
        state_dict["encoder.pos_embed"] = checkpoint["encoder.pos_embed"]

    # Class token
    if "encoder.cls_token" in checkpoint:
        state_dict["encoder.cls_token"] = checkpoint["encoder.cls_token"]

    # Encoder blocks
    for i in range(24):  # 24 blocks in the encoder
        # Check if this block exists
        if f"encoder.blocks.{i}.attn.qkv.weight" not in checkpoint:
            continue

        # Attention components - split qkv into separate q, k, v
        if f"encoder.blocks.{i}.attn.qkv.weight" in checkpoint:
            qkv_weight = checkpoint[f"encoder.blocks.{i}.attn.qkv.weight"]
            qkv_bias = checkpoint[f"encoder.blocks.{i}.attn.qkv.bias"]

            # Split qkv into q, k, v (assuming equal splits)
            # hidden_size = qkv_weight.shape[1]  # 1024
            # head_dim = qkv_weight.shape[0] // 3  # 3072 // 3 = 1024

            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
            q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

            state_dict[f"encoder.blocks.{i}.attn.to_q.weight"] = q_weight
            state_dict[f"encoder.blocks.{i}.attn.to_q.bias"] = q_bias
            state_dict[f"encoder.blocks.{i}.attn.to_k.weight"] = k_weight
            state_dict[f"encoder.blocks.{i}.attn.to_k.bias"] = k_bias
            state_dict[f"encoder.blocks.{i}.attn.to_v.weight"] = v_weight
            state_dict[f"encoder.blocks.{i}.attn.to_v.bias"] = v_bias

        # Attention output projection
        if f"encoder.blocks.{i}.attn.proj.weight" in checkpoint:
            state_dict[f"encoder.blocks.{i}.attn.to_out.0.weight"] = checkpoint[f"encoder.blocks.{i}.attn.proj.weight"]
            state_dict[f"encoder.blocks.{i}.attn.to_out.0.bias"] = checkpoint[f"encoder.blocks.{i}.attn.proj.bias"]

        # Normalization (pre-MLP norm)
        if f"encoder.blocks.{i}.norm2.weight" in checkpoint:
            state_dict[f"encoder.blocks.{i}.norm2.weight"] = checkpoint[f"encoder.blocks.{i}.norm2.weight"]
            state_dict[f"encoder.blocks.{i}.norm2.bias"] = checkpoint[f"encoder.blocks.{i}.norm2.bias"]

        # MLP components (mapped to proj_out FeedForward)
        if f"encoder.blocks.{i}.mlp.fc1.weight" in checkpoint:
            state_dict[f"encoder.blocks.{i}.proj_out.net.0.proj.weight"] = checkpoint[
                f"encoder.blocks.{i}.mlp.fc1.weight"
            ]
            state_dict[f"encoder.blocks.{i}.proj_out.net.0.proj.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc1.bias"]
        if f"encoder.blocks.{i}.mlp.fc2.weight" in checkpoint:
            state_dict[f"encoder.blocks.{i}.proj_out.net.2.weight"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.weight"]
            # Note: fc2 typically doesn't have bias in FeedForward
            if f"encoder.blocks.{i}.mlp.fc2.bias" in checkpoint:
                state_dict[f"encoder.blocks.{i}.proj_out.net.2.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.bias"]

    # Encoder norm
    if "encoder.norm.weight" in checkpoint:
        state_dict["encoder.norm_out.weight"] = checkpoint["encoder.norm.weight"]
        state_dict["encoder.norm_out.bias"] = checkpoint["encoder.norm.bias"]

    # Encoder last layer (projection to latent space)
    if "encoder.last_layer.weight" in checkpoint:
        state_dict["encoder.linear_out.weight"] = checkpoint["encoder.last_layer.weight"]
        state_dict["encoder.linear_out.bias"] = checkpoint["encoder.last_layer.bias"]

    # Decoder mappings
    # Projection from latent space
    if "decoder.proj_in.weight" in checkpoint:
        state_dict["decoder.proj_in.weight"] = checkpoint["decoder.proj_in.weight"]
        state_dict["decoder.proj_in.bias"] = checkpoint["decoder.proj_in.bias"]

    # Position embeddings
    if "decoder.pos_embed" in checkpoint:
        state_dict["decoder.pos_embed"] = checkpoint["decoder.pos_embed"]

    # Class token
    if "decoder.cls_token" in checkpoint:
        state_dict["decoder.cls_token"] = checkpoint["decoder.cls_token"]

    # Decoder blocks
    for i in range(24):  # 24 blocks in the decoder
        # Check if this block exists
        if f"decoder.blocks.{i}.attn.qkv.weight" not in checkpoint:
            continue

        # Attention components - split qkv into separate q, k, v
        if f"decoder.blocks.{i}.attn.qkv.weight" in checkpoint:
            qkv_weight = checkpoint[f"decoder.blocks.{i}.attn.qkv.weight"]
            qkv_bias = checkpoint[f"decoder.blocks.{i}.attn.qkv.bias"]

            # Split qkv into q, k, v (assuming equal splits)
            # hidden_size = qkv_weight.shape[1]  # 1024
            # head_dim = qkv_weight.shape[0] // 3  # 3072 // 3 = 1024

            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
            q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

            state_dict[f"decoder.blocks.{i}.attn.to_q.weight"] = q_weight
            state_dict[f"decoder.blocks.{i}.attn.to_q.bias"] = q_bias
            state_dict[f"decoder.blocks.{i}.attn.to_k.weight"] = k_weight
            state_dict[f"decoder.blocks.{i}.attn.to_k.bias"] = k_bias
            state_dict[f"decoder.blocks.{i}.attn.to_v.weight"] = v_weight
            state_dict[f"decoder.blocks.{i}.attn.to_v.bias"] = v_bias

        # Attention output projection
        if f"decoder.blocks.{i}.attn.proj.weight" in checkpoint:
            state_dict[f"decoder.blocks.{i}.attn.to_out.0.weight"] = checkpoint[f"decoder.blocks.{i}.attn.proj.weight"]
            state_dict[f"decoder.blocks.{i}.attn.to_out.0.bias"] = checkpoint[f"decoder.blocks.{i}.attn.proj.bias"]

        # Normalization (pre-MLP norm)
        if f"decoder.blocks.{i}.norm2.weight" in checkpoint:
            state_dict[f"decoder.blocks.{i}.norm2.weight"] = checkpoint[f"decoder.blocks.{i}.norm2.weight"]
            state_dict[f"decoder.blocks.{i}.norm2.bias"] = checkpoint[f"decoder.blocks.{i}.norm2.bias"]

        # MLP components (mapped to proj_out FeedForward)
        if f"decoder.blocks.{i}.mlp.fc1.weight" in checkpoint:
            state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.weight"] = checkpoint[
                f"decoder.blocks.{i}.mlp.fc1.weight"
            ]
            state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.bias"]
        if f"decoder.blocks.{i}.mlp.fc2.weight" in checkpoint:
            state_dict[f"decoder.blocks.{i}.proj_out.net.2.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.weight"]
            # Note: fc2 typically doesn't have bias in FeedForward
            if f"decoder.blocks.{i}.mlp.fc2.bias" in checkpoint:
                state_dict[f"decoder.blocks.{i}.proj_out.net.2.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.bias"]

    # Decoder norm
    if "decoder.norm.weight" in checkpoint:
        state_dict["decoder.norm_out.weight"] = checkpoint["decoder.norm.weight"]
        state_dict["decoder.norm_out.bias"] = checkpoint["decoder.norm.bias"]

    # Decoder last layer (projection to pixel space - 3D conv)
    if "decoder.last_layer.weight" in checkpoint:
        state_dict["decoder.conv_out.weight"] = checkpoint["decoder.last_layer.weight"]
        state_dict["decoder.conv_out.bias"] = checkpoint["decoder.last_layer.bias"]

    # Note: Original MAGI-1 VAE checkpoint does not contain quantization layers
    # (quant_linear, post_quant_linear, quant_conv, post_quant_conv)

    return state_dict


def load_magi_transformer_checkpoint(checkpoint_path):
    """
    Load a MAGI-1 transformer checkpoint.

    Args:
        checkpoint_path: Path to the MAGI-1 transformer checkpoint.

    Returns:
        The loaded checkpoint state dict.
    """
    if checkpoint_path.endswith(".safetensors"):
        # Load safetensors file directly
        state_dict = load_file(checkpoint_path)
    elif os.path.isdir(checkpoint_path):
        # Check for sharded safetensors files
        safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
        if safetensors_files:
            # Load and merge sharded safetensors files
            state_dict = {}
            for safetensors_file in sorted(safetensors_files):  # Sort to ensure consistent order
                file_path = os.path.join(checkpoint_path, safetensors_file)
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        else:
            # Try loading PyTorch checkpoint
            checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt") or f.endswith(".pth")]
            if not checkpoint_files:
                raise ValueError(f"No checkpoint files found in {checkpoint_path}")

            checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[0])
            checkpoint_data = torch.load(checkpoint_file, map_location="cpu")

            # Handle different checkpoint formats
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
        # Try loading PyTorch checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
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


def convert_magi_transformer_checkpoint(checkpoint_path, transformer_config_file=None, dtype=None):
    """
    Convert a MAGI-1 transformer checkpoint to a diffusers Magi1Transformer3DModel.

    Args:
        checkpoint_path: Path to the MAGI-1 transformer checkpoint.
        transformer_config_file: Optional path to a transformer config file.
        dtype: Optional dtype for the model.

    Returns:
        A diffusers Magi1Transformer3DModel model.
    """
    if transformer_config_file is not None:
        with open(transformer_config_file, "r") as f:
            config = json.load(f)
    else:
        # Default config for MAGI-1 transformer based on the full parameter list
        config = {
            "in_channels": 16,  # Must match VAE latent channels
            "out_channels": 16,  # Must match VAE latent channels
            "num_layers": 34,  # Based on the full parameter list (0-33)
            "num_attention_heads": 16,
            "attention_head_dim": 64,
            "cross_attention_dim": 4096,  # T5 hidden size
            "patch_size": [1, 2, 2],
            "use_linear_projection": True,
            "upcast_attention": False,
        }

    # Create the diffusers transformer model
    transformer = Magi1Transformer3DModel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_layers=config["num_layers"],
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        cross_attention_dim=config["cross_attention_dim"],
        patch_size=config["patch_size"],
        use_linear_projection=config["use_linear_projection"],
        upcast_attention=config["upcast_attention"],
    )

    # Load the checkpoint
    checkpoint = load_magi_transformer_checkpoint(checkpoint_path)

    # Convert and load the state dict
    converted_state_dict = convert_transformer_state_dict(checkpoint)

    # Load the state dict
    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=True)

    if missing_keys:
        print(f"Missing keys in transformer: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in transformer: {unexpected_keys}")

    if dtype is not None:
        transformer = transformer.to(dtype=dtype)

    return transformer


def convert_transformer_state_dict(checkpoint):
    """
    Convert MAGI-1 transformer state dict to diffusers format.

    Uses a key mapping approach similar to WAN conversion for consistency.
    Maps the original MAGI-1 parameter names to diffusers' standard transformer naming.
    """
    original_state_dict = {}

    # First, load all parameters into a flat dictionary
    for key, value in checkpoint.items():
        original_state_dict[key] = value

    # Apply systematic key renaming using the mapping dictionary
    for key in list(original_state_dict.keys()):
        new_key = key[:]

        # Apply transformer block-specific mappings
        if "videodit_blocks.layers." in key:
            # Extract layer number
            parts = key.split(".")
            if len(parts) >= 4 and parts[0] == "videodit_blocks" and parts[1] == "layers":
                layer_num = parts[2]
                remaining_key = ".".join(parts[3:])

                # Map to blocks.{layer_num}.{component}
                for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
                    if remaining_key.startswith(replace_key):
                        suffix = remaining_key[len(replace_key):]
                        new_key = f"blocks.{layer_num}.{rename_key}{suffix}"
                        break
        else:
            # Apply global mappings for non-block components
            for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
                new_key = new_key.replace(replace_key, rename_key)

        # Update the state dict
        if new_key != key:
            original_state_dict[new_key] = original_state_dict.pop(key)

    # Handle special cases that need custom processing
    processed_keys = set()

    # Process each transformer block separately for special handling
    num_layers = 34  # MAGI-1 has 34 layers
    for i in range(num_layers):
        layer_prefix = f"videodit_blocks.layers.{i}"
        block_prefix = f"blocks.{i}"

        # Check if this layer exists
        if f"{layer_prefix}.self_attention.linear_qkv.q.weight" not in checkpoint:
            continue

        # Handle cross-attention K,V splitting
        kv_key = f"{layer_prefix}.self_attention.linear_kv_xattn.weight"
        if kv_key in checkpoint:
            kv_weight = checkpoint[kv_key]
            # Split into separate K and V projections
            k_weight, v_weight = kv_weight.chunk(2, dim=0)
            original_state_dict[f"{block_prefix}.attn2.to_k.weight"] = k_weight
            original_state_dict[f"{block_prefix}.attn2.to_v.weight"] = v_weight
            # Mark original key for removal
            if f"{block_prefix}.attn2.to_kv.weight" in original_state_dict:
                del original_state_dict[f"{block_prefix}.attn2.to_kv.weight"]

        # Share output projection between attn1 and attn2 (original MAGI-1 pattern)
        attn1_out_key = f"{block_prefix}.attn1.to_out.0.weight"
        if attn1_out_key in original_state_dict:
            original_state_dict[f"{block_prefix}.attn2.to_out.0.weight"] = original_state_dict[attn1_out_key]

    # Handle time projection mapping (need to avoid duplicate mapping)
    if "condition_embedder.time_embedder.linear_2.weight" in original_state_dict:
        original_state_dict["condition_embedder.time_proj.weight"] = original_state_dict["condition_embedder.time_embedder.linear_2.weight"]
        original_state_dict["condition_embedder.time_proj.bias"] = original_state_dict["condition_embedder.time_embedder.linear_2.bias"]

    return original_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16", "none"])
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

if __name__ == "__main__":
    args = get_args()

    transformer = convert_magi_transformer(args.model_type)
    #vae = convert_magi_vae()
    #text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl")
    #tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl")
    # flow_shift = 16.0 if "FLF2V" in args.model_type else 3.0
    # scheduler = UniPCMultistepScheduler(
    #     prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
    # )

    # If user has specified "none", we keep the original dtypes of the state dict without any conversion
    if args.dtype != "none":
        dtype = DTYPE_MAPPING[args.dtype]
        transformer.to(dtype)

    # if "I2V" in args.model_type or "FLF2V" in args.model_type:
        # image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.bfloat16
        # )
        # image_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        # pipe = Magi1ImageToVideoPipeline(
        #     transformer=transformer,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     vae=vae,
        #     scheduler=scheduler,
        #     image_encoder=image_encoder,
        #     image_processor=image_processor,
        # )
    # else:
    pipe = Magi1Pipeline(
        transformer=transformer,
        text_encoder=None,#text_encoder,
        tokenizer=None,#tokenizer,
        vae=None,#vae,
        scheduler=None,#scheduler,
    )

    pipe.save_pretrained(args.output_path,
                         safe_serialization=True,
                         max_shard_size="5GB",
                         push_to_hub=True,
                         repo_id=f"tolgacangoz/{args.model_type}-Diffusers",
                         )