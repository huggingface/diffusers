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

"""Convert MAGI-1 checkpoints to diffusers format."""

import argparse
import json
import os
from huggingface_hub import hf_hub_download

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLMagi1,
    FlowMatchEulerDiscreteScheduler,
    #Magi1Transformer3DModel,
    Magi1Pipeline,
)


# Mapping dictionary for transformer weights
TRANSFORMER_KEYS_RENAME_DICT = {
    "t_embedder.mlp.0": "time_embedding.0",
    "t_embedder.mlp.2": "time_embedding.2",
    "y_embedder.y_proj_adaln.0": "text_embedding.0",
    "y_embedder.y_proj_xattn.0": "cross_attention_proj",
    "y_embedder.null_caption_embedding": "null_caption_embedding",
    "rope.bands": "rotary_emb.bands",
    "videodit_blocks.final_layernorm": "transformer_blocks.norm_final",
    "final_linear.linear": "proj_out",
}

# Layer-specific mappings
LAYER_KEYS_RENAME_DICT = {
    "ada_modulate_layer.proj.0": "ff_norm",
    "self_attention.linear_kv_xattn": "attn1.to_kv",
    "self_attention.linear_proj": "attn1.to_out",
    "mlp.linear_fc1": "ff.net.0.proj",
}


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
    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=False)

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

    Maps the keys from the MAGI-1 transformer state dict to the diffusers transformer state dict.
    """
    state_dict = {}

    # Process input projection
    if "x_embedder.weight" in checkpoint:
        state_dict["input_proj.weight"] = checkpoint["x_embedder.weight"]

    # Process time embedding
    if "t_embedder.mlp.0.weight" in checkpoint:
        state_dict["time_embedding.0.weight"] = checkpoint["t_embedder.mlp.0.weight"]
        state_dict["time_embedding.0.bias"] = checkpoint["t_embedder.mlp.0.bias"]
        state_dict["time_embedding.2.weight"] = checkpoint["t_embedder.mlp.2.weight"]
        state_dict["time_embedding.2.bias"] = checkpoint["t_embedder.mlp.2.bias"]

    # Process text embedding
    if "y_embedder.y_proj_adaln.0.weight" in checkpoint:
        state_dict["text_embedding.0.weight"] = checkpoint["y_embedder.y_proj_adaln.0.weight"]
        state_dict["text_embedding.0.bias"] = checkpoint["y_embedder.y_proj_adaln.0.bias"]

    if "y_embedder.y_proj_xattn.0.weight" in checkpoint:
        state_dict["cross_attention_proj.weight"] = checkpoint["y_embedder.y_proj_xattn.0.weight"]
        state_dict["cross_attention_proj.bias"] = checkpoint["y_embedder.y_proj_xattn.0.bias"]

    # Process null caption embedding
    if "y_embedder.null_caption_embedding" in checkpoint:
        state_dict["null_caption_embedding"] = checkpoint["y_embedder.null_caption_embedding"]

    # Process rotary embedding
    if "rope.bands" in checkpoint:
        state_dict["rotary_emb.bands"] = checkpoint["rope.bands"]

    # Process final layer norm
    if "videodit_blocks.final_layernorm.weight" in checkpoint:
        state_dict["transformer_blocks.norm_final.weight"] = checkpoint["videodit_blocks.final_layernorm.weight"]
        state_dict["transformer_blocks.norm_final.bias"] = checkpoint["videodit_blocks.final_layernorm.bias"]

    # Process final linear projection
    if "final_linear.linear.weight" in checkpoint:
        state_dict["proj_out.weight"] = checkpoint["final_linear.linear.weight"]

    # Process transformer blocks
    # Based on the full parameter list, there are 34 layers (0-33)
    num_layers = 34
    for i in range(num_layers):
        # Check if this layer exists in the checkpoint
        layer_prefix = f"videodit_blocks.layers.{i}"
        if f"{layer_prefix}.ada_modulate_layer.proj.0.weight" not in checkpoint:
            continue

        # FF norm (AdaLN projection)
        state_dict[f"transformer_blocks.{i}.ff_norm.weight"] = checkpoint[
            f"{layer_prefix}.ada_modulate_layer.proj.0.weight"
        ]
        state_dict[f"transformer_blocks.{i}.ff_norm.bias"] = checkpoint[
            f"{layer_prefix}.ada_modulate_layer.proj.0.bias"
        ]

        # Self-attention components

        # Query normalization
        if f"{layer_prefix}.self_attention.q_layernorm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_q.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.q_layernorm.weight"
            ]
            state_dict[f"transformer_blocks.{i}.attn1.norm_q.bias"] = checkpoint[
                f"{layer_prefix}.self_attention.q_layernorm.bias"
            ]

        # Key normalization
        if f"{layer_prefix}.self_attention.k_layernorm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_k.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.k_layernorm.weight"
            ]
            state_dict[f"transformer_blocks.{i}.attn1.norm_k.bias"] = checkpoint[
                f"{layer_prefix}.self_attention.k_layernorm.bias"
            ]

        # Cross-attention key normalization
        if f"{layer_prefix}.self_attention.k_layernorm_xattn.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_k_xattn.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.k_layernorm_xattn.weight"
            ]
            state_dict[f"transformer_blocks.{i}.attn1.norm_k_xattn.bias"] = checkpoint[
                f"{layer_prefix}.self_attention.k_layernorm_xattn.bias"
            ]

        # Cross-attention query normalization
        if f"{layer_prefix}.self_attention.q_layernorm_xattn.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_q_xattn.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.q_layernorm_xattn.weight"
            ]
            state_dict[f"transformer_blocks.{i}.attn1.norm_q_xattn.bias"] = checkpoint[
                f"{layer_prefix}.self_attention.q_layernorm_xattn.bias"
            ]

        # QKV linear projections
        if f"{layer_prefix}.self_attention.linear_qkv.q.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_q.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_qkv.q.weight"
            ]

        if f"{layer_prefix}.self_attention.linear_qkv.k.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_k.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_qkv.k.weight"
            ]

        if f"{layer_prefix}.self_attention.linear_qkv.v.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_v.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_qkv.v.weight"
            ]

        if f"{layer_prefix}.self_attention.linear_qkv.qx.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_q_xattn.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_qkv.qx.weight"
            ]

        # QKV layer norm
        if f"{layer_prefix}.self_attention.linear_qkv.layer_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.qkv_norm.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_qkv.layer_norm.weight"
            ]
            state_dict[f"transformer_blocks.{i}.attn1.qkv_norm.bias"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_qkv.layer_norm.bias"
            ]

        # KV cross-attention
        if f"{layer_prefix}.self_attention.linear_kv_xattn.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_kv_xattn.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_kv_xattn.weight"
            ]

        # Output projection
        if f"{layer_prefix}.self_attention.linear_proj.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_out.0.weight"] = checkpoint[
                f"{layer_prefix}.self_attention.linear_proj.weight"
            ]

        # Self-attention post normalization
        if f"{layer_prefix}.self_attn_post_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.norm1.weight"] = checkpoint[
                f"{layer_prefix}.self_attn_post_norm.weight"
            ]
            state_dict[f"transformer_blocks.{i}.norm1.bias"] = checkpoint[f"{layer_prefix}.self_attn_post_norm.bias"]

        # MLP components
        # MLP layer norm
        if f"{layer_prefix}.mlp.layer_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.ff.norm.weight"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.weight"]
            state_dict[f"transformer_blocks.{i}.ff.norm.bias"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.bias"]

        # MLP FC1 (projection)
        if f"{layer_prefix}.mlp.linear_fc1.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = checkpoint[
                f"{layer_prefix}.mlp.linear_fc1.weight"
            ]

        # MLP FC2 (projection)
        if f"{layer_prefix}.mlp.linear_fc2.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.ff.net.2.weight"] = checkpoint[f"{layer_prefix}.mlp.linear_fc2.weight"]

        # MLP post normalization
        if f"{layer_prefix}.mlp_post_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.norm2.weight"] = checkpoint[f"{layer_prefix}.mlp_post_norm.weight"]
            state_dict[f"transformer_blocks.{i}.norm2.bias"] = checkpoint[f"{layer_prefix}.mlp_post_norm.bias"]

    return state_dict


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

    # transformer = convert_transformer(args.model_type)
    #vae = convert_magi_vae()
    text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl")
    tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl")
    # flow_shift = 16.0 if "FLF2V" in args.model_type else 3.0
    # scheduler = UniPCMultistepScheduler(
    #     prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
    # )

    # If user has specified "none", we keep the original dtypes of the state dict without any conversion
    # if args.dtype != "none":
    #     dtype = DTYPE_MAPPING[args.dtype]
    #     transformer.to(dtype)

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
        transformer=None,#transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=None,#vae,
        scheduler=None,#scheduler,
    )

    pipe.save_pretrained(args.output_path,
                         safe_serialization=True,
                         max_shard_size="5GB",
                         push_to_hub=True,
                         repo_id=f"tolgacangoz/{args.model_type}-Diffusers",
                         )