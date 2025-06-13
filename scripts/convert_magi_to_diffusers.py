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

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import (
    AutoencoderKLMagi,
    FlowMatchEulerDiscreteScheduler,
    MagiPipeline,
    MagiTransformer3DModel,
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


def convert_magi_vae_checkpoint(checkpoint_path, vae_config_file=None, dtype=None):
    """
    Convert a MAGI-1 VAE checkpoint to a diffusers AutoencoderKLMagi.

    Args:
        checkpoint_path: Path to the MAGI-1 VAE checkpoint.
        vae_config_file: Optional path to a VAE config file.
        dtype: Optional dtype for the model.

    Returns:
        A diffusers AutoencoderKLMagi model.
    """
    if vae_config_file is not None:
        with open(vae_config_file, "r") as f:
            config = json.load(f)
    else:
        # Default config for MAGI-1 VAE based on the checkpoint structure
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 16,  # Based on encoder.last_layer.weight shape [32, 1024] -> 16 channels (32/2)
            "block_out_channels": [1024],  # Hidden dimension in transformer blocks
            "layers_per_block": 24,  # 24 transformer blocks in encoder/decoder
            "act_fn": "gelu",
            "norm_num_groups": 32,
            "scaling_factor": 0.18215,
            "sample_size": 256,  # Typical image size
        }

    # Create the diffusers VAE model
    vae = AutoencoderKLMagi(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        latent_channels=config["latent_channels"],
        layers_per_block=config["layers_per_block"],
        block_out_channels=config["block_out_channels"],
        act_fn=config["act_fn"],
        norm_num_groups=config["norm_num_groups"],
        scaling_factor=config["scaling_factor"],
        sample_size=config["sample_size"],
    )

    # Load the checkpoint
    if checkpoint_path.endswith(".safetensors"):
        # Load safetensors file
        checkpoint = load_file(checkpoint_path)
    else:
        # Load PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Convert and load the state dict
    converted_state_dict = convert_vae_state_dict(checkpoint)

    # Load the state dict
    missing_keys, unexpected_keys = vae.load_state_dict(converted_state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys in VAE: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in VAE: {unexpected_keys}")

    if dtype is not None:
        vae = vae.to(dtype=dtype)

    return vae


def convert_vae_state_dict(checkpoint):
    """
    Convert MAGI-1 VAE state dict to diffusers format.

    Maps the keys from the MAGI-1 VAE state dict to the diffusers VAE state dict.
    """
    state_dict = {}

    # Encoder mappings
    # Patch embedding
    if "encoder.patch_embed.proj.weight" in checkpoint:
        state_dict["encoder.conv_in.weight"] = checkpoint["encoder.patch_embed.proj.weight"]
        state_dict["encoder.conv_in.bias"] = checkpoint["encoder.patch_embed.proj.bias"]

    # Position embeddings
    if "encoder.pos_embed" in checkpoint:
        state_dict["encoder.pos_embed"] = checkpoint["encoder.pos_embed"]

    # Class token
    if "encoder.cls_token" in checkpoint:
        state_dict["encoder.class_embedding"] = checkpoint["encoder.cls_token"]

    # Encoder blocks
    for i in range(24):  # Assuming 24 blocks in the encoder
        # Check if this block exists
        if f"encoder.blocks.{i}.attn.qkv.weight" not in checkpoint:
            continue

        # Attention components
        state_dict[f"encoder.transformer_blocks.{i}.attn1.to_qkv.weight"] = checkpoint[f"encoder.blocks.{i}.attn.qkv.weight"]
        state_dict[f"encoder.transformer_blocks.{i}.attn1.to_qkv.bias"] = checkpoint[f"encoder.blocks.{i}.attn.qkv.bias"]
        state_dict[f"encoder.transformer_blocks.{i}.attn1.to_out.0.weight"] = checkpoint[f"encoder.blocks.{i}.attn.proj.weight"]
        state_dict[f"encoder.transformer_blocks.{i}.attn1.to_out.0.bias"] = checkpoint[f"encoder.blocks.{i}.attn.proj.bias"]

        # Normalization
        state_dict[f"encoder.transformer_blocks.{i}.norm2.weight"] = checkpoint[f"encoder.blocks.{i}.norm2.weight"]
        state_dict[f"encoder.transformer_blocks.{i}.norm2.bias"] = checkpoint[f"encoder.blocks.{i}.norm2.bias"]

        # MLP components
        state_dict[f"encoder.transformer_blocks.{i}.ff.net.0.proj.weight"] = checkpoint[f"encoder.blocks.{i}.mlp.fc1.weight"]
        state_dict[f"encoder.transformer_blocks.{i}.ff.net.0.proj.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc1.bias"]
        state_dict[f"encoder.transformer_blocks.{i}.ff.net.2.weight"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.weight"]
        state_dict[f"encoder.transformer_blocks.{i}.ff.net.2.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.bias"]

    # Encoder norm
    if "encoder.norm.weight" in checkpoint:
        state_dict["encoder.norm_out.weight"] = checkpoint["encoder.norm.weight"]
        state_dict["encoder.norm_out.bias"] = checkpoint["encoder.norm.bias"]

    # Encoder last layer (projection to latent space)
    if "encoder.last_layer.weight" in checkpoint:
        state_dict["encoder.conv_out.weight"] = checkpoint["encoder.last_layer.weight"]
        state_dict["encoder.conv_out.bias"] = checkpoint["encoder.last_layer.bias"]

    # Decoder mappings
    # Projection from latent space
    if "decoder.proj_in.weight" in checkpoint:
        state_dict["decoder.conv_in.weight"] = checkpoint["decoder.proj_in.weight"]
        state_dict["decoder.conv_in.bias"] = checkpoint["decoder.proj_in.bias"]

    # Position embeddings
    if "decoder.pos_embed" in checkpoint:
        state_dict["decoder.pos_embed"] = checkpoint["decoder.pos_embed"]

    # Class token
    if "decoder.cls_token" in checkpoint:
        state_dict["decoder.class_embedding"] = checkpoint["decoder.cls_token"]

    # Decoder blocks
    for i in range(24):  # Assuming 24 blocks in the decoder
        # Check if this block exists
        if f"decoder.blocks.{i}.attn.qkv.weight" not in checkpoint:
            continue

        # Attention components
        state_dict[f"decoder.transformer_blocks.{i}.attn1.to_qkv.weight"] = checkpoint[f"decoder.blocks.{i}.attn.qkv.weight"]
        state_dict[f"decoder.transformer_blocks.{i}.attn1.to_qkv.bias"] = checkpoint[f"decoder.blocks.{i}.attn.qkv.bias"]
        state_dict[f"decoder.transformer_blocks.{i}.attn1.to_out.0.weight"] = checkpoint[f"decoder.blocks.{i}.attn.proj.weight"]
        state_dict[f"decoder.transformer_blocks.{i}.attn1.to_out.0.bias"] = checkpoint[f"decoder.blocks.{i}.attn.proj.bias"]

        # Normalization
        state_dict[f"decoder.transformer_blocks.{i}.norm2.weight"] = checkpoint[f"decoder.blocks.{i}.norm2.weight"]
        state_dict[f"decoder.transformer_blocks.{i}.norm2.bias"] = checkpoint[f"decoder.blocks.{i}.norm2.bias"]

        # MLP components
        state_dict[f"decoder.transformer_blocks.{i}.ff.net.0.proj.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.weight"]
        state_dict[f"decoder.transformer_blocks.{i}.ff.net.0.proj.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.bias"]
        state_dict[f"decoder.transformer_blocks.{i}.ff.net.2.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.weight"]
        state_dict[f"decoder.transformer_blocks.{i}.ff.net.2.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.bias"]

    # Decoder norm
    if "decoder.norm.weight" in checkpoint:
        state_dict["decoder.norm_out.weight"] = checkpoint["decoder.norm.weight"]
        state_dict["decoder.norm_out.bias"] = checkpoint["decoder.norm.bias"]

    # Decoder last layer (projection to pixel space)
    if "decoder.last_layer.weight" in checkpoint:
        state_dict["decoder.conv_out.weight"] = checkpoint["decoder.last_layer.weight"]
        state_dict["decoder.conv_out.bias"] = checkpoint["decoder.last_layer.bias"]

    # Quant conv (encoder output to latent distribution)
    if "quant_conv.weight" in checkpoint:
        state_dict["quant_conv.weight"] = checkpoint["quant_conv.weight"]
        state_dict["quant_conv.bias"] = checkpoint["quant_conv.bias"]

    # Post quant conv (latent to decoder input)
    if "post_quant_conv.weight" in checkpoint:
        state_dict["post_quant_conv.weight"] = checkpoint["post_quant_conv.weight"]
        state_dict["post_quant_conv.bias"] = checkpoint["post_quant_conv.bias"]

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
            for safetensors_file in safetensors_files:
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
            state_dict = torch.load(checkpoint_file, map_location="cpu")
    else:
        # Try loading PyTorch checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    return state_dict


def convert_magi_transformer_checkpoint(checkpoint_path, transformer_config_file=None, dtype=None):
    """
    Convert a MAGI-1 transformer checkpoint to a diffusers MagiTransformer3DModel.

    Args:
        checkpoint_path: Path to the MAGI-1 transformer checkpoint.
        transformer_config_file: Optional path to a transformer config file.
        dtype: Optional dtype for the model.

    Returns:
        A diffusers MagiTransformer3DModel model.
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
    transformer = MagiTransformer3DModel(
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
        state_dict[f"transformer_blocks.{i}.ff_norm.weight"] = checkpoint[f"{layer_prefix}.ada_modulate_layer.proj.0.weight"]
        state_dict[f"transformer_blocks.{i}.ff_norm.bias"] = checkpoint[f"{layer_prefix}.ada_modulate_layer.proj.0.bias"]

        # Self-attention components

        # Query normalization
        if f"{layer_prefix}.self_attention.q_layernorm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_q.weight"] = checkpoint[f"{layer_prefix}.self_attention.q_layernorm.weight"]
            state_dict[f"transformer_blocks.{i}.attn1.norm_q.bias"] = checkpoint[f"{layer_prefix}.self_attention.q_layernorm.bias"]

        # Key normalization
        if f"{layer_prefix}.self_attention.k_layernorm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_k.weight"] = checkpoint[f"{layer_prefix}.self_attention.k_layernorm.weight"]
            state_dict[f"transformer_blocks.{i}.attn1.norm_k.bias"] = checkpoint[f"{layer_prefix}.self_attention.k_layernorm.bias"]

        # Cross-attention key normalization
        if f"{layer_prefix}.self_attention.k_layernorm_xattn.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_k_xattn.weight"] = checkpoint[f"{layer_prefix}.self_attention.k_layernorm_xattn.weight"]
            state_dict[f"transformer_blocks.{i}.attn1.norm_k_xattn.bias"] = checkpoint[f"{layer_prefix}.self_attention.k_layernorm_xattn.bias"]

        # Cross-attention query normalization
        if f"{layer_prefix}.self_attention.q_layernorm_xattn.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.norm_q_xattn.weight"] = checkpoint[f"{layer_prefix}.self_attention.q_layernorm_xattn.weight"]
            state_dict[f"transformer_blocks.{i}.attn1.norm_q_xattn.bias"] = checkpoint[f"{layer_prefix}.self_attention.q_layernorm_xattn.bias"]

        # QKV linear projections
        if f"{layer_prefix}.self_attention.linear_qkv.q.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_q.weight"] = checkpoint[f"{layer_prefix}.self_attention.linear_qkv.q.weight"]

        if f"{layer_prefix}.self_attention.linear_qkv.k.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_k.weight"] = checkpoint[f"{layer_prefix}.self_attention.linear_qkv.k.weight"]

        if f"{layer_prefix}.self_attention.linear_qkv.v.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_v.weight"] = checkpoint[f"{layer_prefix}.self_attention.linear_qkv.v.weight"]

        if f"{layer_prefix}.self_attention.linear_qkv.qx.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_q_xattn.weight"] = checkpoint[f"{layer_prefix}.self_attention.linear_qkv.qx.weight"]

        # QKV layer norm
        if f"{layer_prefix}.self_attention.linear_qkv.layer_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.qkv_norm.weight"] = checkpoint[f"{layer_prefix}.self_attention.linear_qkv.layer_norm.weight"]
            state_dict[f"transformer_blocks.{i}.attn1.qkv_norm.bias"] = checkpoint[f"{layer_prefix}.self_attention.linear_qkv.layer_norm.bias"]

        # KV cross-attention
        if f"{layer_prefix}.self_attention.linear_kv_xattn.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_kv_xattn.weight"] = checkpoint[f"{layer_prefix}.self_attention.linear_kv_xattn.weight"]

        # Output projection
        if f"{layer_prefix}.self_attention.linear_proj.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.attn1.to_out.0.weight"] = checkpoint[f"{layer_prefix}.self_attention.linear_proj.weight"]

        # Self-attention post normalization
        if f"{layer_prefix}.self_attn_post_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.norm1.weight"] = checkpoint[f"{layer_prefix}.self_attn_post_norm.weight"]
            state_dict[f"transformer_blocks.{i}.norm1.bias"] = checkpoint[f"{layer_prefix}.self_attn_post_norm.bias"]

        # MLP components
        # MLP layer norm
        if f"{layer_prefix}.mlp.layer_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.ff.norm.weight"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.weight"]
            state_dict[f"transformer_blocks.{i}.ff.norm.bias"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.bias"]

        # MLP FC1 (projection)
        if f"{layer_prefix}.mlp.linear_fc1.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = checkpoint[f"{layer_prefix}.mlp.linear_fc1.weight"]

        # MLP FC2 (projection)
        if f"{layer_prefix}.mlp.linear_fc2.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.ff.net.2.weight"] = checkpoint[f"{layer_prefix}.mlp.linear_fc2.weight"]

        # MLP post normalization
        if f"{layer_prefix}.mlp_post_norm.weight" in checkpoint:
            state_dict[f"transformer_blocks.{i}.norm2.weight"] = checkpoint[f"{layer_prefix}.mlp_post_norm.weight"]
            state_dict[f"transformer_blocks.{i}.norm2.bias"] = checkpoint[f"{layer_prefix}.mlp_post_norm.bias"]

    return state_dict


def convert_magi_checkpoint(
    magi_checkpoint_path,
    vae_checkpoint_path=None,
    transformer_checkpoint_path=None,
    t5_model_name="google/umt5-xxl",
    output_path=None,
    dtype=None,
):
    """
    Convert MAGI-1 checkpoints to a diffusers pipeline.

    Args:
        magi_checkpoint_path: Path to the MAGI-1 checkpoint directory.
        vae_checkpoint_path: Optional path to the VAE checkpoint.
        transformer_checkpoint_path: Optional path to the transformer checkpoint.
        t5_model_name: Name of the T5 model to use.
        output_path: Path to save the converted pipeline.
        dtype: Optional dtype for the models.

    Returns:
        A diffusers MagiPipeline.
    """
    # Load or convert the VAE
    if vae_checkpoint_path is None:
        vae_checkpoint_path = os.path.join(magi_checkpoint_path, "ckpt/vae")

    vae = convert_magi_vae_checkpoint(vae_checkpoint_path, dtype=dtype)

    # Load or convert the transformer
    if transformer_checkpoint_path is None:
        transformer_checkpoint_path = os.path.join(magi_checkpoint_path, "ckpt/magi/4.5B_base/inference_weight")

    transformer = convert_magi_transformer_checkpoint(transformer_checkpoint_path, dtype=dtype)

    # Load the text encoder and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    text_encoder = UMT5EncoderModel.from_pretrained(t5_model_name)

    if dtype is not None:
        text_encoder = text_encoder.to(dtype=dtype)

    # Create the scheduler
    scheduler = FlowMatchEulerDiscreteScheduler()

    # Create the pipeline
    pipeline = MagiPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )

    # Save the pipeline if output_path is provided
    if output_path is not None:
        pipeline.save_pretrained(output_path)

    return pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MAGI-1 checkpoints to diffusers format.")
    parser.add_argument(
        "--magi_checkpoint_path",
        type=str,
        required=True,
        help="Path to the MAGI-1 checkpoint directory.",
    )
    parser.add_argument(
        "--vae_checkpoint_path",
        type=str,
        help="Path to the VAE checkpoint. If not provided, will look in magi_checkpoint_path/ckpt/vae.",
    )
    parser.add_argument(
        "--transformer_checkpoint_path",
        type=str,
        help="Path to the transformer checkpoint. If not provided, will look in magi_checkpoint_path/ckpt/magi/4.5B_base.",
    )
    parser.add_argument(
        "--t5_model_name",
        type=str,
        default="google/umt5-xxl",
        help="Name of the T5 model to use.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted pipeline.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Data type for the models.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set the dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("Starting MAGI-1 conversion to diffusers format...")
    print(f"Output will be saved to: {args.output_path}")
    print(f"Using dtype: {args.dtype}")

    try:
        # Convert the VAE
        print("Converting VAE checkpoint...")
        if args.vae_checkpoint_path:
            vae_path = args.vae_checkpoint_path
        else:
            vae_path = os.path.join(args.magi_checkpoint_path, "ckpt/vae/diffusion_pytorch_model.safetensors")
            if not os.path.exists(vae_path):
                vae_path = os.path.join(args.magi_checkpoint_path, "ckpt/vae")

        print(f"VAE checkpoint path: {vae_path}")
        vae = convert_magi_vae_checkpoint(vae_path, dtype=dtype)
        print("VAE conversion complete.")

        # Convert the transformer
        print("Converting transformer checkpoint...")
        if args.transformer_checkpoint_path:
            transformer_path = args.transformer_checkpoint_path
        else:
            transformer_path = os.path.join(args.magi_checkpoint_path, "ckpt/magi/4.5B_base/inference_weight")

        print(f"Transformer checkpoint path: {transformer_path}")
        transformer = convert_magi_transformer_checkpoint(transformer_path, dtype=dtype)
        print("Transformer conversion complete.")

        # Load the text encoder and tokenizer
        print(f"Loading text encoder and tokenizer from {args.t5_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.t5_model_name)
        text_encoder = UMT5EncoderModel.from_pretrained(args.t5_model_name)

        if dtype is not None:
            text_encoder = text_encoder.to(dtype=dtype)
        print("Text encoder and tokenizer loaded successfully.")

        # Create the scheduler
        print("Creating scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler()
        print("Scheduler created successfully.")

        # Create the pipeline
        print("Creating MAGI pipeline...")
        pipeline = MagiPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        print("MAGI pipeline created successfully.")

        # Save the pipeline
        print(f"Saving pipeline to {args.output_path}...")
        pipeline.save_pretrained(args.output_path)
        print("Pipeline saved successfully.")

        print(f"Conversion complete! MAGI-1 pipeline saved to {args.output_path}")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
