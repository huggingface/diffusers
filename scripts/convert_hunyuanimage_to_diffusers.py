#!/usr/bin/env python3
# Copyright 2025 Tencent Hunyuan Team and The HuggingFace Team. All rights reserved.
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
Conversion script for HunyuanImage 2.1 models to Diffusers format.

Usage:
    python convert_hunyuanimage_to_diffusers.py \\
        --transformer_checkpoint_path /path/to/hunyuanimage_dit.pt \\
        --vae_checkpoint_path /path/to/hunyuanimage_vae.pt \\
        --output_path /path/to/output \\
        --model_type hunyuanimage-v2.1

Supported model types:
    - hunyuanimage-v2.1: Base model (50 steps, no guidance embedding)
    - hunyuanimage-v2.1-distilled: Distilled model (8 steps, guidance embedding, MeanFlow)
"""

import argparse
import os
from typing import Dict

import torch
from safetensors.torch import load_file as safetensors_load_file
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import (
    AutoencoderKLHunyuanImage,
    FlowMatchEulerDiscreteScheduler,
    HunyuanImage2DModel,
    HunyuanImagePipeline,
)


def load_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint from either safetensors or pt format."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if checkpoint_path.endswith(".safetensors"):
        return safetensors_load_file(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle nested checkpoint structure
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint


def convert_transformer_state_dict(state_dict: Dict[str, torch.Tensor], model_type: str) -> Dict[str, torch.Tensor]:
    """
    Convert transformer weights from official format to diffusers format.
    
    Key mappings:
    - double_blocks.{i}.attn_q -> double_blocks.{i}.img_attn_q
    - double_blocks.{i}.attn_k -> double_blocks.{i}.img_attn_k
    - double_blocks.{i}.attn_v -> double_blocks.{i}.img_attn_v
    - single_blocks.{i}.linear1_q -> single_blocks.{i}.linear1_q (no change)
    - img_in -> pos_embed
    - txt_in -> text_embedder
    - time_in -> time_embedder
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Handle patch embedding
        if key.startswith("img_in."):
            new_key = key.replace("img_in.", "pos_embed.")
        
        # Handle text embedding
        elif key.startswith("txt_in."):
            new_key = key.replace("txt_in.", "text_embedder.")
        
        # Handle time embedding
        elif key.startswith("time_in."):
            new_key = key.replace("time_in.mlp.", "time_embedder.linear_")
            # Adjust numbering: 0 -> 1, 2 -> 2
            if "mlp.0." in key:
                new_key = new_key.replace("time_embedder.linear_0.", "time_embedder.linear_1.")
            elif "mlp.2." in key:
                new_key = new_key.replace("time_embedder.linear_2.", "time_embedder.linear_2.")
        
        # Handle MeanFlow time_r embedding
        elif key.startswith("time_r_in.") and "distilled" in model_type:
            new_key = key.replace("time_r_in.mlp.", "time_r_embedder.linear_")
            if "mlp.0." in key:
                new_key = new_key.replace("time_r_embedder.linear_0.", "time_r_embedder.linear_1.")
            elif "mlp.2." in key:
                new_key = new_key.replace("time_r_embedder.linear_2.", "time_r_embedder.linear_2.")
        
        # Handle guidance embedding
        elif key.startswith("guidance_in.") and "distilled" in model_type:
            new_key = key.replace("guidance_in.mlp.", "guidance_embedder.linear_")
            if "mlp.0." in key:
                new_key = new_key.replace("guidance_embedder.linear_0.", "guidance_embedder.linear_1.")
            elif "mlp.2." in key:
                new_key = new_key.replace("guidance_embedder.linear_2.", "guidance_embedder.linear_2.")
        
        # The rest of the keys should mostly match
        # (double_blocks, single_blocks, final_layer, etc.)
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def convert_vae_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert VAE weights from official format to diffusers format."""
    # VAE weights should mostly match, but handle any 5D weights
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if "weight" in key and len(value.shape) == 5 and value.shape[2] == 1:
            # Squeeze 5D weights to 4D
            new_state_dict[key] = value.squeeze(2)
        else:
            new_state_dict[key] = value
    
    return new_state_dict


def create_transformer_config(model_type: str) -> Dict:
    """Create transformer configuration based on model type."""
    base_config = {
        "patch_size": [1, 1],
        "in_channels": 64,
        "out_channels": 64,
        "hidden_size": 3584,
        "heads_num": 28,
        "mlp_width_ratio": 4.0,
        "mlp_act_type": "gelu_tanh",
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [64, 64],
        "qkv_bias": True,
        "qk_norm": True,
        "qk_norm_type": "rms",
        "text_states_dim": 3584,
        "rope_theta": 256,
    }
    
    if "distilled" in model_type:
        base_config["guidance_embed"] = True
        base_config["use_meanflow"] = True
    else:
        base_config["guidance_embed"] = False
        base_config["use_meanflow"] = False
    
    return base_config


def create_vae_config() -> Dict:
    """Create VAE configuration."""
    return {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 64,
        "block_out_channels": (512, 1024, 2048, 4096),
        "layers_per_block": 2,
        "ffactor_spatial": 32,
        "sample_size": 512,
        "sample_tsize": 1,
        "scaling_factor": 1.0,
        "shift_factor": None,
        "downsample_match_channel": True,
        "upsample_match_channel": True,
    }


def main(args):
    """Main conversion function."""
    print("=" * 80)
    print("HunyuanImage to Diffusers Conversion Script")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Step 1: Convert transformer
    print("\n[1/4] Converting transformer model...")
    if args.transformer_checkpoint_path:
        transformer_state_dict = load_checkpoint(args.transformer_checkpoint_path)
        transformer_state_dict = convert_transformer_state_dict(transformer_state_dict, args.model_type)
        
        transformer_config = create_transformer_config(args.model_type)
        transformer = HunyuanImage2DModel(**transformer_config)
        
        # Load weights with strict=False to allow for missing/extra keys
        missing_keys, unexpected_keys = transformer.load_state_dict(transformer_state_dict, strict=False)
        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        
        print(f"  ✓ Transformer converted successfully")
    else:
        print("  ⚠ No transformer checkpoint provided, using random initialization")
        transformer_config = create_transformer_config(args.model_type)
        transformer = HunyuanImage2DModel(**transformer_config)
    
    # Step 2: Convert VAE
    print("\n[2/4] Converting VAE model...")
    if args.vae_checkpoint_path:
        vae_state_dict = load_checkpoint(args.vae_checkpoint_path)
        vae_state_dict = convert_vae_state_dict(vae_state_dict)
        
        vae_config = create_vae_config()
        vae = AutoencoderKLHunyuanImage(**vae_config)
        
        missing_keys, unexpected_keys = vae.load_state_dict(vae_state_dict, strict=False)
        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")
        
        print(f"  ✓ VAE converted successfully")
    else:
        print("  ⚠ No VAE checkpoint provided, using random initialization")
        vae_config = create_vae_config()
        vae = AutoencoderKLHunyuanImage(**vae_config)
    
    # Step 3: Load text encoder
    print("\n[3/4] Loading text encoder...")
    text_encoder_path = args.text_encoder_path or "google/t5-v1_1-xxl"
    print(f"  Using text encoder: {text_encoder_path}")
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_path)
    print(f"  ✓ Text encoder loaded successfully")
    
    # Step 4: Create scheduler
    print("\n[4/4] Creating scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5 if "distilled" not in args.model_type else 4)
    print(f"  ✓ Scheduler created successfully")
    
    # Create pipeline
    print("\n[5/5] Assembling pipeline...")
    pipeline = HunyuanImagePipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
    print(f"  ✓ Pipeline assembled successfully")
    
    # Save pipeline
    print(f"\nSaving pipeline to: {args.output_path}")
    pipeline.save_pretrained(
        args.output_path,
        safe_serialization=True,
        max_shard_size="5GB",
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None,
    )
    
    print("\n" + "=" * 80)
    print("✅ Conversion completed successfully!")
    print("=" * 80)
    print(f"\nYou can now load the model with:")
    print(f'  pipe = HunyuanImagePipeline.from_pretrained("{args.output_path}")')
    print(f'  image = pipe("A cute penguin", height=2048, width=2048).images[0]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HunyuanImage checkpoints to Diffusers format")
    
    parser.add_argument(
        "--transformer_checkpoint_path",
        type=str,
        default=None,
        help="Path to the transformer checkpoint (.pt or .safetensors)",
    )
    parser.add_argument(
        "--vae_checkpoint_path",
        type=str,
        default=None,
        help="Path to the VAE checkpoint (.pt or .safetensors)",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path to the text encoder (default: google/t5-v1_1-xxl)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted pipeline",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="hunyuanimage-v2.1",
        choices=["hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"],
        help="Type of model to convert",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Repository ID for pushing to Hub (if --push_to_hub is set)",
    )
    
    args = parser.parse_args()
    main(args)
