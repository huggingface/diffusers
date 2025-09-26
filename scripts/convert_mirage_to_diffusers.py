#!/usr/bin/env python3
"""
Script to convert Mirage checkpoint from original codebase to diffusers format.
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusers.models.transformers.transformer_mirage import MirageTransformer2DModel
from diffusers.pipelines.mirage import MiragePipeline


def load_reference_config(vae_type: str) -> dict:
    """Load transformer config from existing pipeline checkpoint."""

    if vae_type == "flux":
        config_path = "/raid/shared/storage/home/davidb/diffusers/diffusers_pipeline_checkpoints/pipeline_checkpoint_fluxvae_gemmaT5_updated/transformer/config.json"
    elif vae_type == "dc-ae":
        config_path = "/raid/shared/storage/home/davidb/diffusers/diffusers_pipeline_checkpoints/pipeline_checkpoint_dcae_gemmaT5_updated/transformer/config.json"
    else:
        raise ValueError(f"Unsupported VAE type: {vae_type}. Use 'flux' or 'dc-ae'")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Reference config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"✓ Loaded {vae_type} config: in_channels={config['in_channels']}")
    return config


def create_parameter_mapping() -> dict:
    """Create mapping from old parameter names to new diffusers names."""

    # Key mappings for structural changes
    mapping = {}

    # RMSNorm: scale -> weight
    for i in range(16):  # 16 layers
        mapping[f"blocks.{i}.qk_norm.query_norm.scale"] = f"blocks.{i}.qk_norm.query_norm.weight"
        mapping[f"blocks.{i}.qk_norm.key_norm.scale"] = f"blocks.{i}.qk_norm.key_norm.weight"
        mapping[f"blocks.{i}.k_norm.scale"] = f"blocks.{i}.k_norm.weight"

        # Attention: attn_out -> attention.to_out.0
        mapping[f"blocks.{i}.attn_out.weight"] = f"blocks.{i}.attention.to_out.0.weight"

    return mapping


def convert_checkpoint_parameters(old_state_dict: dict) -> dict:
    """Convert old checkpoint parameters to new diffusers format."""

    print("Converting checkpoint parameters...")

    mapping = create_parameter_mapping()
    converted_state_dict = {}

    # First, print available keys to understand structure
    print("Available keys in checkpoint:")
    for key in sorted(old_state_dict.keys())[:10]:  # Show first 10 keys
        print(f"  {key}")
    if len(old_state_dict) > 10:
        print(f"  ... and {len(old_state_dict) - 10} more")

    for key, value in old_state_dict.items():
        new_key = key

        # Apply specific mappings if needed
        if key in mapping:
            new_key = mapping[key]
            print(f"  Mapped: {key} -> {new_key}")

        # Handle img_qkv_proj -> split to to_q, to_k, to_v
        if "img_qkv_proj.weight" in key:
            print(f"  Found QKV projection: {key}")
            # Split QKV weight into separate Q, K, V projections
            qkv_weight = value
            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)

            # Extract layer number from key (e.g., blocks.0.img_qkv_proj.weight -> 0)
            parts = key.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = parts[i + 1]
                    break

            if layer_idx is not None:
                converted_state_dict[f"blocks.{layer_idx}.attention.to_q.weight"] = q_weight
                converted_state_dict[f"blocks.{layer_idx}.attention.to_k.weight"] = k_weight
                converted_state_dict[f"blocks.{layer_idx}.attention.to_v.weight"] = v_weight
                print(f"  Split QKV for layer {layer_idx}")

                # Also keep the original img_qkv_proj for backward compatibility
                converted_state_dict[new_key] = value
        else:
            converted_state_dict[new_key] = value

    print(f"✓ Converted {len(converted_state_dict)} parameters")
    return converted_state_dict


def create_transformer_from_checkpoint(checkpoint_path: str, config: dict) -> MirageTransformer2DModel:
    """Create and load MirageTransformer2DModel from old checkpoint."""

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load old checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    old_checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(old_checkpoint, dict):
        if "model" in old_checkpoint:
            state_dict = old_checkpoint["model"]
        elif "state_dict" in old_checkpoint:
            state_dict = old_checkpoint["state_dict"]
        else:
            state_dict = old_checkpoint
    else:
        state_dict = old_checkpoint

    print(f"✓ Loaded checkpoint with {len(state_dict)} parameters")

    # Convert parameter names if needed
    converted_state_dict = convert_checkpoint_parameters(state_dict)

    # Create transformer with config
    print("Creating MirageTransformer2DModel...")
    transformer = MirageTransformer2DModel(**config)

    # Load state dict
    print("Loading converted parameters...")
    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=False)

    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {unexpected_keys}")

    if not missing_keys and not unexpected_keys:
        print("✓ All parameters loaded successfully!")

    return transformer


def copy_pipeline_components(vae_type: str, output_path: str):
    """Copy VAE, scheduler, text encoder, and tokenizer from reference pipeline."""

    if vae_type == "flux":
        ref_pipeline = "/raid/shared/storage/home/davidb/diffusers/diffusers_pipeline_checkpoints/pipeline_checkpoint_fluxvae_gemmaT5_updated"
    else:  # dc-ae
        ref_pipeline = "/raid/shared/storage/home/davidb/diffusers/diffusers_pipeline_checkpoints/pipeline_checkpoint_dcae_gemmaT5_updated"

    components = ["vae", "scheduler", "text_encoder", "tokenizer"]

    for component in components:
        src_path = os.path.join(ref_pipeline, component)
        dst_path = os.path.join(output_path, component)

        if os.path.exists(src_path):
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
            print(f"✓ Copied {component}")
        else:
            print(f"⚠ Component not found: {src_path}")


def create_model_index(vae_type: str, output_path: str):
    """Create model_index.json for the pipeline."""

    if vae_type == "flux":
        vae_class = "AutoencoderKL"
    else:  # dc-ae
        vae_class = "AutoencoderDC"

    model_index = {
        "_class_name": "MiragePipeline",
        "_diffusers_version": "0.31.0.dev0",
        "_name_or_path": os.path.basename(output_path),
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5GemmaEncoder"],
        "tokenizer": ["transformers", "GemmaTokenizerFast"],
        "transformer": ["diffusers", "MirageTransformer2DModel"],
        "vae": ["diffusers", vae_class],
    }

    model_index_path = os.path.join(output_path, "model_index.json")
    with open(model_index_path, "w") as f:
        json.dump(model_index, f, indent=2)

    print("✓ Created model_index.json")


def main(args):
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    # Load reference config based on VAE type
    config = load_reference_config(args.vae_type)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    print(f"✓ Output directory: {args.output_path}")

    # Create transformer from checkpoint
    transformer = create_transformer_from_checkpoint(args.checkpoint_path, config)

    # Save transformer
    transformer_path = os.path.join(args.output_path, "transformer")
    os.makedirs(transformer_path, exist_ok=True)

    # Save config
    with open(os.path.join(transformer_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save model weights as safetensors
    state_dict = transformer.state_dict()
    save_file(state_dict, os.path.join(transformer_path, "diffusion_pytorch_model.safetensors"))
    print(f"✓ Saved transformer to {transformer_path}")

    # Copy other pipeline components
    copy_pipeline_components(args.vae_type, args.output_path)

    # Create model index
    create_model_index(args.vae_type, args.output_path)

    # Verify the pipeline can be loaded
    try:
        pipeline = MiragePipeline.from_pretrained(args.output_path)
        print("Pipeline loaded successfully!")
        print(f"Transformer: {type(pipeline.transformer).__name__}")
        print(f"VAE: {type(pipeline.vae).__name__}")
        print(f"Text Encoder: {type(pipeline.text_encoder).__name__}")
        print(f"Scheduler: {type(pipeline.scheduler).__name__}")

        # Display model info
        num_params = sum(p.numel() for p in pipeline.transformer.parameters())
        print(f"✓ Transformer parameters: {num_params:,}")

    except Exception as e:
        print(f"Pipeline verification failed: {e}")
        return False

    print("Conversion completed successfully!")
    print(f"Converted pipeline saved to: {args.output_path}")
    print(f"VAE type: {args.vae_type}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mirage checkpoint to diffusers format")

    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the original Mirage checkpoint (.pth file)"
    )

    parser.add_argument(
        "--output_path", type=str, required=True, help="Output directory for the converted diffusers pipeline"
    )

    parser.add_argument(
        "--vae_type",
        type=str,
        choices=["flux", "dc-ae"],
        required=True,
        help="VAE type to use: 'flux' for AutoencoderKL (16 channels) or 'dc-ae' for AutoencoderDC (32 channels)",
    )

    args = parser.parse_args()

    try:
        success = main(args)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
