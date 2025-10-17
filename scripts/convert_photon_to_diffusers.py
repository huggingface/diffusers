#!/usr/bin/env python3
"""
Script to convert Photon checkpoint from original codebase to diffusers format.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from safetensors.torch import save_file

from diffusers.models.transformers.transformer_photon import PhotonTransformer2DModel
from diffusers.pipelines.photon import PhotonPipeline


DEFAULT_RESOLUTION = 512


@dataclass(frozen=True)
class PhotonBase:
    context_in_dim: int = 2304
    hidden_size: int = 1792
    mlp_ratio: float = 3.5
    num_heads: int = 28
    depth: int = 16
    axes_dim: Tuple[int, int] = (32, 32)
    theta: int = 10_000
    time_factor: float = 1000.0
    time_max_period: int = 10_000


@dataclass(frozen=True)
class PhotonFlux(PhotonBase):
    in_channels: int = 16
    patch_size: int = 2


@dataclass(frozen=True)
class PhotonDCAE(PhotonBase):
    in_channels: int = 32
    patch_size: int = 1


def build_config(vae_type: str) -> Tuple[dict, int]:
    if vae_type == "flux":
        cfg = PhotonFlux()
    elif vae_type == "dc-ae":
        cfg = PhotonDCAE()
    else:
        raise ValueError(f"Unsupported VAE type: {vae_type}. Use 'flux' or 'dc-ae'")

    config_dict = asdict(cfg)
    config_dict["axes_dim"] = list(config_dict["axes_dim"])  # type: ignore[index]
    return config_dict


def create_parameter_mapping(depth: int) -> dict:
    """Create mapping from old parameter names to new diffusers names."""

    # Key mappings for structural changes
    mapping = {}

    # Map old structure (layers in PhotonBlock) to new structure (layers in PhotonAttention)
    for i in range(depth):
        # QKV projections moved to attention module
        mapping[f"blocks.{i}.img_qkv_proj.weight"] = f"blocks.{i}.attention.img_qkv_proj.weight"
        mapping[f"blocks.{i}.txt_kv_proj.weight"] = f"blocks.{i}.attention.txt_kv_proj.weight"

        # QK norm moved to attention module and renamed to match Attention's qk_norm structure
        mapping[f"blocks.{i}.qk_norm.query_norm.scale"] = f"blocks.{i}.attention.norm_q.weight"
        mapping[f"blocks.{i}.qk_norm.key_norm.scale"] = f"blocks.{i}.attention.norm_k.weight"
        mapping[f"blocks.{i}.qk_norm.query_norm.weight"] = f"blocks.{i}.attention.norm_q.weight"
        mapping[f"blocks.{i}.qk_norm.key_norm.weight"] = f"blocks.{i}.attention.norm_k.weight"

        # K norm for text tokens moved to attention module
        mapping[f"blocks.{i}.k_norm.scale"] = f"blocks.{i}.attention.norm_added_k.weight"
        mapping[f"blocks.{i}.k_norm.weight"] = f"blocks.{i}.attention.norm_added_k.weight"

        # Attention output projection
        mapping[f"blocks.{i}.attn_out.weight"] = f"blocks.{i}.attention.to_out.0.weight"

    return mapping


def convert_checkpoint_parameters(old_state_dict: Dict[str, torch.Tensor], depth: int) -> Dict[str, torch.Tensor]:
    """Convert old checkpoint parameters to new diffusers format."""

    print("Converting checkpoint parameters...")

    mapping = create_parameter_mapping(depth)
    converted_state_dict = {}

    for key, value in old_state_dict.items():
        new_key = key

        # Apply specific mappings if needed
        if key in mapping:
            new_key = mapping[key]
            print(f"  Mapped: {key} -> {new_key}")

        converted_state_dict[new_key] = value

    print(f"✓ Converted {len(converted_state_dict)} parameters")
    return converted_state_dict


def create_transformer_from_checkpoint(checkpoint_path: str, config: dict) -> PhotonTransformer2DModel:
    """Create and load PhotonTransformer2DModel from old checkpoint."""

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
    model_depth = int(config.get("depth", 16))
    converted_state_dict = convert_checkpoint_parameters(state_dict, depth=model_depth)

    # Create transformer with config
    print("Creating PhotonTransformer2DModel...")
    transformer = PhotonTransformer2DModel(**config)

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


def create_scheduler_config(output_path: str, shift: float):
    """Create FlowMatchEulerDiscreteScheduler config."""

    scheduler_config = {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": shift}

    scheduler_path = os.path.join(output_path, "scheduler")
    os.makedirs(scheduler_path, exist_ok=True)

    with open(os.path.join(scheduler_path, "scheduler_config.json"), "w") as f:
        json.dump(scheduler_config, f, indent=2)

    print("✓ Created scheduler config")


def download_and_save_vae(vae_type: str, output_path: str):
    """Download and save VAE to local directory."""
    from diffusers import AutoencoderDC, AutoencoderKL

    vae_path = os.path.join(output_path, "vae")
    os.makedirs(vae_path, exist_ok=True)

    if vae_type == "flux":
        print("Downloading FLUX VAE from black-forest-labs/FLUX.1-dev...")
        vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
    else:  # dc-ae
        print("Downloading DC-AE VAE from mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers...")
        vae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")

    vae.save_pretrained(vae_path)
    print(f"✓ Saved VAE to {vae_path}")


def download_and_save_text_encoder(output_path: str):
    """Download and save T5Gemma text encoder and tokenizer."""
    from transformers import GemmaTokenizerFast
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaModel

    text_encoder_path = os.path.join(output_path, "text_encoder")
    tokenizer_path = os.path.join(output_path, "tokenizer")
    os.makedirs(text_encoder_path, exist_ok=True)
    os.makedirs(tokenizer_path, exist_ok=True)

    print("Downloading T5Gemma model from google/t5gemma-2b-2b-ul2...")
    t5gemma_model = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2")

    # Extract and save only the encoder
    t5gemma_encoder = t5gemma_model.encoder
    t5gemma_encoder.save_pretrained(text_encoder_path)
    print(f"✓ Saved T5GemmaEncoder to {text_encoder_path}")

    print("Downloading tokenizer from google/t5gemma-2b-2b-ul2...")
    tokenizer = GemmaTokenizerFast.from_pretrained("google/t5gemma-2b-2b-ul2")
    tokenizer.model_max_length = 256
    tokenizer.save_pretrained(tokenizer_path)
    print(f"✓ Saved tokenizer to {tokenizer_path}")


def create_model_index(vae_type: str, default_image_size: int, output_path: str):
    """Create model_index.json for the pipeline."""

    if vae_type == "flux":
        vae_class = "AutoencoderKL"
    else:  # dc-ae
        vae_class = "AutoencoderDC"

    model_index = {
        "_class_name": "PhotonPipeline",
        "_diffusers_version": "0.31.0.dev0",
        "_name_or_path": os.path.basename(output_path),
        "default_sample_size": default_image_size,
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["photon", "T5GemmaEncoder"],
        "tokenizer": ["transformers", "GemmaTokenizerFast"],
        "transformer": ["diffusers", "PhotonTransformer2DModel"],
        "vae": ["diffusers", vae_class],
    }

    model_index_path = os.path.join(output_path, "model_index.json")
    with open(model_index_path, "w") as f:
        json.dump(model_index, f, indent=2)


def main(args):
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    config = build_config(args.vae_type)

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

    # Create scheduler config
    create_scheduler_config(args.output_path, args.shift)

    download_and_save_vae(args.vae_type, args.output_path)
    download_and_save_text_encoder(args.output_path)

    # Create model_index.json
    create_model_index(args.vae_type, args.resolution, args.output_path)

    # Verify the pipeline can be loaded
    try:
        pipeline = PhotonPipeline.from_pretrained(args.output_path)
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
    parser = argparse.ArgumentParser(description="Convert Photon checkpoint to diffusers format")

    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the original Photon checkpoint (.pth file )"
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

    parser.add_argument(
        "--resolution",
        type=int,
        choices=[256, 512, 1024],
        default=DEFAULT_RESOLUTION,
        help="Target resolution for the model (256, 512, or 1024). Affects the transformer's sample_size.",
    )

    parser.add_argument(
        "--shift",
        type=float,
        default=3.0,
        help="Shift for the scheduler",
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
