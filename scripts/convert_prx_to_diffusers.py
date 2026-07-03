#!/usr/bin/env python3
"""
Script to convert a PRX checkpoint from the original codebase to diffusers format.

Supports two checkpoint layouts:
  * a single-file ``torch.save`` checkpoint (``.pt`` / ``.pth``), and
  * a sharded torch Distributed Checkpoint (DCP) directory (``.metadata`` + ``*.distcp``),
    as produced by Composer/FSDP training.

and three model variants (``--variant``):
  * ``flux``  : latent-space, AutoencoderKL  (16ch, patch 2)  -> PRXPipeline
  * ``dc-ae`` : latent-space, AutoencoderDC  (32ch, patch 1)  -> PRXPipeline
  * ``pixel`` : pixel-space PRXPixel (3ch RGB, patch 16, bottleneck img_in, resolution embedder,
                Qwen3-VL text tower, no VAE)                  -> PRXPixelPipeline

The block-level parameter remapping is shared across all variants; the ``pixel`` variant's extra
parameters (``img_in.{0,1}`` bottleneck and ``resolution_embedder.mlp.*``) carry over with no
renaming, so a single mapping generalises across versions.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import save_file

from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel


DEFAULT_RESOLUTION = 512

# Default location of the denoiser weights inside a research (Composer) checkpoint.
DENOISER_PREFIX = "state.model.denoiser."

# Qwen3-VL embedding tower used by the pixel variant.
PIXEL_TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-Embedding-2B"
PIXEL_PROMPT_MAX_TOKENS = 256


@dataclass(frozen=True)
class PRXBase:
    context_in_dim: int = 2304
    hidden_size: int = 1792
    mlp_ratio: float = 3.5
    num_heads: int = 28
    depth: int = 16
    axes_dim: Tuple[int, int] = (32, 32)
    theta: int = 10_000
    time_factor: float = 1000.0
    time_max_period: int = 10_000
    bottleneck_size: Optional[int] = None
    resolution_embeds: bool = False


@dataclass(frozen=True)
class PRXFlux(PRXBase):
    in_channels: int = 16
    patch_size: int = 2


@dataclass(frozen=True)
class PRXDCAE(PRXBase):
    in_channels: int = 32
    patch_size: int = 1


@dataclass(frozen=True)
class PRXPixel(PRXBase):
    # Pixel-space RGB diffusion (PRXPixel / 7B).
    in_channels: int = 3
    patch_size: int = 16
    context_in_dim: int = 2048  # Qwen3-VL-Embedding-2B hidden size
    hidden_size: int = 3584
    num_heads: int = 28
    depth: int = 24
    axes_dim: Tuple[int, int] = (64, 64)
    bottleneck_size: int = 768
    resolution_embeds: bool = True


VARIANTS = {"flux": PRXFlux, "dc-ae": PRXDCAE, "pixel": PRXPixel}


def build_config(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}. Choose from {list(VARIANTS)}")
    config_dict = asdict(VARIANTS[variant]())
    config_dict["axes_dim"] = list(config_dict["axes_dim"])
    if config_dict["bottleneck_size"] is None:
        # Keep config.json clean for variants that don't use the bottleneck.
        config_dict.pop("bottleneck_size")
    return config_dict


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _flatten(nested: dict, parent: str = "") -> Dict[str, torch.Tensor]:
    """Flatten the nested dict returned by DCP loading back into dotted keys."""
    flat = {}
    for k, v in nested.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _is_dcp_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


def load_denoiser_state_dict(checkpoint_path: str, prefix: str = DENOISER_PREFIX) -> Dict[str, torch.Tensor]:
    """Load just the denoiser weights from a research checkpoint (DCP dir or single file)."""
    if _is_dcp_dir(checkpoint_path):
        print(f"Loading DCP (distributed) checkpoint from: {checkpoint_path}")
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

        reader = FileSystemReader(checkpoint_path)
        meta = reader.read_metadata()
        keys = {k for k in meta.state_dict_metadata if k.startswith(prefix)}
        if not keys:
            raise ValueError(f"No keys with prefix '{prefix}' found in {checkpoint_path}")
        print(f"  Reading {len(keys)} denoiser tensors (skipping optimizer / EMA / RNG state)...")
        nested = _load_state_dict_from_keys(keys, storage_reader=reader)
        flat = _flatten(nested)
        state_dict = {k[len(prefix) :]: v for k, v in flat.items() if k.startswith(prefix)}
    else:
        print(f"Loading single-file checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            state_dict = ckpt
        # Strip a denoiser prefix if the keys carry one.
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    print(f"✓ Loaded {len(state_dict)} denoiser parameters")
    return state_dict


# ---------------------------------------------------------------------------
# Parameter name remapping (research -> diffusers)
# ---------------------------------------------------------------------------
def create_parameter_mapping(depth: int) -> dict:
    """Map old parameter names (layers on PRXBlock) to diffusers names (layers on PRXAttention)."""
    mapping = {}
    for i in range(depth):
        mapping[f"blocks.{i}.img_qkv_proj.weight"] = f"blocks.{i}.attention.img_qkv_proj.weight"
        mapping[f"blocks.{i}.txt_kv_proj.weight"] = f"blocks.{i}.attention.txt_kv_proj.weight"
        mapping[f"blocks.{i}.qk_norm.query_norm.scale"] = f"blocks.{i}.attention.norm_q.weight"
        mapping[f"blocks.{i}.qk_norm.key_norm.scale"] = f"blocks.{i}.attention.norm_k.weight"
        mapping[f"blocks.{i}.qk_norm.query_norm.weight"] = f"blocks.{i}.attention.norm_q.weight"
        mapping[f"blocks.{i}.qk_norm.key_norm.weight"] = f"blocks.{i}.attention.norm_k.weight"
        mapping[f"blocks.{i}.k_norm.scale"] = f"blocks.{i}.attention.norm_added_k.weight"
        mapping[f"blocks.{i}.k_norm.weight"] = f"blocks.{i}.attention.norm_added_k.weight"
        mapping[f"blocks.{i}.attn_out.weight"] = f"blocks.{i}.attention.to_out.0.weight"
    return mapping


def convert_checkpoint_parameters(old_state_dict: Dict[str, torch.Tensor], depth: int) -> dict[str, torch.Tensor]:
    """Apply the block remapping. Unmapped keys (img_in, time_in, txt_in, resolution_embedder, final_layer)
    carry over unchanged."""
    mapping = create_parameter_mapping(depth)
    converted = {}
    num_mapped = 0
    for key, value in old_state_dict.items():
        new_key = mapping.get(key, key)
        if new_key != key:
            num_mapped += 1
        converted[new_key] = value
    print(f"✓ Converted {len(converted)} parameters ({num_mapped} block keys remapped)")
    return converted


def create_transformer_from_checkpoint(checkpoint_path: str, config: dict) -> PRXTransformer2DModel:
    """Create and load a PRXTransformer2DModel from a research checkpoint."""
    state_dict = load_denoiser_state_dict(checkpoint_path)
    converted = convert_checkpoint_parameters(state_dict, depth=int(config["depth"]))

    print("Creating PRXTransformer2DModel...")
    transformer = PRXTransformer2DModel(**config)

    # Match the checkpoint dtype (research saves bf16).
    param_dtype = next(iter(converted.values())).dtype
    transformer = transformer.to(param_dtype)

    missing, unexpected = transformer.load_state_dict(converted, strict=False)
    if missing:
        print(f"⚠ Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"⚠ Unexpected keys ({len(unexpected)}): {unexpected}")
    if not missing and not unexpected:
        print("✓ All parameters loaded successfully (0 missing, 0 unexpected)!")
    else:
        raise RuntimeError("Checkpoint did not load cleanly; see missing/unexpected keys above.")
    return transformer


# ---------------------------------------------------------------------------
# Auxiliary components
# ---------------------------------------------------------------------------
def create_scheduler_config(output_path: str, shift: float):
    scheduler_config = {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": shift}
    scheduler_path = os.path.join(output_path, "scheduler")
    os.makedirs(scheduler_path, exist_ok=True)
    with open(os.path.join(scheduler_path, "scheduler_config.json"), "w") as f:
        json.dump(scheduler_config, f, indent=2)
    print("✓ Created scheduler config")


def download_and_save_vae(variant: str, output_path: str):
    from diffusers import AutoencoderDC, AutoencoderKL

    vae_path = os.path.join(output_path, "vae")
    os.makedirs(vae_path, exist_ok=True)
    if variant == "flux":
        print("Downloading FLUX VAE from black-forest-labs/FLUX.1-dev...")
        vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
    else:  # dc-ae
        print("Downloading DC-AE VAE from mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers...")
        vae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
    vae.save_pretrained(vae_path)
    print(f"✓ Saved VAE to {vae_path}")


def download_and_save_t5gemma_text_encoder(output_path: str):
    from transformers import GemmaTokenizerFast
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaModel

    text_encoder_path = os.path.join(output_path, "text_encoder")
    tokenizer_path = os.path.join(output_path, "tokenizer")
    os.makedirs(text_encoder_path, exist_ok=True)
    os.makedirs(tokenizer_path, exist_ok=True)

    print("Downloading T5Gemma model from google/t5gemma-2b-2b-ul2...")
    t5gemma_encoder = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2").encoder
    t5gemma_encoder.save_pretrained(text_encoder_path)
    print(f"✓ Saved T5GemmaEncoder to {text_encoder_path}")

    tokenizer = GemmaTokenizerFast.from_pretrained("google/t5gemma-2b-2b-ul2")
    tokenizer.model_max_length = 256
    tokenizer.save_pretrained(tokenizer_path)
    print(f"✓ Saved tokenizer to {tokenizer_path}")
    return "T5GemmaEncoder", "prx"


def download_and_save_qwen_text_encoder(output_path: str, repo: str = PIXEL_TEXT_ENCODER_REPO):
    """Download the Qwen3-VL embedding tower, keep only the text backbone, and save it."""
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    text_encoder_path = os.path.join(output_path, "text_encoder")
    tokenizer_path = os.path.join(output_path, "tokenizer")
    os.makedirs(text_encoder_path, exist_ok=True)
    os.makedirs(tokenizer_path, exist_ok=True)

    print(f"Downloading Qwen3-VL model from {repo} (vision tower will be discarded)...")
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    text_encoder = full_model.model.language_model
    text_encoder.save_pretrained(text_encoder_path)
    encoder_class = type(text_encoder).__name__
    del full_model
    print(f"✓ Saved {encoder_class} to {text_encoder_path}")

    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer.model_max_length = PIXEL_PROMPT_MAX_TOKENS
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer_class = type(tokenizer).__name__
    print(f"✓ Saved tokenizer ({tokenizer_class}) to {tokenizer_path}")
    return encoder_class, "transformers", tokenizer_class


def create_model_index(
    variant: str,
    default_image_size: int,
    output_path: str,
    text_encoder_class: str,
    text_encoder_lib: str,
    tokenizer_class: str,
):
    if variant == "pixel":
        model_index = {
            "_class_name": "PRXPixelPipeline",
            "_diffusers_version": "0.37.0.dev0",
            "_name_or_path": os.path.basename(output_path),
            "default_sample_size": default_image_size,
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": [text_encoder_lib, text_encoder_class],
            "tokenizer": ["transformers", tokenizer_class],
            "transformer": ["diffusers", "PRXTransformer2DModel"],
            "vae": [None, None],  # pixel-space: no VAE
        }
    else:
        vae_class = "AutoencoderKL" if variant == "flux" else "AutoencoderDC"
        model_index = {
            "_class_name": "PRXPipeline",
            "_diffusers_version": "0.37.0.dev0",
            "_name_or_path": os.path.basename(output_path),
            "default_sample_size": default_image_size,
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": [text_encoder_lib, text_encoder_class],
            "tokenizer": ["transformers", tokenizer_class],
            "transformer": ["diffusers", "PRXTransformer2DModel"],
            "vae": ["diffusers", vae_class],
        }
    with open(os.path.join(output_path, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"✓ Wrote model_index.json ({model_index['_class_name']})")


def main(args):
    config = build_config(args.variant)
    os.makedirs(args.output_path, exist_ok=True)
    print(f"✓ Output directory: {args.output_path}")
    print(f"✓ Variant: {args.variant} | config: {config}")

    # ---- transformer ----
    transformer = create_transformer_from_checkpoint(args.checkpoint_path, config)
    transformer_path = os.path.join(args.output_path, "transformer")
    os.makedirs(transformer_path, exist_ok=True)
    with open(os.path.join(transformer_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    save_file(transformer.state_dict(), os.path.join(transformer_path, "diffusion_pytorch_model.safetensors"))
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"✓ Saved transformer to {transformer_path} ({num_params:,} params)")

    # ---- scheduler ----
    create_scheduler_config(args.output_path, args.shift)

    # ---- vae (none for pixel) ----
    if args.variant != "pixel" and not args.skip_vae:
        download_and_save_vae(args.variant, args.output_path)

    # ---- text encoder + tokenizer ----
    text_encoder_class, text_encoder_lib, tokenizer_class = "T5GemmaEncoder", "prx", "GemmaTokenizerFast"
    if not args.skip_text_encoder:
        if args.variant == "pixel":
            text_encoder_class, text_encoder_lib, tokenizer_class = download_and_save_qwen_text_encoder(
                args.output_path
            )
        else:
            text_encoder_class, text_encoder_lib = download_and_save_t5gemma_text_encoder(args.output_path)
            tokenizer_class = "GemmaTokenizerFast"

    create_model_index(
        args.variant, args.resolution, args.output_path, text_encoder_class, text_encoder_lib, tokenizer_class
    )

    # ---- verify ----
    if args.skip_text_encoder:
        print("Skipped text encoder; verifying the transformer reloads from disk...")
        reloaded = PRXTransformer2DModel.from_pretrained(transformer_path)
        print(
            f"✓ Transformer reloaded: {type(reloaded).__name__} ({sum(p.numel() for p in reloaded.parameters()):,} params)"
        )
    else:
        from diffusers import PRXPipeline, PRXPixelPipeline

        pipe_cls = PRXPixelPipeline if args.variant == "pixel" else PRXPipeline
        pipeline = pipe_cls.from_pretrained(args.output_path)
        print("Pipeline loaded successfully!")
        print(f"  Pipeline: {type(pipeline).__name__}")
        print(f"  Transformer: {type(pipeline.transformer).__name__}")
        print(f"  VAE: {type(pipeline.vae).__name__ if pipeline.vae is not None else None}")
        print(f"  Text Encoder: {type(pipeline.text_encoder).__name__}")
        print(f"  Scheduler: {type(pipeline.scheduler).__name__}")

    print("Conversion completed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PRX checkpoint to diffusers format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original PRX checkpoint (a .pt/.pth file or a DCP directory).",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output directory for the converted diffusers pipeline"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(VARIANTS),
        required=True,
        help="Model variant: 'flux' (AutoencoderKL, 16ch), 'dc-ae' (AutoencoderDC, 32ch), or 'pixel' (RGB PRXPixel).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Default sample size for the pipeline (e.g. 256, 512, 1024).",
    )
    parser.add_argument("--shift", type=float, default=3.0, help="Shift for the scheduler")
    parser.add_argument(
        "--skip_text_encoder",
        action="store_true",
        help="Skip downloading/saving the text encoder + tokenizer (validate the transformer only).",
    )
    parser.add_argument("--skip_vae", action="store_true", help="Skip downloading/saving the VAE.")

    args = parser.parse_args()
    try:
        if not main(args):
            sys.exit(1)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
