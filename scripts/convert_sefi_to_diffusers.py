#!/usr/bin/env python
# Copyright 2026 SeFi-Image Authors and The HuggingFace Team. All rights reserved.
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

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from diffusers import __version__
from diffusers.models import SeFiTransformer2DModel


SEFI_SCALE_PRESETS = {
    "0p5b": {
        "attention_head_dim": 128,
        "num_attention_heads": 12,
        "num_layers": 3,
        "num_single_layers": 10,
        "joint_attention_dim": 6144,
    },
    "1b": {
        "attention_head_dim": 128,
        "num_attention_heads": 16,
        "num_layers": 4,
        "num_single_layers": 12,
        "joint_attention_dim": 6144,
    },
    "2b": {
        "attention_head_dim": 128,
        "num_attention_heads": 20,
        "num_layers": 4,
        "num_single_layers": 16,
        "joint_attention_dim": 6144,
    },
    "3b": {
        "attention_head_dim": 128,
        "num_attention_heads": 22,
        "num_layers": 5,
        "num_single_layers": 18,
        "joint_attention_dim": 7680,
    },
    "4b": {
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "num_layers": 5,
        "num_single_layers": 20,
        "joint_attention_dim": 7680,
    },
    "5b": {
        "attention_head_dim": 128,
        "num_attention_heads": 26,
        "num_layers": 6,
        "num_single_layers": 21,
        "joint_attention_dim": 7680,
    },
    "6b": {
        "attention_head_dim": 128,
        "num_attention_heads": 28,
        "num_layers": 6,
        "num_single_layers": 22,
        "joint_attention_dim": 7680,
    },
    "8b": {
        "attention_head_dim": 128,
        "num_attention_heads": 30,
        "num_layers": 7,
        "num_single_layers": 24,
        "joint_attention_dim": 7680,
    },
    "9b": {
        "attention_head_dim": 128,
        "num_attention_heads": 32,
        "num_layers": 8,
        "num_single_layers": 24,
        "joint_attention_dim": 12288,
    },
}

QWEN3VL_TEXT_HIDDEN_DIMS = {
    "qwen3vl_2b": 2048,
    "qwen3vl_4b": 2560,
    "qwen3vl_8b": 4096,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a SeFi-Image checkpoint to Diffusers format.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint folder or Hugging Face repo id.")
    parser.add_argument("--output", required=True, help="Output Diffusers checkpoint folder.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated checkpoints.")
    parser.add_argument(
        "--variant",
        choices=["base", "rl", "turbo"],
        default=None,
        help="Model family. Inferred from checkpoint name if omitted.",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint: str, cache_dir: str | None, token: str | None) -> Path:
    path = Path(checkpoint).expanduser()
    if path.exists():
        return path
    return Path(snapshot_download(checkpoint, cache_dir=cache_dir, token=token))


def copytree(src: Path, dst: Path, ignore=None):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def infer_variant(checkpoint: str, config: dict, explicit_variant: str | None) -> str:
    if explicit_variant is not None:
        return explicit_variant
    configured = str(config.get("inference", {}).get("family", "") or config.get("model", {}).get("variant", ""))
    text = f"{checkpoint} {configured}".lower()
    if "turbo" in text or "distill" in text:
        return "turbo"
    if "rl" in text:
        return "rl"
    return "base"


def default_steps(variant: str) -> int:
    return 4 if variant == "turbo" else 50


def default_guidance_scale(variant: str) -> float:
    return 1.0 if variant == "turbo" else 4.0


def texture_vae_config_path(root: Path, texture_vae_name: str) -> Path:
    if texture_vae_name == "sd1.5":
        return root / "vae" / "config.json"
    if texture_vae_name in {"flux1", "flux2"}:
        return root / "vae" / "config.json"
    raise ValueError(f"Unsupported texture VAE: {texture_vae_name}")


def build_transformer_config(root: Path, sefi_config: dict) -> dict:
    model_config = sefi_config["model"]
    transformer_config = load_json(root / "transformer" / "config.json")
    transformer_config.pop("_class_name", None)
    transformer_config.pop("_diffusers_version", None)
    transformer_config.pop("_name_or_path", None)
    transformer_config.pop("guidance_embeds", None)

    scale = str(model_config.get("transformer_scale", "")).lower()
    if scale and scale != "custom":
        transformer_config.update(SEFI_SCALE_PRESETS[scale])
    elif scale == "custom":
        transformer_config.update(model_config.get("transformer_overrides", {}))

    semantic_channels = int(model_config["semantic_channels"])
    texture_vae_name = str(model_config["texture_vae"]["name"]).lower()
    vae_config = load_json(texture_vae_config_path(root, texture_vae_name))
    texture_channels = int(vae_config["latent_channels"]) * 4
    total_channels = semantic_channels + texture_channels

    text_config = model_config["text_encoder"]
    hidden_layers = tuple(int(layer) for layer in text_config["hidden_layers"])
    text_dim = int(QWEN3VL_TEXT_HIDDEN_DIMS[text_config["model_name"]]) * len(hidden_layers)

    transformer_config["in_channels"] = total_channels
    transformer_config["out_channels"] = total_channels
    transformer_config["text_input_dim"] = text_dim
    if int(transformer_config["joint_attention_dim"]) != text_dim:
        raise ValueError(
            "Text dimension mismatch: "
            f"transformer joint_attention_dim={transformer_config['joint_attention_dim']} vs text_dim={text_dim}."
        )
    return transformer_config


def load_transformer_state_dict(transformer_dir: Path) -> dict[str, torch.Tensor]:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    single_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    bin_path = transformer_dir / "diffusion_pytorch_model.bin"

    if index_path.exists():
        index = load_json(index_path)
        state_dict = {}
        for shard in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(transformer_dir / shard))
        return state_dict
    if single_path.exists():
        return load_file(single_path)
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No supported transformer weights found under {transformer_dir}.")


def copy_tokenizer_files(src: Path, dst: Path):
    weight_patterns = {
        "model*.safetensors",
        "pytorch_model*.bin",
        "*.index.json",
    }

    def ignore(_dir, names):
        ignored = set()
        for name in names:
            for pattern in weight_patterns:
                if Path(name).match(pattern):
                    ignored.add(name)
        return ignored

    copytree(src, dst, ignore=ignore)


def main():
    args = parse_args()
    root = resolve_checkpoint(args.checkpoint, args.cache_dir, args.token)
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    sefi_config = load_yaml(root / "sefi_config.yaml")
    variant = infer_variant(args.checkpoint, sefi_config, args.variant)
    transformer_config = build_transformer_config(root, sefi_config)

    transformer = SeFiTransformer2DModel(**transformer_config)
    state_dict = load_transformer_state_dict(root / "transformer")
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Transformer state dict mismatch. Missing={missing[:20]}, unexpected={unexpected[:20]}")
    transformer.save_pretrained(output / "transformer", safe_serialization=True)

    copytree(root / "scheduler", output / "scheduler")
    copytree(root / "vae", output / "vae")

    text_encoder_name = sefi_config["model"]["text_encoder"]["model_name"]
    qwen_dir_name = {
        "qwen3vl_2b": "Qwen3-VL-2B-Instruct",
        "qwen3vl_4b": "Qwen3-VL-4B-Instruct",
        "qwen3vl_8b": "Qwen3-VL-8B-Instruct",
    }[text_encoder_name]
    qwen_dir = root / qwen_dir_name
    copytree(qwen_dir, output / "text_encoder")
    copy_tokenizer_files(qwen_dir, output / "tokenizer")

    model_config = sefi_config["model"]
    inference_config = sefi_config.get("inference", {})
    training_sefi_config = sefi_config.get("training", {}).get("sefi", {})
    texture_vae_name = str(model_config["texture_vae"]["name"]).lower()
    vae_class = "AutoencoderKLFlux2" if texture_vae_name == "flux2" else "AutoencoderKL"
    model_index = {
        "_class_name": "SeFiPipeline",
        "_diffusers_version": __version__,
        "transformer": ["diffusers", "SeFiTransformer2DModel"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "vae": ["diffusers", vae_class],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "semantic_channels": int(model_config["semantic_channels"]),
        "texture_vae_name": texture_vae_name,
        "is_turbo": variant == "turbo",
        "default_guidance_scale": float(inference_config.get("guidance_scale", default_guidance_scale(variant))),
        "default_num_inference_steps": int(inference_config.get("steps", default_steps(variant))),
        "delta_t": float(inference_config.get("delta_t", training_sefi_config.get("delta_t_max", 0.1))),
        "timestep_shift_alpha": float(
            inference_config.get("timestep_shift_alpha", 1.0 if variant == "turbo" else 0.3)
        ),
        "text_encoder_hidden_layers": [int(layer) for layer in model_config["text_encoder"]["hidden_layers"]],
        "max_sequence_length": int(model_config["text_encoder"].get("max_length", 1024)),
    }
    save_json(output / "model_index.json", model_index)
    shutil.copy2(root / "sefi_config.yaml", output / "sefi_config.yaml")
    print(f"Saved SeFi-Image Diffusers checkpoint to {output}")


if __name__ == "__main__":
    main()
