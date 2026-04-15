#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage:
#   python scripts/convert_longcat_audio_dit_to_diffusers.py --checkpoint_path /path/to/model --output_path /data/models
#   python scripts/convert_longcat_audio_dit_to_diffusers.py --repo_id meituan-longcat/LongCat-AudioDiT-1B --output_path /data/models
#   python scripts/convert_longcat_audio_dit_to_diffusers.py --checkpoint_path /path/to/model --output_path /data/models --dtype fp16

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    LongCatAudioDiTPipeline,
    LongCatAudioDiTTransformer,
    LongCatAudioDiTVae,
)


def find_checkpoint(input_dir: Path):
    safetensors_file = input_dir / "model.safetensors"
    if safetensors_file.exists():
        return input_dir, safetensors_file

    index_file = input_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        first_weight = list(weight_map.values())[0]
        return input_dir, input_dir / first_weight

    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            safetensors_file = subdir / "model.safetensors"
            if safetensors_file.exists():
                return subdir, safetensors_file
            index_file = subdir / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                first_weight = list(weight_map.values())[0]
                return subdir, subdir / first_weight

    raise FileNotFoundError(f"No checkpoint found in {input_dir}")


def convert_longcat_audio_dit(
    checkpoint_path: str | None = None,
    repo_id: str | None = None,
    output_path: str = "",
    dtype: str = "fp32",
    text_encoder_model: str = "google/umt5-xxl",
):
    if not checkpoint_path and not repo_id:
        raise ValueError("Either --checkpoint_path or --repo_id must be provided")
    if checkpoint_path and repo_id:
        raise ValueError("Cannot specify both --checkpoint_path and --repo_id")

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    if repo_id:
        input_dir = Path(snapshot_download(repo_id, local_files_only=False))
        model_name = repo_id.split("/")[-1]
    else:
        input_dir = Path(checkpoint_path)
        if not input_dir.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
        model_name = None

    model_dir, checkpoint_path = find_checkpoint(input_dir)
    if model_name is None:
        model_name = model_dir.name

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    state_dict = load_file(checkpoint_path)

    transformer_keys = [k for k in state_dict.keys() if k.startswith("transformer.")]
    transformer_state_dict = {key[12:]: state_dict[key] for key in transformer_keys}

    vae_keys = [k for k in state_dict.keys() if k.startswith("vae.")]
    vae_state_dict = {key[4:]: state_dict[key] for key in vae_keys}

    text_encoder_keys = [k for k in state_dict.keys() if k.startswith("text_encoder.")]
    text_encoder_state_dict = {key[13:]: state_dict[key] for key in text_encoder_keys}

    transformer = LongCatAudioDiTTransformer(
        dit_dim=config["dit_dim"],
        dit_depth=config["dit_depth"],
        dit_heads=config["dit_heads"],
        dit_text_dim=config["dit_text_dim"],
        latent_dim=config["latent_dim"],
        dropout=config.get("dit_dropout", 0.0),
        bias=config.get("dit_bias", True),
        cross_attn=config.get("dit_cross_attn", True),
        adaln_type=config.get("dit_adaln_type", "global"),
        adaln_use_text_cond=config.get("dit_adaln_use_text_cond", True),
        long_skip=config.get("dit_long_skip", True),
        text_conv=config.get("dit_text_conv", True),
        qk_norm=config.get("dit_qk_norm", True),
        cross_attn_norm=config.get("dit_cross_attn_norm", False),
        eps=config.get("dit_eps", 1e-6),
        use_latent_condition=config.get("dit_use_latent_condition", True),
    )
    transformer.load_state_dict(transformer_state_dict, strict=True)
    transformer = transformer.to(dtype=torch_dtype)

    vae_config = dict(config["vae_config"])
    vae_config.pop("model_type", None)
    vae = LongCatAudioDiTVae(**vae_config)
    vae.load_state_dict(vae_state_dict, strict=True)
    vae = vae.to(dtype=torch_dtype)

    text_encoder_config = UMT5Config.from_dict(config["text_encoder_config"])
    text_encoder = UMT5EncoderModel(text_encoder_config)
    text_missing, text_unexpected = text_encoder.load_state_dict(text_encoder_state_dict, strict=False)

    allowed_missing = {"shared.weight"}
    unexpected_missing = set(text_missing) - allowed_missing
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing text encoder weights: {sorted(unexpected_missing)}")
    if text_unexpected:
        raise RuntimeError(f"Unexpected text encoder weights: {sorted(text_unexpected)}")
    if "shared.weight" in text_missing:
        text_encoder.shared.weight.data.copy_(text_encoder.encoder.embed_tokens.weight.data)

    text_encoder = text_encoder.to(dtype=torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_model)

    scheduler_config = {"shift": 1.0, "invert_sigmas": True}
    scheduler_config.update(config.get("scheduler_config", {}))
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

    pipeline = LongCatAudioDiTPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )

    pipeline.sample_rate = config.get("sampling_rate", 24000)
    pipeline.vae_scale_factor = config.get("vae_scale_factor", config.get("latent_hop", 2048))
    pipeline.max_wav_duration = config.get("max_wav_duration", 30.0)
    pipeline.text_norm_feat = config.get("text_norm_feat", True)
    pipeline.text_add_embed = config.get("text_add_embed", True)

    output_path = Path(output_path) / f"{model_name}-Diffusers"
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline.save_pretrained(output_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to local model directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repo_id to download model",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for converted weights",
    )
    parser.add_argument(
        "--text_encoder_model",
        type=str,
        default="google/umt5-xxl",
        help="HuggingFace model ID for text encoder tokenizer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert_longcat_audio_dit(
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        output_path=args.output_path,
        dtype=args.dtype,
        text_encoder_model=args.text_encoder_model,
    )
