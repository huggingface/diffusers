#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from diffusers.models.transformers.transformer_rae_dit import RAEDiT2DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a converted RAEDiT checkpoint against the upstream Stage-2 model."
    )
    parser.add_argument("--upstream_repo_path", type=str, required=True, help="Path to the cloned upstream RAE repo.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the upstream Stage-2 YAML config used for the published checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the upstream Stage-2 checkpoint (.pt or .safetensors).",
    )
    parser.add_argument(
        "--converted_transformer_path",
        type=str,
        required=True,
        help="Path to the converted diffusers transformer directory.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device to use. Defaults to cuda if available.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the synthetic parity inputs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for the parity run.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for parity.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for parity.")
    return parser.parse_args()


def _resolve_section(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        section = config.get(key)
        if isinstance(section, dict):
            return section
    raise KeyError(f"Could not find any of {keys} in config.")


def _maybe_strip_common_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if len(state_dict) > 0 and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def unwrap_state_dict(maybe_wrapped: dict[str, Any], prefer_ema: bool = True) -> dict[str, Any]:
    state_dict: dict[str, Any] | Any = maybe_wrapped

    if isinstance(state_dict, dict):
        candidate_keys = ["ema", "model", "state_dict"] if prefer_ema else ["model", "ema", "state_dict"]
        for key in candidate_keys:
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    if not isinstance(state_dict, dict):
        raise ValueError("Resolved checkpoint payload is not a dictionary state dict.")

    state_dict = dict(state_dict)
    for prefix in ("module.", "model.", "model.module."):
        state_dict = _maybe_strip_common_prefix(state_dict, prefix)
    return state_dict


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if checkpoint_path.suffix.lower() == ".safetensors":
        import safetensors.torch

        return safetensors.torch.load_file(checkpoint_path)

    return torch.load(checkpoint_path, map_location="cpu")


def build_inputs(
    batch_size: int,
    in_channels: int,
    sample_size: int,
    num_classes: int,
    shift: float,
    seed: int,
    device: torch.device,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    clean_latents = torch.randn(
        (batch_size, in_channels, sample_size, sample_size), generator=generator, device=device, dtype=torch.float32
    )
    noise = torch.randn(clean_latents.shape, generator=generator, device=device, dtype=torch.float32)

    # Use a spread of normalized timesteps inside the open interval to avoid any
    # boundary-case special handling around t=0 or t=1.
    timesteps = torch.linspace(0.2, 0.8, steps=batch_size, device=device, dtype=torch.float32)
    sigma = shift * timesteps / (1 + (shift - 1) * timesteps)
    sigma = sigma.view(-1, 1, 1, 1)

    noised_latents = (1.0 - sigma) * clean_latents + sigma * noise
    class_labels = torch.arange(batch_size, device=device, dtype=torch.long) % num_classes
    return noised_latents, timesteps, class_labels


def main():
    args = parse_args()

    upstream_repo_path = Path(args.upstream_repo_path).expanduser().resolve()
    sys.path.insert(0, str(upstream_repo_path / "src"))

    from stage2.models.DDT import DiTwDDTHead

    config_path = Path(args.config_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    converted_transformer_path = Path(args.converted_transformer_path).expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    stage2 = _resolve_section(config, "stage_2", "stage2")
    stage2_params = stage2.get("params", {})
    misc = _resolve_section(config, "misc")
    latent_size = misc["latent_size"]
    shift = math.sqrt(
        int(misc.get("time_dist_shift_dim", math.prod(latent_size))) / int(misc.get("time_dist_shift_base", 4096))
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = unwrap_state_dict(load_checkpoint(checkpoint_path), prefer_ema=True)

    upstream_model = DiTwDDTHead(**stage2_params)
    upstream_model.load_state_dict(state_dict, strict=True)
    upstream_model.to(device=device, dtype=torch.float32)
    upstream_model.eval()

    hf_model = RAEDiT2DModel.from_pretrained(converted_transformer_path, low_cpu_mem_usage=False)
    hf_model.to(device=device, dtype=torch.float32)
    hf_model.eval()

    noised_latents, timesteps, class_labels = build_inputs(
        batch_size=args.batch_size,
        in_channels=int(stage2_params["in_channels"]),
        sample_size=int(stage2_params["input_size"]),
        num_classes=int(stage2_params.get("num_classes", misc.get("num_classes", 1000))),
        shift=shift,
        seed=args.seed,
        device=device,
    )

    with torch.no_grad():
        upstream_output = upstream_model(noised_latents, timesteps, class_labels)
        hf_output = hf_model(hidden_states=noised_latents, timestep=timesteps, class_labels=class_labels).sample

    abs_error = (upstream_output - hf_output).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    print(f"device={device}")
    print(f"shape={tuple(hf_output.shape)}")
    print(f"max_abs_error={max_abs_error:.8f}")
    print(f"mean_abs_error={mean_abs_error:.8f}")

    if not torch.allclose(upstream_output, hf_output, atol=args.atol, rtol=args.rtol):
        raise AssertionError(
            f"Parity failed: max_abs_error={max_abs_error:.8f}, mean_abs_error={mean_abs_error:.8f}, "
            f"expected atol={args.atol}, rtol={args.rtol}"
        )


if __name__ == "__main__":
    main()
