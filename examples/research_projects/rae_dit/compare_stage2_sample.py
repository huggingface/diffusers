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
from PIL import Image, ImageDraw

from diffusers import AutoencoderRAE, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_rae_dit import RAEDiT2DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a visual side-by-side sample comparison between upstream and diffusers Stage-2 RAE DiT."
    )
    parser.add_argument("--upstream_repo_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--converted_transformer_path", type=str, required=True)
    parser.add_argument("--vae_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--class_label", type=int, default=207, help="ImageNet class id to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--device", type=str, default=None)
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


def latent_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    return Image.fromarray(array)


def draw_label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), color="white")
    canvas.paste(image, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), text, fill="black")
    return canvas


def main():
    args = parse_args()
    upstream_repo_path = Path(args.upstream_repo_path).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    converted_transformer_path = Path(args.converted_transformer_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    sys.path.insert(0, str(upstream_repo_path / "src"))
    from stage2.models.DDT import DiTwDDTHead

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    stage2 = _resolve_section(config, "stage_2", "stage2")
    stage2_params = stage2.get("params", {})
    misc = _resolve_section(config, "misc")
    latent_size = misc["latent_size"]
    shift = math.sqrt(
        int(misc.get("time_dist_shift_dim", math.prod(latent_size))) / int(misc.get("time_dist_shift_base", 4096))
    )
    num_train_timesteps = int(_resolve_section(config, "transport").get("params", {}).get("num_train_timesteps", 1000))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = unwrap_state_dict(load_checkpoint(checkpoint_path), prefer_ema=True)

    upstream_model = DiTwDDTHead(**stage2_params)
    upstream_model.load_state_dict(state_dict, strict=True)
    upstream_model.to(device=device, dtype=torch.float32)
    upstream_model.eval()

    hf_model = RAEDiT2DModel.from_pretrained(converted_transformer_path, low_cpu_mem_usage=False)
    hf_model.to(device=device, dtype=torch.float32)
    hf_model.eval()

    vae = AutoencoderRAE.from_pretrained(args.vae_model_name_or_path, low_cpu_mem_usage=False)
    vae.to(device=device, dtype=torch.float32)
    vae.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents_init = torch.randn(
        (1, int(stage2_params["in_channels"]), int(stage2_params["input_size"]), int(stage2_params["input_size"])),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    class_labels = torch.tensor([args.class_label], device=device, dtype=torch.long)

    def run_sample(model, latents):
        latents = latents.clone()
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            stochastic_sampling=False,
        )
        scheduler.set_timesteps(args.num_inference_steps, device=device)
        with torch.no_grad():
            for timestep in scheduler.timesteps:
                timestep_input = timestep.expand(latents.shape[0]) / scheduler.config.num_train_timesteps
                if isinstance(model, DiTwDDTHead):
                    model_output = model(latents, timestep_input, class_labels)
                else:
                    model_output = model(
                        hidden_states=latents, timestep=timestep_input, class_labels=class_labels
                    ).sample
                latents = scheduler.step(model_output, timestep, latents).prev_sample
            return vae.decode(latents).sample.clamp(0, 1)

    upstream_image = run_sample(upstream_model, latents_init)[0]
    diffusers_image = run_sample(hf_model, latents_init)[0]
    abs_diff = (upstream_image - diffusers_image).abs()
    diff_vis = (abs_diff / max(abs_diff.max().item(), 1e-8)).clamp(0, 1)

    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    print(f"max_abs_error={max_abs_error:.8f}")
    print(f"mean_abs_error={mean_abs_error:.8f}")

    upstream_pil = draw_label(latent_to_pil(upstream_image), "Upstream")
    diffusers_pil = draw_label(latent_to_pil(diffusers_image), "Diffusers")
    diff_pil = draw_label(latent_to_pil(diff_vis), "Abs Diff")

    canvas = Image.new("RGB", (upstream_pil.width * 3, upstream_pil.height), color="white")
    canvas.paste(upstream_pil, (0, 0))
    canvas.paste(diffusers_pil, (upstream_pil.width, 0))
    canvas.paste(diff_pil, (upstream_pil.width * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
