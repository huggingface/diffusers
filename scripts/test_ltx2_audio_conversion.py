#!/usr/bin/env python
"""
Quick check that an LTX2 audio decoder checkpoint converts cleanly to the diffusers
`AutoencoderKLLTX2Audio` layout and produces matching outputs on dummy data.
"""

import argparse
import sys
from pathlib import Path

import torch


def convert_state_dict(state_dict: dict) -> dict:
    converted = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        if new_key.startswith("decoder."):
            new_key = new_key[len("decoder.") :]
        converted[f"decoder.{new_key}"] = value
    return converted


def load_original_decoder(original_repo: Path, device: torch.device, dtype: torch.dtype, checkpoint_path: Path | None):
    ltx_core_src = original_repo / "ltx-core" / "src"
    if not ltx_core_src.exists():
        raise FileNotFoundError(f"ltx-core sources not found under {ltx_core_src}")
    sys.path.insert(0, str(ltx_core_src))

    from ltx_core.model.audio_vae.model_configurator import VAEDecoderConfigurator

    decoder = VAEDecoderConfigurator.from_config({}).to(device=device, dtype=dtype)

    if checkpoint_path is not None:
        raw_state = torch.load(checkpoint_path, map_location=device)
        state_dict = raw_state.get("state_dict", raw_state)
        decoder_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            trimmed = key
            if trimmed.startswith("audio_vae.decoder."):
                trimmed = trimmed[len("audio_vae.decoder.") :]
            elif trimmed.startswith("decoder."):
                trimmed = trimmed[len("decoder.") :]
            decoder_state[trimmed] = value
        decoder.load_state_dict(decoder_state, strict=False)

    decoder.eval()
    return decoder


def build_diffusers_decoder(device: torch.device, dtype: torch.dtype):
    from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio

    model = AutoencoderKLLTX2Audio().to(device=device, dtype=dtype)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LTX2 audio decoder conversion.")
    parser.add_argument(
        "--original-repo",
        type=Path,
        default=Path("/Users/sayakpaul/Downloads/ltx-2"),
        help="Path to the original ltx-2 repository (needed to import ltx-core).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to an original checkpoint containing decoder weights.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    original_decoder = load_original_decoder(args.original_repo, device, dtype, args.checkpoint)
    diffusers_model = build_diffusers_decoder(device, dtype)

    converted_state = convert_state_dict(original_decoder.state_dict())
    diffusers_model.load_state_dict(converted_state, strict=False)

    levels = len(diffusers_model.decoder.channel_multipliers)
    latent_size = diffusers_model.decoder.resolution // (2 ** (levels - 1))
    dummy = torch.randn(args.batch, diffusers_model.decoder.latent_channels, latent_size, latent_size, device=device, dtype=dtype)

    with torch.no_grad():
        original_out = original_decoder(dummy)
        diffusers_out = diffusers_model.decode(dummy).sample

    torch.testing.assert_close(diffusers_out, original_out, rtol=1e-4, atol=1e-4)
    max_diff = (diffusers_out - original_out).abs().max().item()
    print(f"Conversion successful. Max diff: {max_diff:.6f}")


if __name__ == "__main__":
    main()
