import argparse
from pathlib import Path

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download


def download_checkpoint(
    repo_id="diffusers-internal-dev/new-ltx-model",
    filename="ltx-av-step-1932500-interleaved-new-vae.safetensors",
    device="cuda",
):
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    ckpt = safetensors.torch.load_file(ckpt_path, device=device)["audio_vae"]
    return ckpt


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


def load_original_decoder(device: torch.device, dtype: torch.dtype):
    from ltx_core.model.audio_vae.model_configurator import VAEDecoderConfigurator

    with torch.device("meta"):
        decoder = VAEDecoderConfigurator.from_config({}).to(device=device, dtype=dtype)
    original_state_dict = download_checkpoint(device)

    decoder_state_dict = {}
    for key, value in original_state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        trimmed = key
        if trimmed.startswith("audio_vae.decoder."):
            trimmed = trimmed[len("audio_vae.decoder.") :]
        elif trimmed.startswith("decoder."):
            trimmed = trimmed[len("decoder.") :]
        decoder_state_dict[trimmed] = value
    decoder.load_state_dict(decoder_state_dict, strict=True, assign=True)

    decoder.eval()
    return decoder


def build_diffusers_decoder(device: torch.device, dtype: torch.dtype):
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio

    with torch.device("meta"):
        model = AutoencoderKLLTX2Audio().to(device=device, dtype=dtype)

    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LTX2 audio decoder conversion.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    original_decoder = load_original_decoder(device, dtype)
    diffusers_model = build_diffusers_decoder(device, dtype)

    converted_state = convert_state_dict(original_decoder.state_dict())
    diffusers_model.load_state_dict(converted_state, assign=True, strict=True)

    levels = len(diffusers_model.decoder.channel_multipliers)
    latent_size = diffusers_model.decoder.resolution // (2 ** (levels - 1))
    dummy = torch.randn(
        args.batch, diffusers_model.decoder.latent_channels, latent_size, latent_size, device=device, dtype=dtype
    )

    original_out = original_decoder(dummy)
    diffusers_out = diffusers_model.decode(dummy).sample

    torch.testing.assert_close(diffusers_out, original_out, rtol=1e-4, atol=1e-4)
    max_diff = (diffusers_out - original_out).abs().max().item()
    print(f"Conversion successful. Max diff: {max_diff:.6f}")


if __name__ == "__main__":
    main()
