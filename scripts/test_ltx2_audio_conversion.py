import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


def download_checkpoint(
    repo_id="diffusers-internal-dev/new-ltx-model",
    filename="ltx-av-step-1932500-interleaved-new-vae.safetensors",
):
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return ckpt_path


def convert_state_dict(state_dict: dict) -> dict:
    converted = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        if new_key.startswith("decoder."):
            new_key = new_key[len("decoder.") :]
        converted[f"decoder.{new_key}"] = value

    converted["latents_mean"] = converted.pop("decoder.per_channel_statistics.mean-of-means")
    converted["latents_std"] = converted.pop("decoder.per_channel_statistics.std-of-means")
    return converted


def load_original_decoder(device: torch.device, dtype: torch.dtype):
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.audio_vae.model_configurator import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER
    from ltx_core.model.audio_vae.model_configurator import VAEDecoderConfigurator as AudioDecoderConfigurator

    checkpoint_path = download_checkpoint()

    # The code below comes from `ltx-pipelines/src/ltx_pipelines/txt2vid.py`
    decoder = Builder(
        model_path=checkpoint_path,
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_key_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device)

    decoder.eval()
    return decoder


def build_diffusers_decoder():
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio

    with torch.device("meta"):
        model = AutoencoderKLLTX2Audio()

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
    diffusers_model = build_diffusers_decoder()

    converted_state_dict = convert_state_dict(original_decoder.state_dict())
    diffusers_model.load_state_dict(converted_state_dict, assign=True, strict=False)

    per_channel_len = original_decoder.per_channel_statistics.get_buffer("std-of-means").numel()
    latent_channels = diffusers_model.decoder.latent_channels
    mel_bins_for_match = per_channel_len // latent_channels if per_channel_len % latent_channels == 0 else None

    levels = len(diffusers_model.decoder.channel_multipliers)
    latent_height = diffusers_model.decoder.resolution // (2 ** (levels - 1))
    latent_width = mel_bins_for_match or latent_height

    dummy = torch.randn(
        args.batch,
        diffusers_model.decoder.latent_channels,
        latent_height,
        latent_width,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device).manual_seed(42)
    )

    original_out = original_decoder(dummy)

    from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

    _, a_channels, a_time, a_freq = dummy.shape
    dummy = dummy.permute(0, 2, 1, 3).reshape(-1, a_time, a_channels * a_freq)
    dummy = LTX2Pipeline._denormalize_audio_latents(
        dummy,
        diffusers_model.latents_mean,
        diffusers_model.latents_std,
    )
    dummy = dummy.view(-1, a_time, a_channels, a_freq).permute(0, 2, 1, 3)
    diffusers_out = diffusers_model.decode(dummy).sample

    torch.testing.assert_close(diffusers_out, original_out, rtol=1e-4, atol=1e-4)
    max_diff = (diffusers_out - original_out).abs().max().item()
    print(f"Conversion successful. Max diff: {max_diff:.6f}")

    diffusers_model.to(dtype).save_pretrained(args.output_path)
    print(f"Serialized model to {args.output_path}")


if __name__ == "__main__":
    main()
