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
    return converted


def load_original_decoder(device: torch.device):
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.audio_vae.model_configurator import VAEDecoderConfigurator as AudioDecoderConfigurator
    from ltx_core.model.audio_vae.model_configurator import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER
    
    checkpoint_path = download_checkpoint()
    
    # The code below comes from `ltx-pipelines/src/ltx_pipelines/txt2vid.py`
    decoder = Builder(
        model_path=checkpoint_path,
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_key_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device)

    state_dict = decoder.state_dict()
    for k, v in state_dict.items():
        if "mid" in k:
            print(f"{k=}")
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

    original_decoder = load_original_decoder(device)
    diffusers_model = build_diffusers_decoder()

    converted_state_dict = convert_state_dict(original_decoder.state_dict())
    diffusers_model.load_state_dict(converted_state_dict, assign=True, strict=True)

    levels = len(diffusers_model.decoder.channel_multipliers)
    latent_size = diffusers_model.decoder.resolution // (2 ** (levels - 1))
    dummy = torch.randn(
        args.batch, diffusers_model.decoder.latent_channels, latent_size, latent_size, device=device
    )

    original_out = original_decoder(dummy)
    diffusers_out = diffusers_model.decode(dummy).sample

    torch.testing.assert_close(diffusers_out, original_out, rtol=1e-4, atol=1e-4)
    max_diff = (diffusers_out - original_out).abs().max().item()
    print(f"Conversion successful. Max diff: {max_diff:.6f}")

    diffusers_model.to(dtype).save_pretrained(args.output_path)
    print(f"Serialized model to {args.output_path}")


if __name__ == "__main__":
    main()
