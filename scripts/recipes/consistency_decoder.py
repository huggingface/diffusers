"""Assemble a consistency decoder VAE from the published TorchScript decoder and a Diffusers SD encoder."""

import argparse

import torch

from diffusers import AutoencoderKL, ConsistencyDecoderVAE
from diffusers.loaders.conversion import get_conversion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", required=True, help="Local published decoder.pt TorchScript file")
    parser.add_argument("--encoder", required=True, help="Diffusers AutoencoderKL directory containing the SD encoder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    encoder = AutoencoderKL.from_pretrained(args.encoder)
    original_encoder = get_conversion("AutoencoderKL", dict(encoder.config)).to_original(encoder.state_dict())
    state = {key: value for key, value in original_encoder.items() if key.startswith(("encoder.", "quant_conv."))}
    decoder = torch.jit.load(args.decoder, map_location="cpu")
    state.update({"decoder." + key: value for key, value in decoder.state_dict().items()})
    config = {
        "original_format": "consistency_decoder_jit",
        "scaling_factor": encoder.config.scaling_factor,
        "latent_channels": encoder.config.latent_channels,
        "encoder_block_out_channels": encoder.config.block_out_channels,
        "encoder_down_block_types": encoder.config.down_block_types,
        "encoder_layers_per_block": encoder.config.layers_per_block,
    }
    model = ConsistencyDecoderVAE.from_single_file(state, config=config, torch_dtype=getattr(torch, args.dtype))
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
