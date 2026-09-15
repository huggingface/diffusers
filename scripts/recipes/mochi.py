import argparse

import torch
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler, MochiPipeline, MochiTransformer3DModel
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint


TOKENIZER_MAX_LENGTH = 256

parser = argparse.ArgumentParser()
parser.add_argument("--transformer_checkpoint_path", default=None, type=str)
parser.add_argument("--vae_encoder_checkpoint_path", default=None, type=str)
parser.add_argument("--vae_decoder_checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", required=True, type=str)
parser.add_argument("--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving")
parser.add_argument("--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory")
parser.add_argument("--dtype", type=str, default=None)

args = parser.parse_args()


# This is specific to `AdaLayerNormContinuous`:
# Diffusers implementation split the linear projection into the scale, shift while Mochi split it into shift, scale


def convert_mochi_transformer_checkpoint_to_diffusers(ckpt_path):
    state = load_file(ckpt_path, device="cpu")

    return convert_component_checkpoint(state, {}, "MochiTransformer3DModel")


def convert_mochi_vae_state_dict_to_diffusers(encoder_path, decoder_path):
    encoder = load_file(encoder_path, device="cpu")
    decoder = load_file(decoder_path, device="cpu")
    state = {f"encoder.{key}": value for key, value in encoder.items()}
    state.update({f"decoder.{key}": value for key, value in decoder.items()})
    return convert_component_checkpoint(state, {"latent_channels": 12, "out_channels": 3}, "AutoencoderKLMochi")


def main(args):
    if args.dtype is None:
        dtype = None
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    transformer = None
    vae = None

    if args.transformer_checkpoint_path is not None:
        converted_transformer_state_dict = convert_mochi_transformer_checkpoint_to_diffusers(
            args.transformer_checkpoint_path
        )
        transformer = MochiTransformer3DModel()
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        if dtype is not None:
            transformer = transformer.to(dtype=dtype)

    if args.vae_encoder_checkpoint_path is not None and args.vae_decoder_checkpoint_path is not None:
        vae = AutoencoderKLMochi(latent_channels=12, out_channels=3)
        converted_vae_state_dict = convert_mochi_vae_state_dict_to_diffusers(
            args.vae_encoder_checkpoint_path, args.vae_decoder_checkpoint_path
        )
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        if dtype is not None:
            vae = vae.to(dtype=dtype)

    text_encoder_id = "google/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)

    # Apparently, the conversion does not work anymore without this :shrug:
    for param in text_encoder.parameters():
        param.data = param.data.contiguous()

    pipe = MochiPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(invert_sigmas=True),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB", push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main(args)
