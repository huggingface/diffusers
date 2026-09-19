import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXConditionPipeline,
    LTXLatentUpsamplePipeline,
    LTXPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.loaders.conversion.configs.ltx import (
    get_spatial_latent_upsampler_config,
    get_transformer_config,
    get_vae_config,
)
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel


TOKENIZER_MAX_LENGTH = 128


def get_state_dict(saved_dict: Dict[str, Any]) -> dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def convert_transformer(ckpt_path: str, config, dtype: torch.dtype):
    state = get_state_dict(load_file(ckpt_path))
    return LTXVideoTransformer3DModel.from_single_file(state, config=config, torch_dtype=dtype)


def convert_vae(ckpt_path: str, config, dtype: torch.dtype):
    state = get_state_dict(load_file(ckpt_path))
    return AutoencoderKLLTXVideo.from_single_file(state, config=config, torch_dtype=dtype)


def convert_spatial_latent_upsampler(ckpt_path: str, config, dtype: torch.dtype):
    state = get_state_dict(load_file(ckpt_path))
    return LTXLatentUpsamplerModel.from_single_file(state, config=config, torch_dtype=dtype)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original vae checkpoint")
    parser.add_argument(
        "--spatial_latent_upsampler_path",
        type=str,
        default=None,
        help="Path to original spatial latent upsampler checkpoint",
    )
    parser.add_argument(
        "--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory"
    )
    parser.add_argument(
        "--typecast_text_encoder",
        action="store_true",
        default=False,
        help="Whether or not to apply fp16/bf16 precision to text_encoder",
    )
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="fp32", help="Torch dtype to save the model in.")
    parser.add_argument(
        "--version",
        type=str,
        default="0.9.0",
        choices=["0.9.0", "0.9.1", "0.9.5", "0.9.7", "0.9.8"],
        help="Version of the LTX model",
    )
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

VARIANT_MAPPING = {
    "fp32": None,
    "fp16": "fp16",
    "bf16": "bf16",
}


if __name__ == "__main__":
    args = get_args()

    transformer = None
    dtype = DTYPE_MAPPING[args.dtype]
    variant = VARIANT_MAPPING[args.dtype]
    output_path = Path(args.output_path)

    if args.transformer_ckpt_path is not None:
        config = get_transformer_config(args.version)
        transformer: LTXVideoTransformer3DModel = convert_transformer(args.transformer_ckpt_path, config, dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(
                output_path / "transformer", safe_serialization=True, max_shard_size="5GB", variant=variant
            )

    if args.vae_ckpt_path is not None:
        config = get_vae_config(args.version)
        vae: AutoencoderKLLTXVideo = convert_vae(args.vae_ckpt_path, config, dtype)
        if not args.save_pipeline:
            vae.save_pretrained(output_path / "vae", safe_serialization=True, max_shard_size="5GB", variant=variant)

    if args.spatial_latent_upsampler_path is not None:
        config = get_spatial_latent_upsampler_config(args.version)
        latent_upsampler: LTXLatentUpsamplerModel = convert_spatial_latent_upsampler(
            args.spatial_latent_upsampler_path, config, dtype
        )
        if not args.save_pipeline:
            latent_upsampler.save_pretrained(
                output_path / "latent_upsampler", safe_serialization=True, max_shard_size="5GB", variant=variant
            )

    if args.save_pipeline:
        text_encoder_id = "google/t5-v1_1-xxl"
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)

        if args.typecast_text_encoder:
            text_encoder = text_encoder.to(dtype=dtype)

        # Apparently, the conversion does not work anymore without this :shrug:
        for param in text_encoder.parameters():
            param.data = param.data.contiguous()

        if args.version in ["0.9.5", "0.9.7"]:
            scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False)
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=True,
                base_shift=0.95,
                max_shift=2.05,
                base_image_seq_len=1024,
                max_image_seq_len=4096,
                shift_terminal=0.1,
            )

        if args.version in ["0.9.0", "0.9.1", "0.9.5"]:
            pipe = LTXPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
            )
            pipe.save_pretrained(
                output_path.as_posix(), safe_serialization=True, variant=variant, max_shard_size="5GB"
            )
        elif args.version in ["0.9.7"]:
            pipe = LTXConditionPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
            )
            pipe_upsample = LTXLatentUpsamplePipeline(
                vae=vae,
                latent_upsampler=latent_upsampler,
            )
            pipe.save_pretrained(
                (output_path / "ltx_pipeline").as_posix(),
                safe_serialization=True,
                variant=variant,
                max_shard_size="5GB",
            )
            pipe_upsample.save_pretrained(
                (output_path / "ltx_upsample_pipeline").as_posix(),
                safe_serialization=True,
                variant=variant,
                max_shard_size="5GB",
            )
        else:
            raise ValueError(f"Unsupported version: {args.version}")
