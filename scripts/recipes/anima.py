"""
Convert Anima checkpoints to Diffusers format.

Example:
```bash
python scripts/recipes/anima.py \
    --transformer_ckpt_path anima_model/anima-preview3-base.safetensors \
    --text_encoder_ckpt_path anima_model/qwen_3_06b_base.safetensors \
    --vae_ckpt_path anima_model/qwen_image_vae.safetensors \
    --qwen_tokenizer_path path/to/qwen25_tokenizer \
    --t5_tokenizer_path path/to/t5_tokenizer \
    --output_path anima_model/anima-preview3-diffusers \
    --save_pipeline
```
"""

import argparse
import pathlib
import sys

import torch
from accelerate import init_empty_weights
from cosmos import convert_transformer
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Model, T5TokenizerFast

from diffusers import (
    AnimaAutoBlocks,
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.loaders.conversion.configs.anima import infer_qwen3_config, infer_text_conditioner_config


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def convert_qwen_image_vae(state_dict):
    return AutoencoderKLQwenImage.from_single_file(state_dict, config={})


def convert_text_conditioner(state_dict):
    return AnimaTextConditioner.from_single_file(state_dict, config=infer_text_conditioner_config(state_dict))


def convert_text_encoder(state_dict: dict[str, torch.Tensor]) -> Qwen3Model:
    state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    config = infer_qwen3_config(state_dict)
    with init_empty_weights():
        text_encoder = Qwen3Model(config)

    expected_keys = set(text_encoder.state_dict().keys())
    converted_keys = set(state_dict.keys())
    missing_keys = expected_keys - converted_keys
    unexpected_keys = converted_keys - expected_keys
    if missing_keys or unexpected_keys:
        if missing_keys:
            print(f"ERROR: missing Qwen3 keys ({len(missing_keys)}):", file=sys.stderr)
            for key in sorted(missing_keys):
                print(key, file=sys.stderr)
        if unexpected_keys:
            print(f"ERROR: unexpected Qwen3 keys ({len(unexpected_keys)}):", file=sys.stderr)
            for key in sorted(unexpected_keys):
                print(key, file=sys.stderr)
        sys.exit(1)

    text_encoder.load_state_dict(state_dict, strict=True, assign=True)
    return text_encoder


def split_anima_transformer_checkpoint(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    transformer_state_dict = {}
    text_conditioner_state_dict = {}
    adapter_prefix = "net.llm_adapter."

    for key, value in state_dict.items():
        if key.startswith(adapter_prefix):
            text_conditioner_state_dict[key.removeprefix(adapter_prefix)] = value
        else:
            transformer_state_dict[key] = value

    return transformer_state_dict, text_conditioner_state_dict


def save_pipeline(args, transformer, text_conditioner, text_encoder, vae):
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_tokenizer_path)
    t5_tokenizer = T5TokenizerFast.from_pretrained(args.t5_tokenizer_path)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    pipe = AnimaAutoBlocks().init_pipeline()
    pipe.update_components(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        t5_tokenizer=t5_tokenizer,
        text_conditioner=text_conditioner,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size=args.max_shard_size)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_ckpt_path", type=str, required=True, help="Path to Anima DiT safetensors")
    parser.add_argument("--text_encoder_ckpt_path", type=str, required=True, help="Path to Qwen3 text encoder")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to Qwen-Image VAE safetensors")
    parser.add_argument("--qwen_tokenizer_path", type=str, default=None)
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAPPING.keys()))
    parser.add_argument("--max_shard_size", default="5GB")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    output_path = pathlib.Path(args.output_path)
    dtype = DTYPE_MAPPING[args.dtype]

    raw_transformer_state_dict = load_file(args.transformer_ckpt_path, device="cpu")
    transformer_state_dict, text_conditioner_state_dict = split_anima_transformer_checkpoint(
        raw_transformer_state_dict
    )
    transformer = convert_transformer(
        "Cosmos-2.0-Diffusion-2B-Text2Image", state_dict=transformer_state_dict, weights_only=True
    ).to(dtype=dtype)
    text_conditioner = convert_text_conditioner(text_conditioner_state_dict).to(dtype=dtype)

    text_encoder_state_dict = load_file(args.text_encoder_ckpt_path, device="cpu")
    text_encoder = convert_text_encoder(text_encoder_state_dict).to(dtype=dtype)

    vae_state_dict = load_file(args.vae_ckpt_path, device="cpu")
    vae = convert_qwen_image_vae(vae_state_dict).to(dtype=dtype)

    if args.save_pipeline:
        if args.qwen_tokenizer_path is None or args.t5_tokenizer_path is None:
            raise ValueError("`--qwen_tokenizer_path` and `--t5_tokenizer_path` are required with `--save_pipeline`.")
        save_pipeline(args, transformer, text_conditioner, text_encoder, vae)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(
            output_path / "transformer", safe_serialization=True, max_shard_size=args.max_shard_size
        )
        text_conditioner.save_pretrained(
            output_path / "text_conditioner", safe_serialization=True, max_shard_size=args.max_shard_size
        )
        text_encoder.save_pretrained(
            output_path / "text_encoder", safe_serialization=True, max_shard_size=args.max_shard_size
        )
        vae.save_pretrained(output_path / "vae", safe_serialization=True, max_shard_size=args.max_shard_size)
