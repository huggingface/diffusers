"""
Convert Anima checkpoints to Diffusers format.

Example:
```bash
python scripts/convert_anima_to_diffusers.py \
    --transformer_ckpt_path anima_model/anima-preview3-base.safetensors \
    --text_encoder_ckpt_path anima_model/qwen_3_06b_base.safetensors \
    --vae_ckpt_path anima_model/qwen_image_vae.safetensors \
    --qwen_tokenizer_path /home/user/Dev/ComfyUI/comfy/text_encoders/qwen25_tokenizer \
    --t5_tokenizer_path /home/user/Dev/ComfyUI/comfy/text_encoders/t5_tokenizer \
    --output_path anima_model/anima-preview3-diffusers \
    --save_pipeline
```
"""

import argparse
import pathlib
import sys
from typing import Any

import torch
from accelerate import init_empty_weights
from convert_cosmos_to_diffusers import convert_transformer
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast

from diffusers import (
    AnimaAutoBlocks,
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def rename_residual_key(key: str) -> str:
    replacements = {
        ".residual.0.": ".norm1.",
        ".residual.2.": ".conv1.",
        ".residual.3.": ".norm2.",
        ".residual.6.": ".conv2.",
        ".shortcut.": ".conv_shortcut.",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return key


def rename_mid_key(key: str) -> str:
    replacements = {
        ".middle.0.": ".mid_block.resnets.0.",
        ".middle.1.": ".mid_block.attentions.0.",
        ".middle.2.": ".mid_block.resnets.1.",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return rename_residual_key(key)


def rename_decoder_upsample_key(key: str) -> str:
    prefix = "decoder.upsamples."
    suffix = key.removeprefix(prefix)
    index_str, rest = suffix.split(".", 1)
    index = int(index_str)

    if index in (3, 7, 11):
        block_index = (index - 3) // 4
        new_key = f"decoder.up_blocks.{block_index}.upsamplers.0.{rest}"
    else:
        block_index = index // 4
        resnet_index = index % 4
        new_key = f"decoder.up_blocks.{block_index}.resnets.{resnet_index}.{rest}"

    return rename_residual_key(new_key)


def convert_qwen_image_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("conv1."):
            new_key = key.replace("conv1.", "quant_conv.", 1)
        elif key.startswith("conv2."):
            new_key = key.replace("conv2.", "post_quant_conv.", 1)
        elif key.startswith("encoder.conv1."):
            new_key = key.replace("encoder.conv1.", "encoder.conv_in.", 1)
        elif key.startswith("decoder.conv1."):
            new_key = key.replace("decoder.conv1.", "decoder.conv_in.", 1)
        elif key.startswith("encoder.downsamples."):
            new_key = rename_residual_key(key.replace("encoder.downsamples.", "encoder.down_blocks.", 1))
        elif key.startswith("decoder.upsamples."):
            new_key = rename_decoder_upsample_key(key)
        elif key.startswith("encoder.middle.") or key.startswith("decoder.middle."):
            new_key = rename_mid_key(key)
        elif key.startswith("encoder.head.0."):
            new_key = key.replace("encoder.head.0.", "encoder.norm_out.", 1)
        elif key.startswith("encoder.head.2."):
            new_key = key.replace("encoder.head.2.", "encoder.conv_out.", 1)
        elif key.startswith("decoder.head.0."):
            new_key = key.replace("decoder.head.0.", "decoder.norm_out.", 1)
        elif key.startswith("decoder.head.2."):
            new_key = key.replace("decoder.head.2.", "decoder.conv_out.", 1)
        else:
            new_key = rename_residual_key(key)

        if new_key in converted_state_dict:
            raise ValueError(f"Duplicate converted VAE key: {new_key}")
        converted_state_dict[new_key] = value

    return converted_state_dict


def convert_qwen_image_vae(state_dict: dict[str, torch.Tensor]) -> AutoencoderKLQwenImage:
    converted_state_dict = convert_qwen_image_vae_state_dict(state_dict)
    with init_empty_weights():
        vae = AutoencoderKLQwenImage()

    expected_keys = set(vae.state_dict().keys())
    converted_keys = set(converted_state_dict.keys())
    missing_keys = expected_keys - converted_keys
    unexpected_keys = converted_keys - expected_keys
    if missing_keys or unexpected_keys:
        if missing_keys:
            print(f"ERROR: missing VAE keys ({len(missing_keys)}):", file=sys.stderr)
            for key in sorted(missing_keys):
                print(key, file=sys.stderr)
        if unexpected_keys:
            print(f"ERROR: unexpected VAE keys ({len(unexpected_keys)}):", file=sys.stderr)
            for key in sorted(unexpected_keys):
                print(key, file=sys.stderr)
        sys.exit(1)

    vae.load_state_dict(converted_state_dict, strict=True, assign=True)
    return vae


def infer_text_conditioner_config(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    model_dim = state_dict["blocks.0.self_attn.q_proj.weight"].shape[0]
    source_dim = state_dict["blocks.0.cross_attn.k_proj.weight"].shape[1]
    target_vocab_size, target_dim = state_dict["embed.weight"].shape
    attention_head_dim = state_dict["blocks.0.self_attn.q_norm.weight"].shape[0]
    num_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("blocks."))

    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "model_dim": model_dim,
        "num_layers": num_layers,
        "num_attention_heads": model_dim // attention_head_dim,
        "target_vocab_size": target_vocab_size,
    }


def convert_text_conditioner(state_dict: dict[str, torch.Tensor]) -> AnimaTextConditioner:
    config = infer_text_conditioner_config(state_dict)
    with init_empty_weights():
        text_conditioner = AnimaTextConditioner(**config)

    expected_keys = set(text_conditioner.state_dict().keys())
    converted_keys = set(state_dict.keys())
    missing_keys = expected_keys - converted_keys
    unexpected_keys = converted_keys - expected_keys
    if missing_keys or unexpected_keys:
        if missing_keys:
            print(f"ERROR: missing text conditioner keys ({len(missing_keys)}):", file=sys.stderr)
            for key in sorted(missing_keys):
                print(key, file=sys.stderr)
        if unexpected_keys:
            print(f"ERROR: unexpected text conditioner keys ({len(unexpected_keys)}):", file=sys.stderr)
            for key in sorted(unexpected_keys):
                print(key, file=sys.stderr)
        sys.exit(1)

    text_conditioner.load_state_dict(state_dict, strict=True, assign=True)
    return text_conditioner


def infer_qwen3_config(state_dict: dict[str, torch.Tensor]) -> Qwen3Config:
    vocab_size, hidden_size = state_dict["embed_tokens.weight"].shape
    intermediate_size = state_dict["layers.0.mlp.gate_proj.weight"].shape[0]
    num_hidden_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("layers."))
    head_dim = state_dict["layers.0.self_attn.q_norm.weight"].shape[0]
    num_attention_heads = state_dict["layers.0.self_attn.q_proj.weight"].shape[0] // head_dim
    num_key_value_heads = state_dict["layers.0.self_attn.k_proj.weight"].shape[0] // head_dim

    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        head_dim=head_dim,
        attention_bias=False,
        tie_word_embeddings=False,
    )


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
    transformer_state_dict, text_conditioner_state_dict = split_anima_transformer_checkpoint(raw_transformer_state_dict)
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
