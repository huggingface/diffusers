# Conversion script for MiniMax Music 3 (https://huggingface.co/MiniMaxAI/MiniMax-Music3).
#
# Original checkpoint layout:
#   flowmatching_vae.pth               flow-matching DiT + condition projection
#   dav.pth                            Flow-VAE (DAC-style) decoder
#   qwen_7B/qwen_7B/                   Qwen3 backbone + audio embedding + RVQ depth decoder (sharded safetensors)
#   qwen_7B/qwen3-8B-tokenizer-music/  music tokenizer
#
# Usage:
#   python scripts/convert_minimax_music3_to_diffusers.py \
#       --checkpoint_dir MiniMaxAI/MiniMax-Music3 --output_path ./minimax-music3-diffusers

import argparse
import json
import os

import torch
from safetensors.torch import load_file

from diffusers import FlowMatchEulerDiscreteScheduler, MiniMaxMusic3Transformer1DModel
from diffusers.pipelines.minimax_music3.modeling_minimax_music3 import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Vocoder,
)


def load_dit_state_dict(checkpoint_dir: str) -> dict:
    return torch.load(os.path.join(checkpoint_dir, "flowmatching_vae.pth"), map_location="cpu", weights_only=True)


def load_dav_state_dict(checkpoint_dir: str) -> dict:
    return torch.load(os.path.join(checkpoint_dir, "dav.pth"), map_location="cpu", weights_only=True)


def load_qwen_state_dict(checkpoint_dir: str) -> dict:
    qwen_dir = os.path.join(checkpoint_dir, "qwen_7B", "qwen_7B")
    with open(os.path.join(qwen_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    state_dict = {}
    for filename in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(os.path.join(qwen_dir, filename), device="cpu"))
    return state_dict


def convert_transformer(dit_state_dict: dict) -> MiniMaxMusic3Transformer1DModel:
    prefix = "diffusion_transformer."
    converted = {
        "time_proj.weight": dit_state_dict[prefix + "timestep_features.weight"],
        "time_embed.linear_1.weight": dit_state_dict[prefix + "to_timestep_embed.0.weight"],
        "time_embed.linear_1.bias": dit_state_dict[prefix + "to_timestep_embed.0.bias"],
        "time_embed.linear_2.weight": dit_state_dict[prefix + "to_timestep_embed.2.weight"],
        "time_embed.linear_2.bias": dit_state_dict[prefix + "to_timestep_embed.2.bias"],
        "preprocess_conv.weight": dit_state_dict[prefix + "preprocess_conv.weight"],
        "postprocess_conv.weight": dit_state_dict[prefix + "postprocess_conv.weight"],
        "proj_in.weight": dit_state_dict[prefix + "transformer.project_in.weight"],
        "proj_out.weight": dit_state_dict[prefix + "transformer.project_out.weight"],
    }
    num_layers = 0
    while prefix + f"transformer.layers.{num_layers}.pre_norm.gamma" in dit_state_dict:
        num_layers += 1
    for i in range(num_layers):
        original = prefix + f"transformer.layers.{i}."
        target = f"transformer_blocks.{i}."
        converted[target + "norm1.weight"] = dit_state_dict[original + "pre_norm.gamma"]
        converted[target + "norm1.bias"] = dit_state_dict[original + "pre_norm.beta"]
        query, key, value = dit_state_dict[original + "self_attn.to_qkv.weight"].chunk(3, dim=0)
        converted[target + "attn.to_q.weight"] = query
        converted[target + "attn.to_k.weight"] = key
        converted[target + "attn.to_v.weight"] = value
        converted[target + "attn.to_out.0.weight"] = dit_state_dict[original + "self_attn.to_out.weight"]
        converted[target + "norm2.weight"] = dit_state_dict[original + "ff_norm.gamma"]
        converted[target + "norm2.bias"] = dit_state_dict[original + "ff_norm.beta"]
        converted[target + "ff_in.weight"] = dit_state_dict[original + "ff.ff.0.proj.weight"]
        converted[target + "ff_in.bias"] = dit_state_dict[original + "ff.ff.0.proj.bias"]
        converted[target + "ff_out.weight"] = dit_state_dict[original + "ff.ff.2.weight"]
        converted[target + "ff_out.bias"] = dit_state_dict[original + "ff.ff.2.bias"]

    with torch.device("meta"):
        transformer = MiniMaxMusic3Transformer1DModel(num_layers=num_layers)
    transformer.load_state_dict(converted, strict=True, assign=True)
    return transformer


def convert_condition_encoder(dit_state_dict: dict) -> MiniMaxMusic3ConditionEncoder:
    converted = {
        "layer_weight_logits": dit_state_dict["cond_layer_logits"],
        "layer_scale": dit_state_dict["cond_layer_scale"],
        "proj.weight": dit_state_dict["latent_conditioners.0.weight"],
        "proj.bias": dit_state_dict["latent_conditioners.0.bias"],
    }
    with torch.device("meta"):
        condition_encoder = MiniMaxMusic3ConditionEncoder()
    condition_encoder.load_state_dict(converted, strict=True, assign=True)
    return condition_encoder


def convert_vocoder(dav_state_dict: dict) -> MiniMaxMusic3Vocoder:
    converted = {
        "dec_in_proj.weight": dav_state_dict["dec_in_proj.weight"],
        "dec_in_proj.bias": dav_state_dict["dec_in_proj.bias"],
    }
    # The reference decoder is one nn.Sequential: [conv_in, block*4, snake, conv_out, tanh].
    for suffix in ("weight_g", "weight_v", "bias"):
        converted[f"conv_in.{suffix}"] = dav_state_dict[f"decoder.model.0.{suffix}"]
        converted[f"conv_out.{suffix}"] = dav_state_dict[f"decoder.model.6.{suffix}"]
    converted["snake_out.alpha"] = dav_state_dict["decoder.model.5.alpha"]
    for block_index in range(4):
        original = f"decoder.model.{block_index + 1}.block."
        target = f"blocks.{block_index}."
        converted[target + "snake1.alpha"] = dav_state_dict[original + "0.alpha"]
        for suffix in ("weight_g", "weight_v", "bias"):
            converted[target + f"conv_t1.{suffix}"] = dav_state_dict[original + f"1.{suffix}"]
        for unit_index, unit_name in ((2, "res_unit1"), (3, "res_unit2"), (4, "res_unit3")):
            converted[target + f"{unit_name}.snake1.alpha"] = dav_state_dict[original + f"{unit_index}.block.0.alpha"]
            converted[target + f"{unit_name}.snake2.alpha"] = dav_state_dict[original + f"{unit_index}.block.2.alpha"]
            for suffix in ("weight_g", "weight_v", "bias"):
                converted[target + f"{unit_name}.conv1.{suffix}"] = dav_state_dict[
                    original + f"{unit_index}.block.1.{suffix}"
                ]
                converted[target + f"{unit_name}.conv2.{suffix}"] = dav_state_dict[
                    original + f"{unit_index}.block.3.{suffix}"
                ]

    vocoder = MiniMaxMusic3Vocoder()
    vocoder.load_state_dict(converted, strict=True)
    return vocoder


def convert_rvq_depth_decoder(qwen_state_dict: dict, model_config: dict) -> MiniMaxMusic3RVQDepthDecoder:
    prefix = "model.audio_decoder."
    converted = {
        "audio_embeddings.weight": qwen_state_dict["model.audio_extra_embedding.weight"],
        "projection.weight": qwen_state_dict[prefix + "projection.weight"],
        "pos_embedding.weight": qwen_state_dict[prefix + "pos_embedding.weight"],
        "norm.weight": qwen_state_dict[prefix + "norm.weight"],
    }
    num_codebooks = int(model_config["audio_num_codebooks"])
    for i in range(num_codebooks - 1):
        converted[f"audio_heads.{i}.weight"] = qwen_state_dict[prefix + f"audio_heads.{i}.weight"]
    num_layers = int(model_config["decoder_num_layers"])
    for i in range(num_layers):
        original = prefix + f"layers.{i}."
        target = f"layers.{i}."
        converted[target + "input_layernorm.weight"] = qwen_state_dict[original + "input_layernorm.weight"]
        converted[target + "post_attention_layernorm.weight"] = qwen_state_dict[
            original + "post_attention_layernorm.weight"
        ]
        converted[target + "attn.to_q.weight"] = qwen_state_dict[original + "self_attn.q_proj.weight"]
        converted[target + "attn.to_k.weight"] = qwen_state_dict[original + "self_attn.k_proj.weight"]
        converted[target + "attn.to_v.weight"] = qwen_state_dict[original + "self_attn.v_proj.weight"]
        converted[target + "attn.to_out.weight"] = qwen_state_dict[original + "self_attn.o_proj.weight"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            converted[target + proj + ".weight"] = qwen_state_dict[original + f"mlp.{proj}.weight"]

    with torch.device("meta"):
        rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder(
            hidden_size=int(model_config["hidden_size"]),
            num_layers=num_layers,
            num_attention_heads=int(model_config["decoder_num_heads"]),
            intermediate_size=int(model_config["decoder_intermediate_size"]),
            audio_vocab_size=int(model_config["audio_vocab_size"]),
            num_codebooks=num_codebooks,
        )
    rvq_depth_decoder.load_state_dict(converted, strict=True, assign=True)
    return rvq_depth_decoder


def convert_language_model(qwen_state_dict: dict, model_config: dict):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        head_dim=model_config["head_dim"],
        max_position_embeddings=model_config.get("max_position_embeddings", 10240),
        rope_theta=model_config.get("rope_theta", 1000000),
        rms_norm_eps=model_config.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=model_config.get("tie_word_embeddings", False),
    )
    backbone_state_dict = {
        key: value
        for key, value in qwen_state_dict.items()
        if not key.startswith(("model.audio_extra_embedding", "model.audio_decoder."))
    }
    with torch.device("meta"):
        language_model = Qwen3ForCausalLM(config)
    language_model.load_state_dict(backbone_state_dict, strict=True, assign=True)
    return language_model


def main(args):
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        from huggingface_hub import snapshot_download

        checkpoint_dir = snapshot_download(checkpoint_dir)

    with open(os.path.join(checkpoint_dir, "qwen_7B", "qwen_7B", "config.json")) as f:
        model_config = json.load(f)

    dit_state_dict = load_dit_state_dict(checkpoint_dir)
    transformer = convert_transformer(dit_state_dict).to(args.dtype)
    condition_encoder = convert_condition_encoder(dit_state_dict).to(args.dtype)
    del dit_state_dict
    vocoder = convert_vocoder(load_dav_state_dict(checkpoint_dir)).to(args.dtype)

    qwen_state_dict = load_qwen_state_dict(checkpoint_dir)
    rvq_depth_decoder = convert_rvq_depth_decoder(qwen_state_dict, model_config).to(torch.bfloat16)
    language_model = convert_language_model(qwen_state_dict, model_config)
    del qwen_state_dict

    from transformers import AutoTokenizer

    from diffusers import MiniMaxMusic3Pipeline

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, "qwen_7B", "qwen3-8B-tokenizer-music"))
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0, invert_sigmas=True)

    pipeline = MiniMaxMusic3Pipeline(
        language_model=language_model,
        rvq_depth_decoder=rvq_depth_decoder,
        condition_encoder=condition_encoder,
        transformer=transformer,
        vocoder=vocoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    pipeline.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="MiniMaxAI/MiniMax-Music3",
        help="Local directory or Hugging Face Hub repo id of the original checkpoint.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", type=lambda name: getattr(torch, name), default="float32")
    args = parser.parse_args()
    main(args)
