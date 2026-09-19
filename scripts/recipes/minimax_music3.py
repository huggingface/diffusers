# Conversion script for MiniMax Music 3 (https://huggingface.co/MiniMaxAI/MiniMax-Music3).
#
# Original checkpoint layout:
#   flowmatching_vae.pth               flow-matching DiT + condition projection
#   dav.pth                            Flow-VAE (DAC-style) decoder
#   qwen_7B/qwen_7B/                   Qwen3 backbone + audio embedding + RVQ depth decoder (sharded safetensors)
#   qwen_7B/qwen3-8B-tokenizer-music/  music tokenizer
#
# Usage:
#   python scripts/recipes/minimax_music3.py \
#       --checkpoint_dir MiniMaxAI/MiniMax-Music3 --output_path ./minimax-music3-diffusers

import argparse
import json
import os

import torch
from safetensors.torch import load_file

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
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


def convert_transformer(dit_state_dict):
    prefix = "diffusion_transformer."
    state = {key: value for key, value in dit_state_dict.items() if key.startswith(prefix)}
    num_layers = 1 + max(int(key.split(".")[3]) for key in state if key.startswith(prefix + "transformer.layers."))
    return MiniMaxMusic3Transformer1DModel.from_single_file(state, config={"num_layers": num_layers})


def convert_condition_encoder(dit_state_dict):
    state = {
        key: value for key, value in dit_state_dict.items() if key.startswith(("cond_layer_", "latent_conditioners."))
    }
    return MiniMaxMusic3ConditionEncoder.from_single_file(state, config={})


def convert_vocoder(dav_state_dict):
    state = {key: value for key, value in dav_state_dict.items() if key.startswith(("dec_in_proj.", "decoder."))}
    return MiniMaxMusic3Vocoder.from_single_file(state, config={})


def convert_rvq_depth_decoder(qwen_state_dict, model_config):
    state = {
        key: value
        for key, value in qwen_state_dict.items()
        if key.startswith(("model.audio_decoder.", "model.audio_extra_embedding."))
    }
    config = {
        "hidden_size": int(model_config["hidden_size"]),
        "num_layers": int(model_config["decoder_num_layers"]),
        "num_attention_heads": int(model_config["decoder_num_heads"]),
        "intermediate_size": int(model_config["decoder_intermediate_size"]),
        "audio_vocab_size": int(model_config["audio_vocab_size"]),
        "num_codebooks": int(model_config["audio_num_codebooks"]),
    }
    return MiniMaxMusic3RVQDepthDecoder.from_single_file(state, config=config)


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

    from diffusers import MiniMaxMusic3Blocks

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, "qwen_7B", "qwen3-8B-tokenizer-music"))
    # num_train_timesteps=1 keeps `scheduler.timesteps` equal to the flow-matching time in [0, 1] that the
    # transformer's Fourier embedding expects.
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0, invert_sigmas=True)

    pipeline = MiniMaxMusic3Blocks().init_pipeline()
    pipeline.update_components(
        language_model=language_model,
        rvq_depth_decoder=rvq_depth_decoder,
        condition_encoder=condition_encoder,
        transformer=transformer,
        vocoder=vocoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    pipeline.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    # save_pretrained bakes the local output path into the modular index's loading specs; point them at the
    # Hub repo the components will be uploaded to instead.
    index_path = os.path.join(args.output_path, "modular_model_index.json")
    with open(index_path) as f:
        index = json.load(f)
    for entry in index.values():
        if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[2], dict):
            if entry[2].get("pretrained_model_name_or_path") == args.output_path:
                entry[2]["pretrained_model_name_or_path"] = args.repo_id
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="MiniMaxAI/MiniMax-Music3",
        help="Local directory or Hugging Face Hub repo id of the original checkpoint.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--repo_id",
        type=str,
        default="MiniMaxAI/MiniMax-Music3",
        help="Hub repo id the converted components will live in (written into the modular index loading specs).",
    )
    parser.add_argument("--dtype", type=lambda name: getattr(torch, name), default="float32")
    args = parser.parse_args()
    main(args)
