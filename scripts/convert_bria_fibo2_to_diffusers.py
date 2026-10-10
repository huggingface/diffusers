import argparse
import json
import os

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKLFlux2,
    BriaFibo2Pipeline,
    BriaFibo2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.configuration_utils import FrozenDict


"""
# Transformer only

python scripts/convert_bria_fibo2_to_diffusers.py \
  --original_repo_id briaai/fibo-2-turbo-merge \
  --output_path fibo-2-turbo-diffusers

# Full pipeline

python scripts/convert_bria_fibo2_to_diffusers.py \
  --original_repo_id briaai/fibo-2-turbo-merge \
  --output_path fibo-2-turbo-diffusers \
  --full_pipe
"""

parser = argparse.ArgumentParser()
parser.add_argument("--original_repo_id", default=None, type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--full_pipe", action="store_true")

args = parser.parse_args()


TRANSFORMER_KEYS_RENAME_DICT = {
    # one embedder and one final layer instead of Z-Image's one per patch size
    "all_x_embedder.2-1.": "x_embedder.",
    "all_final_layer.2-1.": "final_layer.",
    # injection blocks and Perceiver layers: cross-attention to the text encoder
    "cross_q.": "cross_attention.to_q.",
    "cross_k.": "cross_attention.to_k.",
    "cross_v.": "cross_attention.to_v.",
    "cross_out.": "cross_attention.to_out.0.",
    "cross_norm_q.": "cross_attention.norm_q.",
    "cross_norm_k.": "cross_attention.norm_k.",
    # Perceiver layers: self-attention among the gist tokens
    "self_q.": "self_attention.to_q.",
    "self_k.": "self_attention.to_k.",
    "self_v.": "self_attention.to_v.",
    "self_out.": "self_attention.to_out.0.",
    "self_norm_q.": "self_attention.norm_q.",
    "self_norm_k.": "self_attention.norm_k.",
}

# BriaFibo2Pipeline.text_encoder_out_layers: the Qwen3-VL layers behind the Perceiver, then each injection block
PIPELINE_TEXT_ENCODER_LAYERS = [[9, 20, 31], [5, 16, 27], [7, 18, 29], [9, 20, 31], [11, 22, 33], [13, 24, 35]]


def get_checkpoint_path(args):
    if args.original_repo_id is not None:
        allow_patterns = None if args.full_pipe else ["transformer/*"]
        return snapshot_download(args.original_repo_id, allow_patterns=allow_patterns)
    elif args.checkpoint_path is not None:
        return args.checkpoint_path
    else:
        raise ValueError("please provide either `original_repo_id` or a local `checkpoint_path`")


def load_original_checkpoint(checkpoint_path):
    transformer_path = os.path.join(checkpoint_path, "transformer")
    with open(os.path.join(transformer_path, "config.json")) as f:
        original_config = json.load(f)
    with open(os.path.join(transformer_path, "diffusion_pytorch_model.safetensors.index.json")) as f:
        shard_names = sorted(set(json.load(f)["weight_map"].values()))

    original_state_dict = {}
    for shard_name in shard_names:
        original_state_dict.update(safetensors.torch.load_file(os.path.join(transformer_path, shard_name)))
    return original_config, original_state_dict


def get_transformer_config(original_config):
    if original_config["perceiver_num_heads"] not in (None, original_config["n_heads"]):
        raise ValueError("The Perceiver must use as many attention heads as the main blocks.")
    # BriaFibo2Pipeline feeds the transformer fixed Qwen3-VL layers: the Perceiver's, then three per injection block
    text_encoder_layers = [original_config["perceiver_llm_layers"], *original_config["injection_llm_layers"]]
    if text_encoder_layers != PIPELINE_TEXT_ENCODER_LAYERS:
        raise ValueError(
            f"The checkpoint reads Qwen3-VL layers {text_encoder_layers}, but BriaFibo2Pipeline feeds it"
            f" {PIPELINE_TEXT_ENCODER_LAYERS}."
        )

    return {
        "in_channels": original_config["in_channels"],
        "patch_size": original_config["patch_size"],
        "dim": original_config["dim"],
        "n_layers": original_config["n_layers"],
        "n_refiner_layers": original_config["n_refiner_layers"],
        "n_heads": original_config["n_heads"],
        "cap_feat_dim": original_config["cond_llm_dim"],
        "injection_layer_ids": original_config["injection_layer_ids"],
        "perceiver_num_layers": original_config["perceiver_num_layers"],
        "min_num_gist_tokens": original_config["min_num_gist_tokens"],
        "max_num_gist_tokens": original_config["max_num_gist_tokens"],
        "gist_step": original_config["gist_step"],
        "gist_min_text_len": original_config["gist_min_text_len"],
        "gist_max_text_len": original_config["gist_max_text_len"],
    }


def convert_bria_fibo2_transformer_to_diffusers(original_config, original_state_dict):
    with init_empty_weights():
        transformer = BriaFibo2Transformer2DModel.from_config(get_transformer_config(original_config))

    converted_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key
        for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(old, new)
        converted_state_dict[new_key] = value

    transformer.load_state_dict(converted_state_dict, strict=True, assign=True)
    return transformer


def main(args):
    checkpoint_path = get_checkpoint_path(args)
    original_config, original_state_dict = load_original_checkpoint(checkpoint_path)
    transformer = convert_bria_fibo2_transformer_to_diffusers(original_config, original_state_dict)

    if not args.full_pipe:
        transformer.save_pretrained(os.path.join(args.output_path, "transformer"))
        return

    vae = AutoencoderKLFlux2.from_pretrained(checkpoint_path, subfolder="vae")
    # the original config carries `force_upcast`, which the Flux 2 VAE doesn't take; left in, every load warns about it
    vae_config = dict(vae.config)
    vae_config.pop("force_upcast", None)
    vae._internal_dict = FrozenDict(vae_config)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path, subfolder="text_encoder", dtype=torch.bfloat16
    )
    # fibo-2's limits on an image's size in pixels, not Qwen3-VL's defaults; they decide how many tokens an image to
    # edit becomes
    processor = Qwen3VLProcessor.from_pretrained(checkpoint_path, subfolder="text_encoder")
    processor.image_processor = AutoImageProcessor.from_pretrained(
        checkpoint_path, subfolder="text_encoder", size={"shortest_edge": 4 * 28 * 28, "longest_edge": 2048 * 28 * 28}
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")

    pipe = BriaFibo2Pipeline(
        transformer=transformer, scheduler=scheduler, vae=vae, text_encoder=text_encoder, processor=processor
    )
    pipe.save_pretrained(args.output_path)


if __name__ == "__main__":
    main(args)
