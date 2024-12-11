import argparse
from typing import Any, Dict

import torch
from accelerate import init_empty_weights

from diffusers import HunyuanVideoTransformer3DModel


def remap_norm_scale_shift_(key, state_dict):
    weight = state_dict.pop(key)
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    state_dict[key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")] = new_weight


TRANSFORMER_KEYS_RENAME_DICT = {
    # "time_in.mlp.0": "time_text_embed.timestep_embedder.linear_1",
    # "time_in.mlp.2": "time_text_embed.timestep_embedder.linear_2",
    # "guidance_in.mlp.0": "time_text_embed.guidance_embedder.linear_1",
    # "guidance_in.mlp.2": "time_text_embed.guidance_embedder.linear_2",
    # "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
    # "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
    "double_blocks": "transformer_blocks",
    "single_blocks": "single_transformer_blocks",
    "img_mod.linear": "norm1.linear",
    "img_norm1": "norm1.norm",
    "img_norm2": "norm2",
    "img_mlp": "ff",
    "txt_mod.linear": "norm1_context.linear",
    "txt_norm1": "norm1.norm",
    "txt_norm2": "norm2_context",
    "txt_mlp": "ff_context",
    "modulation.linear": "norm.linear",
    "pre_norm": "norm.norm",
    "final_layer.norm_final": "norm_out.norm",
    "final_layer.linear": "proj_out",
    "fc1": "net.0.proj",
    "fc2": "net.2",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
}

VAE_KEYS_RENAME_DICT = {}

VAE_SPECIAL_KEYS_REMAP = {}


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def convert_transformer(ckpt_path: str):
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    with init_empty_weights():
        transformer = HunyuanVideoTransformer3DModel()

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    transformer.load_state_dict(original_state_dict, strict=True, assign=True)
    return transformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the model in.")
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


if __name__ == "__main__":
    args = get_args()

    transformer = None
    dtype = DTYPE_MAPPING[args.dtype]

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None and args.vae_ckpt_path is not None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
