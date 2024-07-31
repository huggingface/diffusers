import argparse
from typing import Any, Dict

import torch

from diffusers import CogVideoXTransformer3D


def reassign_query_key_value_inplace(key: str, state_dict: Dict[str, Any]):
    to_q_key = key.replace("query_key_value", "to_q")
    to_k_key = key.replace("query_key_value", "to_k")
    to_v_key = key.replace("query_key_value", "to_v")
    to_q, to_k, to_v = torch.chunk(state_dict[key], chunks=3, dim=0)
    state_dict[to_q_key] = to_q
    state_dict[to_k_key] = to_k
    state_dict[to_v_key] = to_v
    state_dict.pop(key)


def reassign_query_key_layernorm_inplace(key: str, state_dict: Dict[str, Any]):
    layer_id, weight_or_bias = key.split(".")[-2:]

    if "query" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_q.{weight_or_bias}"
    elif "key" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_k.{weight_or_bias}"

    state_dict[new_key] = state_dict.pop(key)


def reassign_adaln_norm_inplace(key: str, state_dict: Dict[str, Any]):
    layer_id, _, weight_or_bias = key.split(".")[-3:]

    weights_or_biases = state_dict[key].chunk(12, dim=0)
    norm1_weights_or_biases = torch.cat(weights_or_biases[0:3] + weights_or_biases[6:9])
    norm2_weights_or_biases = torch.cat(weights_or_biases[3:6] + weights_or_biases[9:12])

    norm1_key = f"transformer_blocks.{layer_id}.norm1.linear.{weight_or_bias}"
    state_dict[norm1_key] = norm1_weights_or_biases

    norm2_key = f"transformer_blocks.{layer_id}.norm2.linear.{weight_or_bias}"
    state_dict[norm2_key] = norm2_weights_or_biases

    state_dict.pop(key)


TRANSFORMER_KEYS_RENAME_DICT = {
    "transformer.final_layernorm": "norm_final",
    "transformer": "transformer_blocks",
    "attention": "attn1",
    "mlp": "ff.net",
    "dense_h_to_4h": "0.proj",
    "dense_4h_to_h": "2",
    ".layers": "",
    "dense": "to_out.0",
    "input_layernorm": "norm1.norm",
    "post_attn1_layernorm": "norm2.norm",
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    "mixins.patch_embed": "patch_embed",
    "mixins.final_layer.norm_final": "norm_out",
    "mixins.final_layer.linear": "proj_out",
    "mixins.final_layer.adaLN_modulation.1": "adaln_out.1",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "query_key_value": reassign_query_key_value_inplace,
    "query_layernorm_list": reassign_query_key_layernorm_inplace,
    "key_layernorm_list": reassign_query_key_layernorm_inplace,
    "adaln_layer.adaLN_modulations": reassign_adaln_norm_inplace,
}


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def update_state_dict_inplace(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def convert_transformer(ckpt_path: str, output_path: str, fp16: bool = False, push_to_hub: bool = False) -> None:
    PREFIX_KEY = "model.diffusion_model."

    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    transformer = CogVideoXTransformer3D()

    for key in list(original_state_dict.keys()):
        new_key = key[len(PREFIX_KEY) :]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    transformer.load_state_dict(original_state_dict, strict=True)
    transformer.save_pretrained(output_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformercheckpoint"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--fp16", action="store_true", default=False, help="Whether to save the model weights in fp16")
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.transformer_ckpt_path is not None:
        convert_transformer(args.transformer_ckpt_path, args.output_path, args.fp16, args.push_to_hub)
