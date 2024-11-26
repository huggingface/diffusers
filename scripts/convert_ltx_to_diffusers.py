import argparse
from typing import Any, Dict

import torch
from safetensors.torch import load_file

from diffusers import LTXTransformer3DModel


TRANSFORMER_KEYS_RENAME_DICT = {
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {}


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


def convert_transformer(
    ckpt_path: str,
    dtype: torch.dtype,
):
    PREFIX_KEY = ""

    original_state_dict = get_state_dict(load_file(ckpt_path))
    transformer = LTXTransformer3DModel().to(dtype=dtype)

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
    return transformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="fp32", help="Torch dtype to save the model in.")
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

    if args.transformer_ckpt_path is not None:
        transformer: LTXTransformer3DModel = convert_transformer(args.transformer_ckpt_path, dtype)

    variant = VARIANT_MAPPING[args.dtype]
    transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
