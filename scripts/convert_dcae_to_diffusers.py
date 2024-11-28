import argparse
from typing import Any, Dict

import torch
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderDC


def remove_keys_(key: str, state_dict: Dict[str, Any]):
    state_dict.pop(key)


TOKENIZER_MAX_LENGTH = 128

TRANSFORMER_KEYS_RENAME_DICT = {}

TRANSFORMER_SPECIAL_KEYS_REMAP = {}

VAE_KEYS_RENAME_DICT = {
    # common
    "main.": "",
    "op_list.": "",
    "norm.": "norm.norm.",
    "depth_conv": "conv_depth",
    "point_conv": "conv_point",
    "inverted_conv": "conv_inverted",
    "conv.conv.": "conv.",
    # encoder
    "encoder.project_in.conv": "encoder.conv_in",
    "encoder.project_out.0.conv": "encoder.conv_out",
    # decoder
    "decoder.project_in.conv": "decoder.conv_in",
    "decoder.project_out.0": "decoder.norm_out.norm",
    "decoder.project_out.2.conv": "decoder.conv_out",
}

VAE_SPECIAL_KEYS_REMAP = {}


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


# def convert_transformer(
#     ckpt_path: str,
#     dtype: torch.dtype,
# ):
#     PREFIX_KEY = ""

#     original_state_dict = get_state_dict(load_file(ckpt_path))
#     transformer = LTXTransformer3DModel().to(dtype=dtype)

#     for key in list(original_state_dict.keys()):
#         new_key = key[len(PREFIX_KEY) :]
#         for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
#             new_key = new_key.replace(replace_key, rename_key)
#         update_state_dict_inplace(original_state_dict, key, new_key)

#     for key in list(original_state_dict.keys()):
#         for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
#             if special_key not in key:
#                 continue
#             handler_fn_inplace(key, original_state_dict)

#     transformer.load_state_dict(original_state_dict, strict=True)
#     return transformer


def convert_vae(ckpt_path: str, dtype: torch.dtype):
    original_state_dict = get_state_dict(load_file(ckpt_path))
    vae = AutoencoderDC(
        in_channels=3,
        latent_channels=32,
        encoder_width_list=[128, 256, 512, 512, 1024, 1024],
        encoder_depth_list=[2, 2, 2, 3, 3, 3],
        encoder_block_type=["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
        encoder_norm="rms2d",
        encoder_act="silu",
        downsample_block_type="Conv",
        decoder_width_list=[128, 256, 512, 512, 1024, 1024],
        decoder_depth_list=[3, 3, 3, 3, 3, 3],
        decoder_block_type=["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
        decoder_norm="rms2d",
        decoder_act="silu",
        upsample_block_type="InterpolateConv",
        scaling_factor=0.41407,
    ).to(dtype=dtype)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True)
    return vae


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original vae checkpoint")
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

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None and args.vae_ckpt_path is not None

    # if args.transformer_ckpt_path is not None:
    #     transformer = convert_transformer(args.transformer_ckpt_path, dtype)
    #     if not args.save_pipeline:
    #         transformer.save_pretrained(
    #             args.output_path, safe_serialization=True, max_shard_size="5GB", variant=variant
    #         )

    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path, dtype)
        if not args.save_pipeline:
            vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB", variant=variant)
