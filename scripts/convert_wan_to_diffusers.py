import argparse
import pathlib
from typing import Any, Dict

import torch
from accelerate import init_empty_weights
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
    WanTransformer3DModel,
)


TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {}

VAE_KEYS_RENAME_DICT = {}

VAE_SPECIAL_KEYS_REMAP = {}


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def load_sharded_safetensors(dir: pathlib.Path):
    file_paths = list(dir.glob("diffusion_pytorch_model*.safetensors"))
    print(file_paths)
    state_dict = {}
    for path in file_paths:
        state_dict.update(load_file(path))
    return state_dict


def get_transformer_config(model_type: str) -> Dict[str, Any]:
    if model_type == "Wan-T2V-1.3B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-1.3B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    return config


def convert_transformer(model_type: str):
    config = get_transformer_config(model_type)
    diffusers_config = config["diffusers_config"]
    model_id = config["model_id"]
    model_dir = pathlib.Path(snapshot_download(model_id, repo_type="model"))

    original_state_dict = load_sharded_safetensors(model_dir)

    with init_empty_weights():
        transformer = WanTransformer3DModel.from_config(diffusers_config)

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


# def convert_vae(ckpt_path: str):
#     original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

#     with init_empty_weights():
#         vae = AutoencoderKLWan()

#     for key in list(original_state_dict.keys()):
#         new_key = key[:]
#         for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
#             new_key = new_key.replace(replace_key, rename_key)
#         update_state_dict_(original_state_dict, key, new_key)

#     for key in list(original_state_dict.keys()):
#         for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
#             if special_key not in key:
#                 continue
#             handler_fn_inplace(key, original_state_dict)

#     vae.load_state_dict(original_state_dict, strict=True, assign=True)
#     return vae


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None)
    # parser.add_argument("--vae_ckpt_path", type=str, default=None)
    # parser.add_argument("--text_encoder_path", type=str, default=None)
    # parser.add_argument("--tokenizer_path", type=str, default=None)
    # parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="fp32")
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

    transformer = convert_transformer(args.model_type)
    transformer = transformer.to(dtype=dtype)
    transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    # if args.vae_ckpt_path is not None:
    #     vae = convert_vae(args.vae_ckpt_path)
    #     if not args.save_pipeline:
    #         vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    # if args.save_pipeline:
    #     # TODO(aryan): update these
    #     text_encoder = AutoModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.float16)
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right")
    #     scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

    #     pipe = WanPipeline(
    #         transformer=transformer,
    #         vae=vae,
    #         text_encoder=text_encoder,
    #         tokenizer=tokenizer,
    #         scheduler=scheduler,
    #     )
    #     pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
