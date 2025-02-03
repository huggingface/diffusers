import argparse
from typing import Any, Dict

import torch
from accelerate import init_empty_weights

from diffusers import CosmosTransformer3DModel


def remove_keys_(key: str, state_dict: Dict[str, Any]):
    state_dict.pop(key)


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def rename_transformer_blocks_(key: str, state_dict: Dict[str, Any]):
    block_index = int(key.split(".")[1].removeprefix("block"))
    new_key = key

    old_prefix = f"blocks.block{block_index}"
    new_prefix = f"transformer_blocks.{block_index}"
    new_key = new_prefix + new_key.removeprefix(old_prefix)

    state_dict[new_key] = state_dict.pop(key)


TRANSFORMER_KEYS_RENAME_DICT = {
    "t_embedder.1": "time_embed.t_embedder",
    "affline_norm": "time_embed.norm",
    ".blocks.0.block.attn": ".attn1",
    ".blocks.1.block.attn": ".attn2",
    ".blocks.2.block": ".ff",
    ".blocks.0.adaLN_modulation.1": ".norm1.linear_1",
    ".blocks.0.adaLN_modulation.2": ".norm1.linear_2",
    ".blocks.1.adaLN_modulation.1": ".norm2.linear_1",
    ".blocks.1.adaLN_modulation.2": ".norm2.linear_2",
    ".blocks.2.adaLN_modulation.1": ".norm3.linear_1",
    ".blocks.2.adaLN_modulation.2": ".norm3.linear_2",
    "to_q.0": "to_q",
    "to_q.1": "norm_q",
    "to_k.0": "to_k",
    "to_k.1": "norm_k",
    "to_v.0": "to_v",
    "layer1": "net.0.proj",
    "layer2": "net.2",
    "proj.1": "proj",
    "x_embedder": "patch_embed",
    "extra_pos_embedder": "learnable_pos_embed",
    "final_layer.adaLN_modulation.1": "norm_out.linear_1",
    "final_layer.adaLN_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "blocks.block": rename_transformer_blocks_,
    "logvar.0.freqs": remove_keys_,
    "logvar.0.phases": remove_keys_,
    "logvar.1.weight": remove_keys_,
    "pos_embedder.seq": remove_keys_,
}

VAE_KEYS_RENAME_DICT = {}

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


def convert_transformer(ckpt_path: str):
    PREFIX_KEY = "net."
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    with init_empty_weights():
        transformer = CosmosTransformer3DModel()

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        if new_key.startswith(PREFIX_KEY):
            new_key = new_key.removeprefix(PREFIX_KEY)
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
#         vae = AutoencoderKLHunyuanVideo()

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
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original VAE checkpoint")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to original llama checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to original llama tokenizer")
    parser.add_argument("--text_encoder_2_path", type=str, default=None, help="Path to original clip checkpoint")
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the transformer in.")
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
        assert args.text_encoder_path is not None
        assert args.tokenizer_path is not None
        assert args.text_encoder_2_path is not None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path)
        transformer = transformer.to(dtype=dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    # if args.vae_ckpt_path is not None:
    #     vae = convert_vae(args.vae_ckpt_path)
    #     if not args.save_pipeline:
    #         vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    # if args.save_pipeline:
    #     text_encoder = AutoModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.float16)
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right")
    #     text_encoder_2 = CLIPTextModel.from_pretrained(args.text_encoder_2_path, torch_dtype=torch.float16)
    #     tokenizer_2 = CLIPTokenizer.from_pretrained(args.text_encoder_2_path)
    #     scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

    #     pipe = CosmosPipeline(
    #         transformer=transformer,
    #         vae=vae,
    #         text_encoder=text_encoder,
    #         tokenizer=tokenizer,
    #         text_encoder_2=text_encoder_2,
    #         tokenizer_2=tokenizer_2,
    #         scheduler=scheduler,
    #     )
    #     pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
