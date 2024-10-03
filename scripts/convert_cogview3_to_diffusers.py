import argparse
from typing import Any, Dict

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKL, CogVideoXDDIMScheduler
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

from diffusers import (
    CogView3PlusTransformer2DModel,
    CogView3PlusPipeline,
)


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
        new_key = f"transformer_blocks.{layer_id}.attn.norm_q.{weight_or_bias}"
    elif "key" in key:
        new_key = f"transformer_blocks.{layer_id}.attn.norm_k.{weight_or_bias}"

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


def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
    state_dict.pop(key)


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


TRANSFORMER_KEYS_RENAME_DICT = {
    "transformer": "transformer_blocks",
    "attention": "attn",
    "mlp": "mlp.net",
    "dense_h_to_4h": "0.proj",
    "dense_4h_to_h": "2",
    ".layers": "",
    "dense": "to_out.0",
    "mixins.patch_embed": "image_patch_embed",
    "mixins.adaln.adaln_modules": "adaln_module",
    "time_embed": "time_embed",
    "label_emb": "label_embed",
    "mixins.final_layer.adaln": "final_layer.adaln",
    "mixins.final_layer.linear": "proj_out",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "query_key_value": reassign_query_key_value_inplace,
}

TOKENIZER_MAX_LENGTH = 224


# VAE of CogView3Plus can be converted to diffusers without any changes
def convert_vae(ckpt_path: str, scaling_factor: float, dtype: torch.dtype):
    original_state_dict = torch.load(ckpt_path, map_location='cpu')["state_dict"]

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 512, 1024, 1024),
        layers_per_block=3,
        act_fn="silu",
        latent_channels=16,
        norm_num_groups=32,
        sample_size=1024,
        scaling_factor=scaling_factor,
        force_upcast=True,
        use_quant_conv=False,
        use_post_quant_conv=False,
        mid_block_add_attention=False,
    ).to(dtype=dtype)

    # Convert the state dict to a format compatible with diffusers
    converted_state_dict = convert_ldm_vae_checkpoint(original_state_dict, vae.config)

    # Load the converted state dict into the VAE model
    vae.load_state_dict(converted_state_dict, strict=False)

    return vae


def update_state_dict_inplace(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def convert_transformer(
        ckpt_path: str,
        num_layers: int,
        num_attention_heads: int,
        dtype: torch.dtype,
):
    PREFIX_KEY = "model.diffusion_model."

    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    transformer = CogView3PlusTransformer2DModel(
        in_channels=16,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
    ).to(dtype=dtype)

    for key in list(original_state_dict.keys()):
        new_key = key[len(PREFIX_KEY):]
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
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original vae checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--fp16", action="store_true", default=False, help="Whether to save the model weights in fp16")
    parser.add_argument("--bf16", action="store_true", default=False, help="Whether to save the model weights in bf16")
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving"
    )
    parser.add_argument(
        "--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory"
    )
    parser.add_argument("--num_layers", type=int, default=30, help="Number of transformer blocks")
    parser.add_argument("--num_attention_heads", type=int, default=64, help="Number of transformer blocks")
    parser.add_argument("--scaling_factor", type=float, default=0.18215, help="Scaling factor in the VAE")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    transformer = None
    vae = None

    if args.fp16 and args.bf16:
        raise ValueError("You cannot pass both --fp16 and --bf16 at the same time.")

    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(
            args.transformer_ckpt_path,
            args.num_layers,
            args.num_attention_heads,
            dtype
        )

    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path, args.scaling_factor, dtype)

    text_encoder_id = "/share/official_pretrains/hf_home/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)

    scheduler = CogVideoXDDIMScheduler.from_config(
        {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "set_alpha_to_one": True,
            "timestep_spacing": "trailing",
        }
    )

    pipe = CogView3PlusPipeline(
        tokenizer=tokenizer,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )

    if args.fp16:
        pipe = pipe.to(dtype=torch.float16)
    if args.bf16:
        pipe = pipe.to(dtype=torch.bfloat16)

    # We don't use variant here because the model must be run in fp16 (2B) or bf16 (5B). It would be weird
    # for users to specify variant when the default is not fp32 and they want to run with the correct default (which
    # is either fp16/bf16 here).

    # This is necessary This is necessary for users with insufficient memory,
    # such as those using Colab and notebooks, as it can save some memory used for model loading.
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB", push_to_hub=args.push_to_hub)
