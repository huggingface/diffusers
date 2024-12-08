"""
Convert a CogView3 checkpoint to the Diffusers format.

This script converts a CogView3 checkpoint to the Diffusers format, which can then be used
with the Diffusers library.

Example usage:
    python scripts/convert_cogview3_to_diffusers.py \
        --transformer_checkpoint_path 'your path/cogview3plus_3b/1/mp_rank_00_model_states.pt' \
        --vae_checkpoint_path 'your path/3plus_ae/imagekl_ch16.pt' \
        --output_path "/raid/yiyi/cogview3_diffusers" \
        --dtype "bf16"

Arguments:
    --transformer_checkpoint_path: Path to Transformer state dict.
    --vae_checkpoint_path: Path to VAE state dict.
    --output_path: The path to save the converted model.
    --push_to_hub: Whether to push the converted checkpoint to the HF Hub or not. Defaults to `False`.
    --text_encoder_cache_dir: Cache directory where text encoder is located. Defaults to None, which means HF_HOME will be used
    --dtype: The dtype to save the model in (default: "bf16", options: "fp16", "bf16", "fp32"). If None, the dtype of the state dict is considered.

    Default is "bf16" because CogView3 uses bfloat16 for Training.

Note: You must provide either --original_state_dict_repo_id or --checkpoint_path.
"""

import argparse
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKL, CogVideoXDDIMScheduler, CogView3PlusPipeline, CogView3PlusTransformer2DModel
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available() else nullcontext

TOKENIZER_MAX_LENGTH = 224

parser = argparse.ArgumentParser()
parser.add_argument("--transformer_checkpoint_path", default=None, type=str)
parser.add_argument("--vae_checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", required=True, type=str)
parser.add_argument("--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving")
parser.add_argument("--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory")
parser.add_argument("--dtype", type=str, default="bf16")

args = parser.parse_args()


# this is specific to `AdaLayerNormContinuous`:
# diffusers implementation split the linear projection into the scale, shift while CogView3 split it tino shift, scale
def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_cogview3_transformer_checkpoint_to_diffusers(ckpt_path):
    original_state_dict = torch.load(ckpt_path, map_location="cpu")
    original_state_dict = original_state_dict["module"]
    original_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in original_state_dict.items()}

    new_state_dict = {}

    # Convert patch_embed
    new_state_dict["patch_embed.proj.weight"] = original_state_dict.pop("mixins.patch_embed.proj.weight")
    new_state_dict["patch_embed.proj.bias"] = original_state_dict.pop("mixins.patch_embed.proj.bias")
    new_state_dict["patch_embed.text_proj.weight"] = original_state_dict.pop("mixins.patch_embed.text_proj.weight")
    new_state_dict["patch_embed.text_proj.bias"] = original_state_dict.pop("mixins.patch_embed.text_proj.bias")

    # Convert time_condition_embed
    new_state_dict["time_condition_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "time_embed.0.weight"
    )
    new_state_dict["time_condition_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
        "time_embed.0.bias"
    )
    new_state_dict["time_condition_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "time_embed.2.weight"
    )
    new_state_dict["time_condition_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
        "time_embed.2.bias"
    )
    new_state_dict["time_condition_embed.condition_embedder.linear_1.weight"] = original_state_dict.pop(
        "label_emb.0.0.weight"
    )
    new_state_dict["time_condition_embed.condition_embedder.linear_1.bias"] = original_state_dict.pop(
        "label_emb.0.0.bias"
    )
    new_state_dict["time_condition_embed.condition_embedder.linear_2.weight"] = original_state_dict.pop(
        "label_emb.0.2.weight"
    )
    new_state_dict["time_condition_embed.condition_embedder.linear_2.bias"] = original_state_dict.pop(
        "label_emb.0.2.bias"
    )

    # Convert transformer blocks
    for i in range(30):
        block_prefix = f"transformer_blocks.{i}."
        old_prefix = f"transformer.layers.{i}."
        adaln_prefix = f"mixins.adaln.adaln_modules.{i}."

        new_state_dict[block_prefix + "norm1.linear.weight"] = original_state_dict.pop(adaln_prefix + "1.weight")
        new_state_dict[block_prefix + "norm1.linear.bias"] = original_state_dict.pop(adaln_prefix + "1.bias")

        qkv_weight = original_state_dict.pop(old_prefix + "attention.query_key_value.weight")
        qkv_bias = original_state_dict.pop(old_prefix + "attention.query_key_value.bias")
        q, k, v = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

        new_state_dict[block_prefix + "attn1.to_q.weight"] = q
        new_state_dict[block_prefix + "attn1.to_q.bias"] = q_bias
        new_state_dict[block_prefix + "attn1.to_k.weight"] = k
        new_state_dict[block_prefix + "attn1.to_k.bias"] = k_bias
        new_state_dict[block_prefix + "attn1.to_v.weight"] = v
        new_state_dict[block_prefix + "attn1.to_v.bias"] = v_bias

        new_state_dict[block_prefix + "attn1.to_out.0.weight"] = original_state_dict.pop(
            old_prefix + "attention.dense.weight"
        )
        new_state_dict[block_prefix + "attn1.to_out.0.bias"] = original_state_dict.pop(
            old_prefix + "attention.dense.bias"
        )

        new_state_dict[block_prefix + "ff.net.0.proj.weight"] = original_state_dict.pop(
            old_prefix + "mlp.dense_h_to_4h.weight"
        )
        new_state_dict[block_prefix + "ff.net.0.proj.bias"] = original_state_dict.pop(
            old_prefix + "mlp.dense_h_to_4h.bias"
        )
        new_state_dict[block_prefix + "ff.net.2.weight"] = original_state_dict.pop(
            old_prefix + "mlp.dense_4h_to_h.weight"
        )
        new_state_dict[block_prefix + "ff.net.2.bias"] = original_state_dict.pop(old_prefix + "mlp.dense_4h_to_h.bias")

    # Convert final norm and projection
    new_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        original_state_dict.pop("mixins.final_layer.adaln.1.weight"), dim=0
    )
    new_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        original_state_dict.pop("mixins.final_layer.adaln.1.bias"), dim=0
    )
    new_state_dict["proj_out.weight"] = original_state_dict.pop("mixins.final_layer.linear.weight")
    new_state_dict["proj_out.bias"] = original_state_dict.pop("mixins.final_layer.linear.bias")

    return new_state_dict


def convert_cogview3_vae_checkpoint_to_diffusers(ckpt_path, vae_config):
    original_state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    return convert_ldm_vae_checkpoint(original_state_dict, vae_config)


def main(args):
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    transformer = None
    vae = None

    if args.transformer_checkpoint_path is not None:
        converted_transformer_state_dict = convert_cogview3_transformer_checkpoint_to_diffusers(
            args.transformer_checkpoint_path
        )
        transformer = CogView3PlusTransformer2DModel()
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        if dtype is not None:
            # Original checkpoint data type will be preserved
            transformer = transformer.to(dtype=dtype)

    if args.vae_checkpoint_path is not None:
        vae_config = {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ("DownEncoderBlock2D",) * 4,
            "up_block_types": ("UpDecoderBlock2D",) * 4,
            "block_out_channels": (128, 512, 1024, 1024),
            "layers_per_block": 3,
            "act_fn": "silu",
            "latent_channels": 16,
            "norm_num_groups": 32,
            "sample_size": 1024,
            "scaling_factor": 1.0,
            "force_upcast": True,
            "use_quant_conv": False,
            "use_post_quant_conv": False,
            "mid_block_add_attention": False,
        }
        converted_vae_state_dict = convert_cogview3_vae_checkpoint_to_diffusers(args.vae_checkpoint_path, vae_config)
        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        if dtype is not None:
            vae = vae.to(dtype=dtype)

    text_encoder_id = "google/t5-v1_1-xxl"
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)

    # Apparently, the conversion does not work anymore without this :shrug:
    for param in text_encoder.parameters():
        param.data = param.data.contiguous()

    scheduler = CogVideoXDDIMScheduler.from_config(
        {
            "snr_shift_scale": 4.0,
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
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )

    # This is necessary for users with insufficient memory, such as those using Colab and notebooks, as it can
    # save some memory used for model loading.
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB", push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main(args)
