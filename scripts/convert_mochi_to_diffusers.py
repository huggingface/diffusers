import argparse
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

# from transformers import T5EncoderModel, T5Tokenizer
from diffusers import MochiTransformer3DModel
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext

TOKENIZER_MAX_LENGTH = 256

parser = argparse.ArgumentParser()
parser.add_argument("--transformer_checkpoint_path", default=None, type=str)
# parser.add_argument("--vae_checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", required=True, type=str)
parser.add_argument("--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving")
parser.add_argument("--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory")
parser.add_argument("--dtype", type=str, default=None)

args = parser.parse_args()


# This is specific to `AdaLayerNormContinuous`:
# Diffusers implementation split the linear projection into the scale, shift while Mochi split it into shift, scale
def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_mochi_transformer_checkpoint_to_diffusers(ckpt_path):
    original_state_dict = load_file(ckpt_path, device="cpu")
    new_state_dict = {}

    # Convert patch_embed
    new_state_dict["patch_embed.proj.weight"] = original_state_dict.pop("x_embedder.proj.weight")
    new_state_dict["patch_embed.proj.bias"] = original_state_dict.pop("x_embedder.proj.bias")

    # Convert time_embed
    new_state_dict["time_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop("t_embedder.mlp.0.weight")
    new_state_dict["time_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop("t_embedder.mlp.0.bias")
    new_state_dict["time_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop("t_embedder.mlp.2.weight")
    new_state_dict["time_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop("t_embedder.mlp.2.bias")
    new_state_dict["time_embed.pooler.to_kv.weight"] = original_state_dict.pop("t5_y_embedder.to_kv.weight")
    new_state_dict["time_embed.pooler.to_kv.bias"] = original_state_dict.pop("t5_y_embedder.to_kv.bias")
    new_state_dict["time_embed.pooler.to_q.weight"] = original_state_dict.pop("t5_y_embedder.to_q.weight")
    new_state_dict["time_embed.pooler.to_q.bias"] = original_state_dict.pop("t5_y_embedder.to_q.bias")
    new_state_dict["time_embed.pooler.to_out.weight"] = original_state_dict.pop("t5_y_embedder.to_out.weight")
    new_state_dict["time_embed.pooler.to_out.bias"] = original_state_dict.pop("t5_y_embedder.to_out.bias")
    new_state_dict["time_embed.caption_proj.weight"] = original_state_dict.pop("t5_yproj.weight")
    new_state_dict["time_embed.caption_proj.bias"] = original_state_dict.pop("t5_yproj.bias")

    # Convert transformer blocks
    num_layers = 48
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        old_prefix = f"blocks.{i}."

        # norm1
        new_state_dict[block_prefix + "norm1.linear.weight"] = original_state_dict.pop(old_prefix + "mod_x.weight")
        new_state_dict[block_prefix + "norm1.linear.bias"] = original_state_dict.pop(old_prefix + "mod_x.bias")
        if i < num_layers - 1:
            new_state_dict[block_prefix + "norm1_context.linear.weight"] = original_state_dict.pop(
                old_prefix + "mod_y.weight"
            )
            new_state_dict[block_prefix + "norm1_context.linear.bias"] = original_state_dict.pop(
                old_prefix + "mod_y.bias"
            )
        else:
            new_state_dict[block_prefix + "norm1_context.linear_1.weight"] = original_state_dict.pop(
                old_prefix + "mod_y.weight"
            )
            new_state_dict[block_prefix + "norm1_context.linear_1.bias"] = original_state_dict.pop(
                old_prefix + "mod_y.bias"
            )

        # Visual attention
        qkv_weight = original_state_dict.pop(old_prefix + "attn.qkv_x.weight")
        q, k, v = qkv_weight.chunk(3, dim=0)

        new_state_dict[block_prefix + "attn1.to_q.weight"] = q
        new_state_dict[block_prefix + "attn1.to_k.weight"] = k
        new_state_dict[block_prefix + "attn1.to_v.weight"] = v
        new_state_dict[block_prefix + "attn1.norm_q.weight"] = original_state_dict.pop(
            old_prefix + "attn.q_norm_x.weight"
        )
        new_state_dict[block_prefix + "attn1.norm_k.weight"] = original_state_dict.pop(
            old_prefix + "attn.k_norm_x.weight"
        )
        new_state_dict[block_prefix + "attn1.to_out.0.weight"] = original_state_dict.pop(
            old_prefix + "attn.proj_x.weight"
        )
        new_state_dict[block_prefix + "attn1.to_out.0.bias"] = original_state_dict.pop(old_prefix + "attn.proj_x.bias")

        # Context attention
        qkv_weight = original_state_dict.pop(old_prefix + "attn.qkv_y.weight")
        q, k, v = qkv_weight.chunk(3, dim=0)

        new_state_dict[block_prefix + "attn1.add_q_proj.weight"] = q
        new_state_dict[block_prefix + "attn1.add_k_proj.weight"] = k
        new_state_dict[block_prefix + "attn1.add_v_proj.weight"] = v
        new_state_dict[block_prefix + "attn1.norm_added_q.weight"] = original_state_dict.pop(
            old_prefix + "attn.q_norm_y.weight"
        )
        new_state_dict[block_prefix + "attn1.norm_added_k.weight"] = original_state_dict.pop(
            old_prefix + "attn.k_norm_y.weight"
        )
        if i < num_layers - 1:
            new_state_dict[block_prefix + "attn1.to_add_out.weight"] = original_state_dict.pop(
                old_prefix + "attn.proj_y.weight"
            )
            new_state_dict[block_prefix + "attn1.to_add_out.bias"] = original_state_dict.pop(
                old_prefix + "attn.proj_y.bias"
            )

        # MLP
        new_state_dict[block_prefix + "ff.net.0.proj.weight"] = original_state_dict.pop(old_prefix + "mlp_x.w1.weight")
        new_state_dict[block_prefix + "ff.net.2.weight"] = original_state_dict.pop(old_prefix + "mlp_x.w2.weight")
        if i < num_layers - 1:
            new_state_dict[block_prefix + "ff_context.net.0.proj.weight"] = original_state_dict.pop(
                old_prefix + "mlp_y.w1.weight"
            )
            new_state_dict[block_prefix + "ff_context.net.2.weight"] = original_state_dict.pop(
                old_prefix + "mlp_y.w2.weight"
            )

    # Output layers
    new_state_dict["norm_out.linear.weight"] = original_state_dict.pop("final_layer.mod.weight")
    new_state_dict["norm_out.linear.bias"] = original_state_dict.pop("final_layer.mod.bias")
    new_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    new_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")

    new_state_dict["pos_frequencies"] = original_state_dict.pop("pos_frequencies")

    print("Remaining Keys:", original_state_dict.keys())

    return new_state_dict


# def convert_mochi_vae_checkpoint_to_diffusers(ckpt_path, vae_config):
#     original_state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
#     return convert_ldm_vae_checkpoint(original_state_dict, vae_config)


def main(args):
    if args.dtype is None:
        dtype = None
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    transformer = None
    # vae = None

    if args.transformer_checkpoint_path is not None:
        converted_transformer_state_dict = convert_mochi_transformer_checkpoint_to_diffusers(
            args.transformer_checkpoint_path
        )
        transformer = MochiTransformer3DModel()
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        if dtype is not None:
            # Original checkpoint data type will be preserved
            transformer = transformer.to(dtype=dtype)

    # text_encoder_id = "google/t5-v1_1-xxl"
    # tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
    # text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)

    # # Apparently, the conversion does not work anymore without this :shrug:
    # for param in text_encoder.parameters():
    #     param.data = param.data.contiguous()

    transformer.save_pretrained("/raid/aryan/mochi-diffusers", subfolder="transformer")


if __name__ == "__main__":
    main(args)
