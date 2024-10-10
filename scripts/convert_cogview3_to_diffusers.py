"""
Convert a CogView3 checkpoint to the Diffusers format.

This script converts a CogView3 checkpoint to the Diffusers format, which can then be used
with the Diffusers library.

Example usage:
    python scripts/convert_cogview3_to_diffusers.py \
        --original_state_dict_repo_id "THUDM/cogview3" \
        --filename "cogview3.pt" \
        --transformer \
        --output_path "./cogview3_diffusers" \
        --dtype "bf16"

Alternatively, if you have a local checkpoint:
    python scripts/convert_cogview3_to_diffusers.py \
        --checkpoint_path '/raid/.cache/huggingface/models--ZP2HF--CogView3-SAT/snapshots/ca86ce9ba94f9a7f2dd109e7a59e4c8ad04121be/cogview3plus_3b/1/mp_rank_00_model_states.pt' \
        --transformer \
        --output_path "/raid/yiyi/cogview3_diffusers" \
        --dtype "bf16"

Arguments:
    --original_state_dict_repo_id: The Hugging Face repo ID containing the original checkpoint.
    --filename: The filename of the checkpoint in the repo (default: "flux.safetensors").
    --checkpoint_path: Path to a local checkpoint file (alternative to repo_id and filename).
    --transformer: Flag to convert the transformer model.
    --output_path: The path to save the converted model.
    --dtype: The dtype to save the model in (default: "bf16", options: "fp16", "bf16", "fp32").

Note: You must provide either --original_state_dict_repo_id or --checkpoint_path.
"""

import argparse
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from diffusers import CogView3PlusTransformer2DModel
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--filename", default="flux.safetensors", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--transformer", action="store_true")
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str, default="bf16")

args = parser.parse_args()


def load_original_checkpoint(args):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = torch.load(ckpt_path, map_location="cpu")
    return original_state_dict


# this is specific to `AdaLayerNormContinuous`:
# diffusers imnplementation split the linear projection into the scale, shift while CogView3 split it tino shift, scale
def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_cogview3_transformer_checkpoint_to_diffusers(original_state_dict):
    new_state_dict = {}

    # Convert pos_embed
    new_state_dict["pos_embed.proj.weight"] = original_state_dict.pop("mixins.patch_embed.proj.weight")
    new_state_dict["pos_embed.proj.bias"] = original_state_dict.pop("mixins.patch_embed.proj.bias")
    new_state_dict["pos_embed.text_proj.weight"] = original_state_dict.pop("mixins.patch_embed.text_proj.weight")
    new_state_dict["pos_embed.text_proj.bias"] = original_state_dict.pop("mixins.patch_embed.text_proj.bias")

    # Convert time_text_embed
    new_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "time_embed.0.weight"
    )
    new_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop("time_embed.0.bias")
    new_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "time_embed.2.weight"
    )
    new_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop("time_embed.2.bias")
    new_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop("label_emb.0.0.weight")
    new_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop("label_emb.0.0.bias")
    new_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop("label_emb.0.2.weight")
    new_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop("label_emb.0.2.bias")

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

        new_state_dict[block_prefix + "attn.to_q.weight"] = q
        new_state_dict[block_prefix + "attn.to_q.bias"] = q_bias
        new_state_dict[block_prefix + "attn.to_k.weight"] = k
        new_state_dict[block_prefix + "attn.to_k.bias"] = k_bias
        new_state_dict[block_prefix + "attn.to_v.weight"] = v
        new_state_dict[block_prefix + "attn.to_v.bias"] = v_bias

        new_state_dict[block_prefix + "attn.to_out.0.weight"] = original_state_dict.pop(
            old_prefix + "attention.dense.weight"
        )
        new_state_dict[block_prefix + "attn.to_out.0.bias"] = original_state_dict.pop(
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


def main(args):
    original_ckpt = load_original_checkpoint(args)
    original_ckpt = original_ckpt["module"]
    original_ckpt = {k.replace("model.diffusion_model.", ""): v for k, v in original_ckpt.items()}

    original_dtype = next(iter(original_ckpt.values())).dtype
    dtype = None
    if args.dtype is None:
        dtype = original_dtype
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    if args.transformer:
        converted_transformer_state_dict = convert_cogview3_transformer_checkpoint_to_diffusers(original_ckpt)
        transformer = CogView3PlusTransformer2DModel()
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)

        print(f"Saving CogView3 Transformer in Diffusers format in {args.output_path}/transformer")
        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

    if len(original_ckpt) > 0:
        print(f"Warning: {len(original_ckpt)} keys were not converted and will be saved as is: {original_ckpt.keys()}")


if __name__ == "__main__":
    main(args)
