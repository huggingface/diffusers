"""
python scripts/convert_hunyuan_video1_5_to_diffusers.py \
    --original_state_dict_folder /raid/yiyi/new-model-vid \
    --output_path /raid/yiyi/hunyuanvideo15-480p_i2v-diffusers \
    --transformer_type 480p_i2v \
    --dtype fp32
"""

import argparse
from typing import Any, Dict

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

import pathlib  
from diffusers import HunyuanVideo15Transformer3DModel

TRANSFORMER_CONFIGS = {
    "480p_i2v": {
        "in_channels": 65,
        "out_channels": 32,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 54,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "text_embed_dim": 3584,
        "text_embed_2_dim": 1472,
        "image_embed_dim": 1152,
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
        "use_meanflow": False,
    },
}

def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_hyvideo15_transformer_to_diffusers(original_state_dict):
    """
    Convert HunyuanVideo 1.5 original checkpoint to Diffusers format.
    """
    converted_state_dict = {}

    # 1. time_embed.timestep_embedder <- time_in
    converted_state_dict["time_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "time_in.mlp.0.weight"
    )
    converted_state_dict["time_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
        "time_in.mlp.0.bias"
    )
    converted_state_dict["time_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "time_in.mlp.2.weight"
    )
    converted_state_dict["time_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
        "time_in.mlp.2.bias"
    )

    # 2. context_embedder.time_text_embed.timestep_embedder <- txt_in.t_embedder
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_1.weight"] = (
        original_state_dict.pop("txt_in.t_embedder.mlp.0.weight")
    )
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_1.bias"] = (
        original_state_dict.pop("txt_in.t_embedder.mlp.0.bias")
    )
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_2.weight"] = (
        original_state_dict.pop("txt_in.t_embedder.mlp.2.weight")
    )
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_2.bias"] = (
        original_state_dict.pop("txt_in.t_embedder.mlp.2.bias")
    )

    # 3. context_embedder.time_text_embed.text_embedder <- txt_in.c_embedder
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_1.weight"] = (
        original_state_dict.pop("txt_in.c_embedder.linear_1.weight")
    )
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_1.bias"] = (
        original_state_dict.pop("txt_in.c_embedder.linear_1.bias")
    )
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_2.weight"] = (
        original_state_dict.pop("txt_in.c_embedder.linear_2.weight")
    )
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_2.bias"] = (
        original_state_dict.pop("txt_in.c_embedder.linear_2.bias")
    )

    # 4. context_embedder.proj_in <- txt_in.input_embedder
    converted_state_dict["context_embedder.proj_in.weight"] = original_state_dict.pop(
        "txt_in.input_embedder.weight"
    )
    converted_state_dict["context_embedder.proj_in.bias"] = original_state_dict.pop("txt_in.input_embedder.bias")

    # 5. context_embedder.token_refiner <- txt_in.individual_token_refiner
    num_refiner_blocks = 2
    for i in range(num_refiner_blocks):
        block_prefix = f"context_embedder.token_refiner.refiner_blocks.{i}."
        orig_prefix = f"txt_in.individual_token_refiner.blocks.{i}."

        # norm1
        converted_state_dict[f"{block_prefix}norm1.weight"] = original_state_dict.pop(f"{orig_prefix}norm1.weight")
        converted_state_dict[f"{block_prefix}norm1.bias"] = original_state_dict.pop(f"{orig_prefix}norm1.bias")

        # Split self_attn_qkv into to_q, to_k, to_v
        qkv_weight = original_state_dict.pop(f"{orig_prefix}self_attn_qkv.weight")
        qkv_bias = original_state_dict.pop(f"{orig_prefix}self_attn_qkv.bias")
        q, k, v = torch.chunk(qkv_weight, 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)

        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = q
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = q_bias
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = k
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = k_bias
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = v
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = v_bias

        # self_attn_proj -> attn.to_out.0
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
            f"{orig_prefix}self_attn_proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
            f"{orig_prefix}self_attn_proj.bias"
        )

        # norm2
        converted_state_dict[f"{block_prefix}norm2.weight"] = original_state_dict.pop(f"{orig_prefix}norm2.weight")
        converted_state_dict[f"{block_prefix}norm2.bias"] = original_state_dict.pop(f"{orig_prefix}norm2.bias")

        # mlp -> ff
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
            f"{orig_prefix}mlp.fc1.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
            f"{orig_prefix}mlp.fc1.bias"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
            f"{orig_prefix}mlp.fc2.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(f"{orig_prefix}mlp.fc2.bias")

        # adaLN_modulation -> norm_out
        converted_state_dict[f"{block_prefix}norm_out.linear.weight"] = original_state_dict.pop(
            f"{orig_prefix}adaLN_modulation.1.weight"
        )
        converted_state_dict[f"{block_prefix}norm_out.linear.bias"] = original_state_dict.pop(
            f"{orig_prefix}adaLN_modulation.1.bias"
        )

    # 6. context_embedder_2 <- byt5_in
    converted_state_dict["context_embedder_2.norm.weight"] = original_state_dict.pop("byt5_in.layernorm.weight")
    converted_state_dict["context_embedder_2.norm.bias"] = original_state_dict.pop("byt5_in.layernorm.bias")
    converted_state_dict["context_embedder_2.linear_1.weight"] = original_state_dict.pop("byt5_in.fc1.weight")
    converted_state_dict["context_embedder_2.linear_1.bias"] = original_state_dict.pop("byt5_in.fc1.bias")
    converted_state_dict["context_embedder_2.linear_2.weight"] = original_state_dict.pop("byt5_in.fc2.weight")
    converted_state_dict["context_embedder_2.linear_2.bias"] = original_state_dict.pop("byt5_in.fc2.bias")
    converted_state_dict["context_embedder_2.linear_3.weight"] = original_state_dict.pop("byt5_in.fc3.weight")
    converted_state_dict["context_embedder_2.linear_3.bias"] = original_state_dict.pop("byt5_in.fc3.bias")

    # 7. image_embedder <- vision_in
    converted_state_dict["image_embedder.norm_in.weight"] = original_state_dict.pop("vision_in.proj.0.weight")
    converted_state_dict["image_embedder.norm_in.bias"] = original_state_dict.pop("vision_in.proj.0.bias")
    converted_state_dict["image_embedder.linear_1.weight"] = original_state_dict.pop("vision_in.proj.1.weight")
    converted_state_dict["image_embedder.linear_1.bias"] = original_state_dict.pop("vision_in.proj.1.bias")
    converted_state_dict["image_embedder.linear_2.weight"] = original_state_dict.pop("vision_in.proj.3.weight")
    converted_state_dict["image_embedder.linear_2.bias"] = original_state_dict.pop("vision_in.proj.3.bias")
    converted_state_dict["image_embedder.norm_out.weight"] = original_state_dict.pop("vision_in.proj.4.weight")
    converted_state_dict["image_embedder.norm_out.bias"] = original_state_dict.pop("vision_in.proj.4.bias")

    # 8. x_embedder <- img_in
    converted_state_dict["x_embedder.proj.weight"] = original_state_dict.pop("img_in.proj.weight")
    converted_state_dict["x_embedder.proj.bias"] = original_state_dict.pop("img_in.proj.bias")

    # 9. cond_type_embed <- cond_type_embedding
    converted_state_dict["cond_type_embed.weight"] = original_state_dict.pop("cond_type_embedding.weight")

    # 10. transformer_blocks <- double_blocks
    num_layers = 54
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        orig_prefix = f"double_blocks.{i}."

        # norm1 (img_mod)
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_mod.linear.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
            f"{orig_prefix}img_mod.linear.bias"
        )

        # norm1_context (txt_mod)
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_mod.linear.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
            f"{orig_prefix}txt_mod.linear.bias"
        )

        # img attention (to_q, to_k, to_v)
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_q.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_q.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_k.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_k.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_v.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_v.bias"
        )

        # img attention qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_q_norm.weight"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_k_norm.weight"
        )

        # img attention output projection
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
            f"{orig_prefix}img_attn_proj.bias"
        )

        # txt attention (add_q_proj, add_k_proj, add_v_proj)
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_q.weight"
        )
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_q.bias"
        )
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_k.weight"
        )
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_k.bias"
        )
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_v.weight"
        )
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_v.bias"
        )

        # txt attention qk norm
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_q_norm.weight"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_k_norm.weight"
        )

        # txt attention output projection
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
            f"{orig_prefix}txt_attn_proj.bias"
        )

        # norm2 and norm2_context (these don't have weights in the original, they're LayerNorm with elementwise_affine=False)
        # So we skip them

        # img_mlp -> ff
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_mlp.fc1.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
            f"{orig_prefix}img_mlp.fc1.bias"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
            f"{orig_prefix}img_mlp.fc2.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
            f"{orig_prefix}img_mlp.fc2.bias"
        )

        # txt_mlp -> ff_context
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_mlp.fc1.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
            f"{orig_prefix}txt_mlp.fc1.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
            f"{orig_prefix}txt_mlp.fc2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
            f"{orig_prefix}txt_mlp.fc2.bias"
        )

    # 11. norm_out and proj_out <- final_layer
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(original_state_dict.pop(
        "final_layer.adaLN_modulation.1.weight"
    ))
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(original_state_dict.pop("final_layer.adaLN_modulation.1.bias"))
    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")

    return converted_state_dict


def load_sharded_safetensors(dir: pathlib.Path):
    file_paths = list(dir.glob("diffusion_pytorch_model*.safetensors"))
    state_dict = {}
    for path in file_paths:
        state_dict.update(load_file(path))
    return state_dict


def load_original_state_dict(args):
    if args.original_state_dict_repo_id is not None:
        model_dir = snapshot_download(
            args.original_state_dict_repo_id, 
            repo_type="model",
            allow_patterns="transformer/" + args.transformer_type + "/*"
        )
    elif args.original_state_dict_folder is not None:
        model_dir = pathlib.Path(args.original_state_dict_folder)
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or `original_state_dict_folder`")
    model_dir = pathlib.Path(model_dir)
    model_dir = model_dir / "transformer" / args.transformer_type
    return load_sharded_safetensors(model_dir)

def convert_transformer(args):
    original_state_dict = load_original_state_dict(args)

    config = TRANSFORMER_CONFIGS[args.transformer_type]
    with init_empty_weights():
        transformer = HunyuanVideo15Transformer3DModel(**config)
    state_dict = convert_hyvideo15_transformer_to_diffusers(original_state_dict)
    transformer.load_state_dict(state_dict, strict=True, assign=True)

    return transformer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_state_dict_repo_id", type=str, default=None, help="Path to original hub_id for the model"
    )
    parser.add_argument("--original_state_dict_folder", type=str, default=None, help="Folder name of the original state dict")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the transformer in.")
    parser.add_argument(
        "--transformer_type", type=str, default="480p_i2v", choices=list(TRANSFORMER_CONFIGS.keys())
    )
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

    transformer = convert_transformer(args)
    transformer = transformer.to(dtype=dtype)
    transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
