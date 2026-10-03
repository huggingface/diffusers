import argparse
import logging

import torch
from safetensors import safe_open

from diffusers import AutoencoderKLHunyuanImage, AutoencoderKLHunyuanImageRefiner, HunyuanImageTransformer2DModel


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


"""
Usage examples
==============

python scripts/convert_hunyuan_image_to_diffusers.py \
    --model_type hunyuanimage2.1 \
    --transformer_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/dit/hunyuanimage2.1.safetensors" \
    --vae_checkpoint_path "HunyuanImage-2.1/ckpts/vae/vae_2_1/pytorch_model.ckpt" \
    --output_path "/raid/yiyi/test-hy21-diffusers" \
    --dtype fp32

python scripts/convert_hunyuan_image_to_diffusers.py \
    --model_type hunyuanimage2.1-distilled \
    --transformer_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/dit/hunyuanimage2.1-distilled.safetensors" \
    --vae_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/vae/vae_2_1/pytorch_model.ckpt" \
    --output_path "/raid/yiyi/test-hy21-distilled-diffusers" \
    --dtype fp32


python scripts/convert_hunyuan_image_to_diffusers.py \
  --model_type hunyuanimage-refiner \
  --transformer_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/dit/hunyuanimage-refiner.safetensors" \
  --vae_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/vae/vae_refiner/pytorch_model.pt" \
  --output_path "/raid/yiyi/test-hy2-refiner-diffusers" \
  --dtype fp32
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type", type=str, default=None
)  # hunyuanimage2.1, hunyuanimage2.1-distilled, hunyuanimage-refiner
parser.add_argument("--transformer_checkpoint_path", default=None, type=str)  # ckpts/dit/hunyuanimage2.1.safetensors
parser.add_argument("--vae_checkpoint_path", default=None, type=str)  # ckpts/vae/vae_2_1/pytorch_model.ckpt
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str, default="fp32")

args = parser.parse_args()
dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32


# copied from https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/hyimage/models/hunyuan/modules/hunyuanimage_dit.py#L21
def convert_hunyuan_dict_for_tensor_parallel(state_dict):
    """
    Convert a Hunyuan model state dict to be compatible with tensor parallel architectures.

    Args:
        state_dict: Original state dict

    Returns:
        new_dict: Converted state dict
    """
    new_dict = {}
    for k, w in state_dict.items():
        if k.startswith("double_blocks") and "attn_qkv.weight" in k:
            hidden_size = w.shape[1]
            k1 = k.replace("attn_qkv.weight", "attn_q.weight")
            w1 = w[:hidden_size, :]
            new_dict[k1] = w1
            k2 = k.replace("attn_qkv.weight", "attn_k.weight")
            w2 = w[hidden_size : 2 * hidden_size, :]
            new_dict[k2] = w2
            k3 = k.replace("attn_qkv.weight", "attn_v.weight")
            w3 = w[-hidden_size:, :]
            new_dict[k3] = w3
        elif k.startswith("double_blocks") and "attn_qkv.bias" in k:
            hidden_size = w.shape[0] // 3
            k1 = k.replace("attn_qkv.bias", "attn_q.bias")
            w1 = w[:hidden_size]
            new_dict[k1] = w1
            k2 = k.replace("attn_qkv.bias", "attn_k.bias")
            w2 = w[hidden_size : 2 * hidden_size]
            new_dict[k2] = w2
            k3 = k.replace("attn_qkv.bias", "attn_v.bias")
            w3 = w[-hidden_size:]
            new_dict[k3] = w3
        elif k.startswith("single_blocks") and "linear1" in k:
            hidden_size = state_dict[k.replace("linear1", "linear2")].shape[0]
            k1 = k.replace("linear1", "linear1_q")
            w1 = w[:hidden_size]
            new_dict[k1] = w1
            k2 = k.replace("linear1", "linear1_k")
            w2 = w[hidden_size : 2 * hidden_size]
            new_dict[k2] = w2
            k3 = k.replace("linear1", "linear1_v")
            w3 = w[2 * hidden_size : 3 * hidden_size]
            new_dict[k3] = w3
            k4 = k.replace("linear1", "linear1_mlp")
            w4 = w[3 * hidden_size :]
            new_dict[k4] = w4
        elif k.startswith("single_blocks") and "linear2" in k:
            k1 = k.replace("linear2", "linear2.fc")
            new_dict[k1] = w
        else:
            new_dict[k] = w
    return new_dict


def load_original_vae_checkpoint(args):
    # "ckpts/vae/vae_2_1/pytorch_model.ckpt"
    state_dict = torch.load(args.vae_checkpoint_path)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    vae_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_state_dict[k.replace("vae.", "")] = v

    for k, v in vae_state_dict.items():
        if "weight" in k:
            if len(v.shape) == 5 and v.shape[2] == 1:
                vae_state_dict[k] = v.squeeze(2)
            else:
                vae_state_dict[k] = v
        else:
            vae_state_dict[k] = v
    return vae_state_dict


def load_original_refiner_vae_checkpoint(args):
    # "ckpts/vae/vae_refiner/pytorch_model.pt"
    state_dict = torch.load(args.vae_checkpoint_path)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    vae_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_state_dict[k.replace("vae.", "")] = v
    return vae_state_dict


def load_original_transformer_checkpoint(args):
    # ckpts/dit/hunyuanimage-refiner.safetensors"
    # ckpts/dit/hunyuanimage2.1.safetensors"
    state_dict = {}
    with safe_open(args.transformer_checkpoint_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    if args.model_type == "hunyuanimage-2.1":
        state_dict = convert_hunyuan_dict_for_tensor_parallel(state_dict)
    return state_dict


def convert_hunyuan_image_transformer_checkpoint_to_diffusers(
    original_state_dict, use_byt5=True, guidance_distilled=False, use_meanflow=False
):
    converted_state_dict = {}

    # 1. byt5_in -> context_embedder_2
    if use_byt5:
        converted_state_dict["context_embedder_2.norm.weight"] = original_state_dict.pop("byt5_in.layernorm.weight")
        converted_state_dict["context_embedder_2.norm.bias"] = original_state_dict.pop("byt5_in.layernorm.bias")
        converted_state_dict["context_embedder_2.linear_1.weight"] = original_state_dict.pop("byt5_in.fc1.weight")
        converted_state_dict["context_embedder_2.linear_1.bias"] = original_state_dict.pop("byt5_in.fc1.bias")
        converted_state_dict["context_embedder_2.linear_2.weight"] = original_state_dict.pop("byt5_in.fc2.weight")
        converted_state_dict["context_embedder_2.linear_2.bias"] = original_state_dict.pop("byt5_in.fc2.bias")
        converted_state_dict["context_embedder_2.linear_3.weight"] = original_state_dict.pop("byt5_in.fc3.weight")
        converted_state_dict["context_embedder_2.linear_3.bias"] = original_state_dict.pop("byt5_in.fc3.bias")

    # 2. img_in -> x_embedder
    converted_state_dict["x_embedder.proj.weight"] = original_state_dict.pop("img_in.proj.weight")
    converted_state_dict["x_embedder.proj.bias"] = original_state_dict.pop("img_in.proj.bias")

    # 3. txt_in -> context_embedder (complex mapping)
    # txt_in.input_embedder -> context_embedder.proj_in
    converted_state_dict["context_embedder.proj_in.weight"] = original_state_dict.pop("txt_in.input_embedder.weight")
    converted_state_dict["context_embedder.proj_in.bias"] = original_state_dict.pop("txt_in.input_embedder.bias")

    # txt_in.t_embedder -> context_embedder.time_text_embed.timestep_embedder
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_1.weight"] = (
        original_state_dict.pop("txt_in.t_embedder.mlp.0.weight")
    )
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
        "txt_in.t_embedder.mlp.0.bias"
    )
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_2.weight"] = (
        original_state_dict.pop("txt_in.t_embedder.mlp.2.weight")
    )
    converted_state_dict["context_embedder.time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
        "txt_in.t_embedder.mlp.2.bias"
    )

    # txt_in.c_embedder -> context_embedder.time_text_embed.text_embedder
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop(
        "txt_in.c_embedder.linear_1.weight"
    )
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop(
        "txt_in.c_embedder.linear_1.bias"
    )
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop(
        "txt_in.c_embedder.linear_2.weight"
    )
    converted_state_dict["context_embedder.time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop(
        "txt_in.c_embedder.linear_2.bias"
    )

    # txt_in.individual_token_refiner -> context_embedder.token_refiner
    for i in range(2):  # 2 refiner blocks
        block_prefix = f"context_embedder.token_refiner.refiner_blocks.{i}."
        # norm1
        converted_state_dict[f"{block_prefix}norm1.weight"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.norm1.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.bias"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.norm1.bias"
        )
        # norm2
        converted_state_dict[f"{block_prefix}norm2.weight"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.norm2.weight"
        )
        converted_state_dict[f"{block_prefix}norm2.bias"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.norm2.bias"
        )

        # Split QKV
        qkv_weight = original_state_dict.pop(f"txt_in.individual_token_refiner.blocks.{i}.self_attn_qkv.weight")
        qkv_bias = original_state_dict.pop(f"txt_in.individual_token_refiner.blocks.{i}.self_attn_qkv.bias")
        q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)

        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = q_weight
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = q_bias
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = k_weight
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = k_bias
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = v_weight
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = v_bias

        # attn projection
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.self_attn_proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.self_attn_proj.bias"
        )

        # MLP
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.mlp.fc1.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.mlp.fc1.bias"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.mlp.fc2.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.mlp.fc2.bias"
        )

        # norm_out
        converted_state_dict[f"{block_prefix}norm_out.linear.weight"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.adaLN_modulation.1.weight"
        )
        converted_state_dict[f"{block_prefix}norm_out.linear.bias"] = original_state_dict.pop(
            f"txt_in.individual_token_refiner.blocks.{i}.adaLN_modulation.1.bias"
        )

    # 4. time_in -> time_text_embed.timestep_embedder
    converted_state_dict["time_guidance_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "time_in.mlp.0.weight"
    )
    converted_state_dict["time_guidance_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
        "time_in.mlp.0.bias"
    )
    converted_state_dict["time_guidance_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "time_in.mlp.2.weight"
    )
    converted_state_dict["time_guidance_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
        "time_in.mlp.2.bias"
    )

    # time_r_in -> time_guidance_embed.timestep_r_embedder
    if use_meanflow:
        converted_state_dict["time_guidance_embed.timestep_embedder_r.linear_1.weight"] = original_state_dict.pop(
            "time_r_in.mlp.0.weight"
        )
        converted_state_dict["time_guidance_embed.timestep_embedder_r.linear_1.bias"] = original_state_dict.pop(
            "time_r_in.mlp.0.bias"
        )
        converted_state_dict["time_guidance_embed.timestep_embedder_r.linear_2.weight"] = original_state_dict.pop(
            "time_r_in.mlp.2.weight"
        )
        converted_state_dict["time_guidance_embed.timestep_embedder_r.linear_2.bias"] = original_state_dict.pop(
            "time_r_in.mlp.2.bias"
        )

    # guidance_in -> time_guidance_embed.guidance_embedder
    if guidance_distilled:
        converted_state_dict["time_guidance_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop(
            "guidance_in.mlp.0.weight"
        )
        converted_state_dict["time_guidance_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop(
            "guidance_in.mlp.0.bias"
        )
        converted_state_dict["time_guidance_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop(
            "guidance_in.mlp.2.weight"
        )
        converted_state_dict["time_guidance_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop(
            "guidance_in.mlp.2.bias"
        )

    # 5. double_blocks -> transformer_blocks
    for i in range(20):  # 20 double blocks
        block_prefix = f"transformer_blocks.{i}."

        # norm1 (img_mod)
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.linear.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.linear.bias"
        )

        # norm1_context (txt_mod)
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.linear.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.linear.bias"
        )

        # img attention
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_q.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_q.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_k.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_k.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_v.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_v.bias"
        )

        # img attention norms
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_q_norm.weight"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_k_norm.weight"
        )

        # img attention projection
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn_proj.bias"
        )

        # img MLP
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.fc1.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.fc1.bias"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.fc2.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.fc2.bias"
        )

        # txt attention (additional projections)
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_q.weight"
        )
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_q.bias"
        )
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_k.weight"
        )
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_k.bias"
        )
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_v.weight"
        )
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_v.bias"
        )

        # txt attention norms
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_q_norm.weight"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_k_norm.weight"
        )

        # txt attention projection
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn_proj.bias"
        )

        # txt MLP (ff_context)
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.fc1.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.fc1.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.fc2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.fc2.bias"
        )

    # 6. single_blocks -> single_transformer_blocks
    for i in range(40):  # 40 single blocks
        block_prefix = f"single_transformer_blocks.{i}."

        # norm
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.linear.weight"
        )
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.linear.bias"
        )

        # attention Q, K, V
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_q.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_q.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_k.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_k.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_v.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_v.bias"
        )

        # attention norms
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.q_norm.weight"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.k_norm.weight"
        )

        # MLP projection
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_mlp.weight"
        )
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear1_mlp.bias"
        )

        # output projection
        converted_state_dict[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.fc.weight"
        )
        converted_state_dict[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.fc.bias"
        )

    # 7. final_layer -> norm_out + proj_out
    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    shift_w, scale_w = original_state_dict.pop("final_layer.adaLN_modulation.1.weight").chunk(2, dim=0)
    shift_b, scale_b = original_state_dict.pop("final_layer.adaLN_modulation.1.bias").chunk(2, dim=0)
    converted_state_dict["norm_out.linear.weight"] = torch.cat([scale_w, shift_w], dim=0)
    converted_state_dict["norm_out.linear.bias"] = torch.cat([scale_b, shift_b], dim=0)

    return converted_state_dict, original_state_dict


def convert_hunyuan_image_vae_checkpoint_to_diffusers(
    original_state_dict, block_out_channels=[128, 256, 512, 512, 1024, 1024], layers_per_block=2
):
    """Convert original VAE state dict to Diffusers format."""
    converted = {}

    # 1. Encoder
    # 1.1 conv_in
    converted["encoder.conv_in.weight"] = original_state_dict.pop("encoder.conv_in.weight")
    converted["encoder.conv_in.bias"] = original_state_dict.pop("encoder.conv_in.bias")

    # 1.2 down blocks
    diffusers_block_idx = 0

    for block_index in range(len(block_out_channels)):
        for resnet_block_index in range(layers_per_block):
            orig_prefix = f"encoder.down.{block_index}.block.{resnet_block_index}"
            diff_prefix = f"encoder.down_blocks.{diffusers_block_idx}"

            # resnet blocks
            converted[f"{diff_prefix}.norm1.weight"] = original_state_dict.pop(f"{orig_prefix}.norm1.weight")
            converted[f"{diff_prefix}.norm1.bias"] = original_state_dict.pop(f"{orig_prefix}.norm1.bias")
            converted[f"{diff_prefix}.conv1.weight"] = original_state_dict.pop(f"{orig_prefix}.conv1.weight")
            converted[f"{diff_prefix}.conv1.bias"] = original_state_dict.pop(f"{orig_prefix}.conv1.bias")
            converted[f"{diff_prefix}.norm2.weight"] = original_state_dict.pop(f"{orig_prefix}.norm2.weight")
            converted[f"{diff_prefix}.norm2.bias"] = original_state_dict.pop(f"{orig_prefix}.norm2.bias")
            converted[f"{diff_prefix}.conv2.weight"] = original_state_dict.pop(f"{orig_prefix}.conv2.weight")
            converted[f"{diff_prefix}.conv2.bias"] = original_state_dict.pop(f"{orig_prefix}.conv2.bias")

            diffusers_block_idx += 1

        # downsample blocks
        if f"encoder.down.{block_index}.downsample.conv.weight" in original_state_dict:
            converted[f"encoder.down_blocks.{diffusers_block_idx}.conv.weight"] = original_state_dict.pop(
                f"encoder.down.{block_index}.downsample.conv.weight"
            )
            converted[f"encoder.down_blocks.{diffusers_block_idx}.conv.bias"] = original_state_dict.pop(
                f"encoder.down.{block_index}.downsample.conv.bias"
            )
            diffusers_block_idx += 1

    # 1.3 mid block
    converted["encoder.mid_block.resnets.0.norm1.weight"] = original_state_dict.pop("encoder.mid.block_1.norm1.weight")
    converted["encoder.mid_block.resnets.0.norm1.bias"] = original_state_dict.pop("encoder.mid.block_1.norm1.bias")
    converted["encoder.mid_block.resnets.0.conv1.weight"] = original_state_dict.pop("encoder.mid.block_1.conv1.weight")
    converted["encoder.mid_block.resnets.0.conv1.bias"] = original_state_dict.pop("encoder.mid.block_1.conv1.bias")
    converted["encoder.mid_block.resnets.0.norm2.weight"] = original_state_dict.pop("encoder.mid.block_1.norm2.weight")
    converted["encoder.mid_block.resnets.0.norm2.bias"] = original_state_dict.pop("encoder.mid.block_1.norm2.bias")
    converted["encoder.mid_block.resnets.0.conv2.weight"] = original_state_dict.pop("encoder.mid.block_1.conv2.weight")
    converted["encoder.mid_block.resnets.0.conv2.bias"] = original_state_dict.pop("encoder.mid.block_1.conv2.bias")

    converted["encoder.mid_block.resnets.1.norm1.weight"] = original_state_dict.pop("encoder.mid.block_2.norm1.weight")
    converted["encoder.mid_block.resnets.1.norm1.bias"] = original_state_dict.pop("encoder.mid.block_2.norm1.bias")
    converted["encoder.mid_block.resnets.1.conv1.weight"] = original_state_dict.pop("encoder.mid.block_2.conv1.weight")
    converted["encoder.mid_block.resnets.1.conv1.bias"] = original_state_dict.pop("encoder.mid.block_2.conv1.bias")
    converted["encoder.mid_block.resnets.1.norm2.weight"] = original_state_dict.pop("encoder.mid.block_2.norm2.weight")
    converted["encoder.mid_block.resnets.1.norm2.bias"] = original_state_dict.pop("encoder.mid.block_2.norm2.bias")
    converted["encoder.mid_block.resnets.1.conv2.weight"] = original_state_dict.pop("encoder.mid.block_2.conv2.weight")
    converted["encoder.mid_block.resnets.1.conv2.bias"] = original_state_dict.pop("encoder.mid.block_2.conv2.bias")

    converted["encoder.mid_block.attentions.0.norm.weight"] = original_state_dict.pop("encoder.mid.attn_1.norm.weight")
    converted["encoder.mid_block.attentions.0.norm.bias"] = original_state_dict.pop("encoder.mid.attn_1.norm.bias")
    converted["encoder.mid_block.attentions.0.to_q.weight"] = original_state_dict.pop("encoder.mid.attn_1.q.weight")
    converted["encoder.mid_block.attentions.0.to_q.bias"] = original_state_dict.pop("encoder.mid.attn_1.q.bias")
    converted["encoder.mid_block.attentions.0.to_k.weight"] = original_state_dict.pop("encoder.mid.attn_1.k.weight")
    converted["encoder.mid_block.attentions.0.to_k.bias"] = original_state_dict.pop("encoder.mid.attn_1.k.bias")
    converted["encoder.mid_block.attentions.0.to_v.weight"] = original_state_dict.pop("encoder.mid.attn_1.v.weight")
    converted["encoder.mid_block.attentions.0.to_v.bias"] = original_state_dict.pop("encoder.mid.attn_1.v.bias")
    converted["encoder.mid_block.attentions.0.proj.weight"] = original_state_dict.pop(
        "encoder.mid.attn_1.proj_out.weight"
    )
    converted["encoder.mid_block.attentions.0.proj.bias"] = original_state_dict.pop("encoder.mid.attn_1.proj_out.bias")

    # 1.4 encoder output
    converted["encoder.norm_out.weight"] = original_state_dict.pop("encoder.norm_out.weight")
    converted["encoder.norm_out.bias"] = original_state_dict.pop("encoder.norm_out.bias")
    converted["encoder.conv_out.weight"] = original_state_dict.pop("encoder.conv_out.weight")
    converted["encoder.conv_out.bias"] = original_state_dict.pop("encoder.conv_out.bias")

    # 2. Decoder
    # 2.1 conv_in
    converted["decoder.conv_in.weight"] = original_state_dict.pop("decoder.conv_in.weight")
    converted["decoder.conv_in.bias"] = original_state_dict.pop("decoder.conv_in.bias")

    # 2.2 mid block
    converted["decoder.mid_block.resnets.0.norm1.weight"] = original_state_dict.pop("decoder.mid.block_1.norm1.weight")
    converted["decoder.mid_block.resnets.0.norm1.bias"] = original_state_dict.pop("decoder.mid.block_1.norm1.bias")
    converted["decoder.mid_block.resnets.0.conv1.weight"] = original_state_dict.pop("decoder.mid.block_1.conv1.weight")
    converted["decoder.mid_block.resnets.0.conv1.bias"] = original_state_dict.pop("decoder.mid.block_1.conv1.bias")
    converted["decoder.mid_block.resnets.0.norm2.weight"] = original_state_dict.pop("decoder.mid.block_1.norm2.weight")
    converted["decoder.mid_block.resnets.0.norm2.bias"] = original_state_dict.pop("decoder.mid.block_1.norm2.bias")
    converted["decoder.mid_block.resnets.0.conv2.weight"] = original_state_dict.pop("decoder.mid.block_1.conv2.weight")
    converted["decoder.mid_block.resnets.0.conv2.bias"] = original_state_dict.pop("decoder.mid.block_1.conv2.bias")

    converted["decoder.mid_block.resnets.1.norm1.weight"] = original_state_dict.pop("decoder.mid.block_2.norm1.weight")
    converted["decoder.mid_block.resnets.1.norm1.bias"] = original_state_dict.pop("decoder.mid.block_2.norm1.bias")
    converted["decoder.mid_block.resnets.1.conv1.weight"] = original_state_dict.pop("decoder.mid.block_2.conv1.weight")
    converted["decoder.mid_block.resnets.1.conv1.bias"] = original_state_dict.pop("decoder.mid.block_2.conv1.bias")
    converted["decoder.mid_block.resnets.1.norm2.weight"] = original_state_dict.pop("decoder.mid.block_2.norm2.weight")
    converted["decoder.mid_block.resnets.1.norm2.bias"] = original_state_dict.pop("decoder.mid.block_2.norm2.bias")
    converted["decoder.mid_block.resnets.1.conv2.weight"] = original_state_dict.pop("decoder.mid.block_2.conv2.weight")
    converted["decoder.mid_block.resnets.1.conv2.bias"] = original_state_dict.pop("decoder.mid.block_2.conv2.bias")

    converted["decoder.mid_block.attentions.0.norm.weight"] = original_state_dict.pop("decoder.mid.attn_1.norm.weight")
    converted["decoder.mid_block.attentions.0.norm.bias"] = original_state_dict.pop("decoder.mid.attn_1.norm.bias")
    converted["decoder.mid_block.attentions.0.to_q.weight"] = original_state_dict.pop("decoder.mid.attn_1.q.weight")
    converted["decoder.mid_block.attentions.0.to_q.bias"] = original_state_dict.pop("decoder.mid.attn_1.q.bias")
    converted["decoder.mid_block.attentions.0.to_k.weight"] = original_state_dict.pop("decoder.mid.attn_1.k.weight")
    converted["decoder.mid_block.attentions.0.to_k.bias"] = original_state_dict.pop("decoder.mid.attn_1.k.bias")
    converted["decoder.mid_block.attentions.0.to_v.weight"] = original_state_dict.pop("decoder.mid.attn_1.v.weight")
    converted["decoder.mid_block.attentions.0.to_v.bias"] = original_state_dict.pop("decoder.mid.attn_1.v.bias")
    converted["decoder.mid_block.attentions.0.proj.weight"] = original_state_dict.pop(
        "decoder.mid.attn_1.proj_out.weight"
    )
    converted["decoder.mid_block.attentions.0.proj.bias"] = original_state_dict.pop("decoder.mid.attn_1.proj_out.bias")

    # 2.3 up blocks
    diffusers_block_idx = 0
    for up_block_index in range(len(block_out_channels)):
        # resnet blocks
        for resnet_block_index in range(layers_per_block + 1):
            orig_prefix = f"decoder.up.{up_block_index}.block.{resnet_block_index}"
            diff_prefix = f"decoder.up_blocks.{diffusers_block_idx}"

            converted[f"{diff_prefix}.norm1.weight"] = original_state_dict.pop(f"{orig_prefix}.norm1.weight")
            converted[f"{diff_prefix}.norm1.bias"] = original_state_dict.pop(f"{orig_prefix}.norm1.bias")
            converted[f"{diff_prefix}.conv1.weight"] = original_state_dict.pop(f"{orig_prefix}.conv1.weight")
            converted[f"{diff_prefix}.conv1.bias"] = original_state_dict.pop(f"{orig_prefix}.conv1.bias")
            converted[f"{diff_prefix}.norm2.weight"] = original_state_dict.pop(f"{orig_prefix}.norm2.weight")
            converted[f"{diff_prefix}.norm2.bias"] = original_state_dict.pop(f"{orig_prefix}.norm2.bias")
            converted[f"{diff_prefix}.conv2.weight"] = original_state_dict.pop(f"{orig_prefix}.conv2.weight")
            converted[f"{diff_prefix}.conv2.bias"] = original_state_dict.pop(f"{orig_prefix}.conv2.bias")

            diffusers_block_idx += 1

        # upsample blocks
        if f"decoder.up.{up_block_index}.upsample.conv.weight" in original_state_dict:
            converted[f"decoder.up_blocks.{diffusers_block_idx}.conv.weight"] = original_state_dict.pop(
                f"decoder.up.{up_block_index}.upsample.conv.weight"
            )
            converted[f"decoder.up_blocks.{diffusers_block_idx}.conv.bias"] = original_state_dict.pop(
                f"decoder.up.{up_block_index}.upsample.conv.bias"
            )
            diffusers_block_idx += 1

    # 2.4 decoder output
    converted["decoder.norm_out.weight"] = original_state_dict.pop("decoder.norm_out.weight")
    converted["decoder.norm_out.bias"] = original_state_dict.pop("decoder.norm_out.bias")
    converted["decoder.conv_out.weight"] = original_state_dict.pop("decoder.conv_out.weight")
    converted["decoder.conv_out.bias"] = original_state_dict.pop("decoder.conv_out.bias")

    return converted, original_state_dict


def convert_hunyuan_image_refiner_vae_checkpoint_to_diffusers(
    original_state_dict, block_out_channels=[128, 256, 512, 1024, 1024], layers_per_block=2
):
    converted = {}

    # 1. Encoder
    # 1.1 conv_in
    converted["encoder.conv_in.conv.weight"] = original_state_dict.pop("encoder.conv_in.conv.weight")
    converted["encoder.conv_in.conv.bias"] = original_state_dict.pop("encoder.conv_in.conv.bias")

    # 1.2 Down blocks
    for down_block_index in range(len(block_out_channels)):  # 0 to 4
        # ResNet blocks
        for resnet_block_index in range(layers_per_block):  # 0 to 1
            converted[f"encoder.down_blocks.{down_block_index}.resnets.{resnet_block_index}.norm1.gamma"] = (
                original_state_dict.pop(f"encoder.down.{down_block_index}.block.{resnet_block_index}.norm1.gamma")
            )
            converted[f"encoder.down_blocks.{down_block_index}.resnets.{resnet_block_index}.conv1.conv.weight"] = (
                original_state_dict.pop(
                    f"encoder.down.{down_block_index}.block.{resnet_block_index}.conv1.conv.weight"
                )
            )
            converted[f"encoder.down_blocks.{down_block_index}.resnets.{resnet_block_index}.conv1.conv.bias"] = (
                original_state_dict.pop(f"encoder.down.{down_block_index}.block.{resnet_block_index}.conv1.conv.bias")
            )
            converted[f"encoder.down_blocks.{down_block_index}.resnets.{resnet_block_index}.norm2.gamma"] = (
                original_state_dict.pop(f"encoder.down.{down_block_index}.block.{resnet_block_index}.norm2.gamma")
            )
            converted[f"encoder.down_blocks.{down_block_index}.resnets.{resnet_block_index}.conv2.conv.weight"] = (
                original_state_dict.pop(
                    f"encoder.down.{down_block_index}.block.{resnet_block_index}.conv2.conv.weight"
                )
            )
            converted[f"encoder.down_blocks.{down_block_index}.resnets.{resnet_block_index}.conv2.conv.bias"] = (
                original_state_dict.pop(f"encoder.down.{down_block_index}.block.{resnet_block_index}.conv2.conv.bias")
            )

        # Downsample (if exists)
        if f"encoder.down.{down_block_index}.downsample.conv.conv.weight" in original_state_dict:
            converted[f"encoder.down_blocks.{down_block_index}.downsamplers.0.conv.conv.weight"] = (
                original_state_dict.pop(f"encoder.down.{down_block_index}.downsample.conv.conv.weight")
            )
            converted[f"encoder.down_blocks.{down_block_index}.downsamplers.0.conv.conv.bias"] = (
                original_state_dict.pop(f"encoder.down.{down_block_index}.downsample.conv.conv.bias")
            )

    # 1.3 Mid block
    converted["encoder.mid_block.resnets.0.norm1.gamma"] = original_state_dict.pop("encoder.mid.block_1.norm1.gamma")
    converted["encoder.mid_block.resnets.0.conv1.conv.weight"] = original_state_dict.pop(
        "encoder.mid.block_1.conv1.conv.weight"
    )
    converted["encoder.mid_block.resnets.0.conv1.conv.bias"] = original_state_dict.pop(
        "encoder.mid.block_1.conv1.conv.bias"
    )
    converted["encoder.mid_block.resnets.0.norm2.gamma"] = original_state_dict.pop("encoder.mid.block_1.norm2.gamma")
    converted["encoder.mid_block.resnets.0.conv2.conv.weight"] = original_state_dict.pop(
        "encoder.mid.block_1.conv2.conv.weight"
    )
    converted["encoder.mid_block.resnets.0.conv2.conv.bias"] = original_state_dict.pop(
        "encoder.mid.block_1.conv2.conv.bias"
    )

    converted["encoder.mid_block.resnets.1.norm1.gamma"] = original_state_dict.pop("encoder.mid.block_2.norm1.gamma")
    converted["encoder.mid_block.resnets.1.conv1.conv.weight"] = original_state_dict.pop(
        "encoder.mid.block_2.conv1.conv.weight"
    )
    converted["encoder.mid_block.resnets.1.conv1.conv.bias"] = original_state_dict.pop(
        "encoder.mid.block_2.conv1.conv.bias"
    )
    converted["encoder.mid_block.resnets.1.norm2.gamma"] = original_state_dict.pop("encoder.mid.block_2.norm2.gamma")
    converted["encoder.mid_block.resnets.1.conv2.conv.weight"] = original_state_dict.pop(
        "encoder.mid.block_2.conv2.conv.weight"
    )
    converted["encoder.mid_block.resnets.1.conv2.conv.bias"] = original_state_dict.pop(
        "encoder.mid.block_2.conv2.conv.bias"
    )

    # Attention block
    converted["encoder.mid_block.attentions.0.norm.gamma"] = original_state_dict.pop("encoder.mid.attn_1.norm.gamma")
    converted["encoder.mid_block.attentions.0.to_q.weight"] = original_state_dict.pop("encoder.mid.attn_1.q.weight")
    converted["encoder.mid_block.attentions.0.to_q.bias"] = original_state_dict.pop("encoder.mid.attn_1.q.bias")
    converted["encoder.mid_block.attentions.0.to_k.weight"] = original_state_dict.pop("encoder.mid.attn_1.k.weight")
    converted["encoder.mid_block.attentions.0.to_k.bias"] = original_state_dict.pop("encoder.mid.attn_1.k.bias")
    converted["encoder.mid_block.attentions.0.to_v.weight"] = original_state_dict.pop("encoder.mid.attn_1.v.weight")
    converted["encoder.mid_block.attentions.0.to_v.bias"] = original_state_dict.pop("encoder.mid.attn_1.v.bias")
    converted["encoder.mid_block.attentions.0.proj_out.weight"] = original_state_dict.pop(
        "encoder.mid.attn_1.proj_out.weight"
    )
    converted["encoder.mid_block.attentions.0.proj_out.bias"] = original_state_dict.pop(
        "encoder.mid.attn_1.proj_out.bias"
    )

    # 1.4 Encoder output
    converted["encoder.norm_out.gamma"] = original_state_dict.pop("encoder.norm_out.gamma")
    converted["encoder.conv_out.conv.weight"] = original_state_dict.pop("encoder.conv_out.conv.weight")
    converted["encoder.conv_out.conv.bias"] = original_state_dict.pop("encoder.conv_out.conv.bias")

    # 2. Decoder
    # 2.1 conv_in
    converted["decoder.conv_in.conv.weight"] = original_state_dict.pop("decoder.conv_in.conv.weight")
    converted["decoder.conv_in.conv.bias"] = original_state_dict.pop("decoder.conv_in.conv.bias")

    # 2.2 Mid block
    converted["decoder.mid_block.resnets.0.norm1.gamma"] = original_state_dict.pop("decoder.mid.block_1.norm1.gamma")
    converted["decoder.mid_block.resnets.0.conv1.conv.weight"] = original_state_dict.pop(
        "decoder.mid.block_1.conv1.conv.weight"
    )
    converted["decoder.mid_block.resnets.0.conv1.conv.bias"] = original_state_dict.pop(
        "decoder.mid.block_1.conv1.conv.bias"
    )
    converted["decoder.mid_block.resnets.0.norm2.gamma"] = original_state_dict.pop("decoder.mid.block_1.norm2.gamma")
    converted["decoder.mid_block.resnets.0.conv2.conv.weight"] = original_state_dict.pop(
        "decoder.mid.block_1.conv2.conv.weight"
    )
    converted["decoder.mid_block.resnets.0.conv2.conv.bias"] = original_state_dict.pop(
        "decoder.mid.block_1.conv2.conv.bias"
    )

    converted["decoder.mid_block.resnets.1.norm1.gamma"] = original_state_dict.pop("decoder.mid.block_2.norm1.gamma")
    converted["decoder.mid_block.resnets.1.conv1.conv.weight"] = original_state_dict.pop(
        "decoder.mid.block_2.conv1.conv.weight"
    )
    converted["decoder.mid_block.resnets.1.conv1.conv.bias"] = original_state_dict.pop(
        "decoder.mid.block_2.conv1.conv.bias"
    )
    converted["decoder.mid_block.resnets.1.norm2.gamma"] = original_state_dict.pop("decoder.mid.block_2.norm2.gamma")
    converted["decoder.mid_block.resnets.1.conv2.conv.weight"] = original_state_dict.pop(
        "decoder.mid.block_2.conv2.conv.weight"
    )
    converted["decoder.mid_block.resnets.1.conv2.conv.bias"] = original_state_dict.pop(
        "decoder.mid.block_2.conv2.conv.bias"
    )

    # Decoder attention block
    converted["decoder.mid_block.attentions.0.norm.gamma"] = original_state_dict.pop("decoder.mid.attn_1.norm.gamma")
    converted["decoder.mid_block.attentions.0.to_q.weight"] = original_state_dict.pop("decoder.mid.attn_1.q.weight")
    converted["decoder.mid_block.attentions.0.to_q.bias"] = original_state_dict.pop("decoder.mid.attn_1.q.bias")
    converted["decoder.mid_block.attentions.0.to_k.weight"] = original_state_dict.pop("decoder.mid.attn_1.k.weight")
    converted["decoder.mid_block.attentions.0.to_k.bias"] = original_state_dict.pop("decoder.mid.attn_1.k.bias")
    converted["decoder.mid_block.attentions.0.to_v.weight"] = original_state_dict.pop("decoder.mid.attn_1.v.weight")
    converted["decoder.mid_block.attentions.0.to_v.bias"] = original_state_dict.pop("decoder.mid.attn_1.v.bias")
    converted["decoder.mid_block.attentions.0.proj_out.weight"] = original_state_dict.pop(
        "decoder.mid.attn_1.proj_out.weight"
    )
    converted["decoder.mid_block.attentions.0.proj_out.bias"] = original_state_dict.pop(
        "decoder.mid.attn_1.proj_out.bias"
    )

    # 2.3 Up blocks
    for up_block_index in range(len(block_out_channels)):  # 0 to 5
        # ResNet blocks
        for resnet_block_index in range(layers_per_block + 1):  # 0 to 2 (decoder has 3 resnets per level)
            converted[f"decoder.up_blocks.{up_block_index}.resnets.{resnet_block_index}.norm1.gamma"] = (
                original_state_dict.pop(f"decoder.up.{up_block_index}.block.{resnet_block_index}.norm1.gamma")
            )
            converted[f"decoder.up_blocks.{up_block_index}.resnets.{resnet_block_index}.conv1.conv.weight"] = (
                original_state_dict.pop(f"decoder.up.{up_block_index}.block.{resnet_block_index}.conv1.conv.weight")
            )
            converted[f"decoder.up_blocks.{up_block_index}.resnets.{resnet_block_index}.conv1.conv.bias"] = (
                original_state_dict.pop(f"decoder.up.{up_block_index}.block.{resnet_block_index}.conv1.conv.bias")
            )
            converted[f"decoder.up_blocks.{up_block_index}.resnets.{resnet_block_index}.norm2.gamma"] = (
                original_state_dict.pop(f"decoder.up.{up_block_index}.block.{resnet_block_index}.norm2.gamma")
            )
            converted[f"decoder.up_blocks.{up_block_index}.resnets.{resnet_block_index}.conv2.conv.weight"] = (
                original_state_dict.pop(f"decoder.up.{up_block_index}.block.{resnet_block_index}.conv2.conv.weight")
            )
            converted[f"decoder.up_blocks.{up_block_index}.resnets.{resnet_block_index}.conv2.conv.bias"] = (
                original_state_dict.pop(f"decoder.up.{up_block_index}.block.{resnet_block_index}.conv2.conv.bias")
            )

        # Upsample (if exists)
        if f"decoder.up.{up_block_index}.upsample.conv.conv.weight" in original_state_dict:
            converted[f"decoder.up_blocks.{up_block_index}.upsamplers.0.conv.conv.weight"] = original_state_dict.pop(
                f"decoder.up.{up_block_index}.upsample.conv.conv.weight"
            )
            converted[f"decoder.up_blocks.{up_block_index}.upsamplers.0.conv.conv.bias"] = original_state_dict.pop(
                f"decoder.up.{up_block_index}.upsample.conv.conv.bias"
            )

    # 2.4 Decoder output
    converted["decoder.norm_out.gamma"] = original_state_dict.pop("decoder.norm_out.gamma")
    converted["decoder.conv_out.conv.weight"] = original_state_dict.pop("decoder.conv_out.conv.weight")
    converted["decoder.conv_out.conv.bias"] = original_state_dict.pop("decoder.conv_out.conv.bias")

    return converted, original_state_dict


def main(args):
    if args.model_type == "hunyuanimage2.1":
        original_transformer_state_dict = load_original_transformer_checkpoint(args)
        original_vae_state_dict = load_original_vae_checkpoint(args)

        transformer_config = {
            "in_channels": 64,
            "out_channels": 64,
            "num_attention_heads": 28,
            "attention_head_dim": 128,
            "num_layers": 20,
            "num_single_layers": 40,
            "num_refiner_layers": 2,
            "patch_size": (1, 1),
            "qk_norm": "rms_norm",
            "guidance_embeds": False,
            "text_embed_dim": 3584,
            "text_embed_2_dim": 1472,
            "rope_theta": 256.0,
            "rope_axes_dim": (64, 64),
        }

        converted_transformer_state_dict, original_transformer_state_dict = (
            convert_hunyuan_image_transformer_checkpoint_to_diffusers(
                original_transformer_state_dict, use_byt5=True, guidance_distilled=False
            )
        )

        if original_transformer_state_dict:
            logger.warning(
                f"Unused {len(original_transformer_state_dict)} original keys for transformer: {list(original_transformer_state_dict.keys())}"
            )

        transformer = HunyuanImageTransformer2DModel(**transformer_config)
        missing_keys, unexpected_key = transformer.load_state_dict(converted_transformer_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for transformer: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for transformer: {unexpected_key}")

        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

        vae_config_diffusers = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 64,
            "block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "layers_per_block": 2,
            "spatial_compression_ratio": 32,
            "sample_size": 384,
            "scaling_factor": 0.75289,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
        }
        converted_vae_state_dict, original_vae_state_dict = convert_hunyuan_image_vae_checkpoint_to_diffusers(
            original_vae_state_dict, block_out_channels=[128, 256, 512, 512, 1024, 1024], layers_per_block=2
        )
        if original_vae_state_dict:
            logger.warning(
                f"Unused {len(original_vae_state_dict)} original keys for vae: {list(original_vae_state_dict.keys())}"
            )

        vae = AutoencoderKLHunyuanImage(**vae_config_diffusers)
        missing_keys, unexpected_key = vae.load_state_dict(converted_vae_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for vae: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for vae: {unexpected_key}")

        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")

    elif args.model_type == "hunyuanimage2.1-distilled":
        original_transformer_state_dict = load_original_transformer_checkpoint(args)
        original_vae_state_dict = load_original_vae_checkpoint(args)

        transformer_config = {
            "in_channels": 64,
            "out_channels": 64,
            "num_attention_heads": 28,
            "attention_head_dim": 128,
            "num_layers": 20,
            "num_single_layers": 40,
            "num_refiner_layers": 2,
            "patch_size": (1, 1),
            "qk_norm": "rms_norm",
            "guidance_embeds": True,
            "text_embed_dim": 3584,
            "text_embed_2_dim": 1472,
            "rope_theta": 256.0,
            "rope_axes_dim": (64, 64),
            "use_meanflow": True,
        }

        converted_transformer_state_dict, original_transformer_state_dict = (
            convert_hunyuan_image_transformer_checkpoint_to_diffusers(
                original_transformer_state_dict, use_byt5=True, guidance_distilled=True, use_meanflow=True
            )
        )

        if original_transformer_state_dict:
            logger.warning(
                f"Unused {len(original_transformer_state_dict)} original keys for transformer: {list(original_transformer_state_dict.keys())}"
            )

        transformer = HunyuanImageTransformer2DModel(**transformer_config)
        missing_keys, unexpected_key = transformer.load_state_dict(converted_transformer_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for transformer: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for transformer: {unexpected_key}")

        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

        vae_config_diffusers = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 64,
            "block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "layers_per_block": 2,
            "spatial_compression_ratio": 32,
            "sample_size": 384,
            "scaling_factor": 0.75289,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
        }
        converted_vae_state_dict, original_vae_state_dict = convert_hunyuan_image_vae_checkpoint_to_diffusers(
            original_vae_state_dict, block_out_channels=[128, 256, 512, 512, 1024, 1024], layers_per_block=2
        )
        if original_vae_state_dict:
            logger.warning(
                f"Unused {len(original_vae_state_dict)} original keys for vae: {list(original_vae_state_dict.keys())}"
            )

        vae = AutoencoderKLHunyuanImage(**vae_config_diffusers)
        missing_keys, unexpected_key = vae.load_state_dict(converted_vae_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for vae: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for vae: {unexpected_key}")

        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")

    elif args.model_type == "hunyuanimage-refiner":
        original_transformer_state_dict = load_original_transformer_checkpoint(args)
        original_vae_state_dict = load_original_refiner_vae_checkpoint(args)

        transformer_config = {
            "in_channels": 128,
            "out_channels": 64,
            "num_layers": 20,
            "num_single_layers": 40,
            "rope_axes_dim": [16, 56, 56],
            "num_attention_heads": 26,
            "attention_head_dim": 128,
            "mlp_ratio": 4,
            "patch_size": (1, 1, 1),
            "text_embed_dim": 3584,
            "guidance_embeds": True,
        }
        converted_transformer_state_dict, original_transformer_state_dict = (
            convert_hunyuan_image_transformer_checkpoint_to_diffusers(
                original_transformer_state_dict, use_byt5=False, guidance_distilled=True
            )
        )
        if original_transformer_state_dict:
            logger.warning(
                f"Unused {len(original_transformer_state_dict)} original keys for transformer: {list(original_transformer_state_dict.keys())}"
            )

        transformer = HunyuanImageTransformer2DModel(**transformer_config)
        missing_keys, unexpected_key = transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        if missing_keys:
            logger.warning(f"Missing keys for transformer: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for transformer: {unexpected_key}")

        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

        vae = AutoencoderKLHunyuanImageRefiner()

        converted_vae_state_dict, original_vae_state_dict = convert_hunyuan_image_refiner_vae_checkpoint_to_diffusers(
            original_vae_state_dict
        )
        if original_vae_state_dict:
            logger.warning(
                f"Unused {len(original_vae_state_dict)} original keys for vae: {list(original_vae_state_dict.keys())}"
            )

        missing_keys, unexpected_key = vae.load_state_dict(converted_vae_state_dict, strict=True)
        logger.warning(f"Missing keys for vae: {missing_keys}")
        logger.warning(f"Unexpected keys for vae: {unexpected_key}")

        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")


if __name__ == "__main__":
    main(args)
