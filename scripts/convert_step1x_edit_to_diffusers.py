import argparse
from contextlib import nullcontext

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from diffusers import AutoencoderKL, FluxTransformer2DModel, Step1XEditTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from diffusers.utils.import_utils import is_accelerate_available


"""
# Transformer

python scripts/convert_step1x_edit_to_diffusers.py  \
--checkpoint_path "/mnt/lib/Step1X-Edit/step1x-edit-v1p1-official.safetensors" \
--output_path "/mnt/lib/Step1X-Edit-diffusers" \
--transformer

"""

"""
# VAE

python scripts/convert_step1x_edit_to_diffusers.py  \
--checkpoint_path "/mnt/lib/FLUX.1-dev/ae.safetensors" \
--output_path "/mnt/lib/Step1X-Edit-diffusers" \
--dtype "fp32" \
--vae

"""

"""
# LLM Encoder

python scripts/convert_step1x_edit_to_diffusers.py  \
--original_state_dict_repo_id "/mnt/lib/Step1X-Edit/Qwen2.5-VL-7B-Instruct" \
--output_path "/mnt/lib/Step1X-Edit-diffusers" \
--text_encoder
"""

"""
# Scheduler

python scripts/convert_step1x_edit_to_diffusers.py  \
--original_state_dict_repo_id "/mnt/lib/FLUX.1-dev" \
--output_path "/mnt/lib/Step1X-Edit-diffusers" \
--scheduler
"""

CTX = init_empty_weights if is_accelerate_available() else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--filename", default="flux.safetensors", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--in_channels", type=int, default=64)
parser.add_argument("--out_channels", type=int, default=64)
parser.add_argument("--vae", action="store_true")
parser.add_argument("--text_encoder", action="store_true")
parser.add_argument("--transformer", action="store_true")
parser.add_argument("--scheduler", action="store_true")
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str, default="bf16")

args = parser.parse_args()
dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32


def load_original_checkpoint(args):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError(" please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict

# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_step1x_edit_transformer_checkpoint_to_diffusers(
    original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio=4.0
):
    converted_state_dict = {}

    ## time_embed <-  time_in
    converted_state_dict["time_embed.in_layer.weight"] = original_state_dict.pop(
        "time_in.in_layer.weight"
    )
    converted_state_dict["time_embed.in_layer.bias"] = original_state_dict.pop(
        "time_in.in_layer.bias"
    )
    converted_state_dict["time_embed.out_layer.weight"] = original_state_dict.pop(
        "time_in.out_layer.weight"
    )
    converted_state_dict["time_embed.out_layer.bias"] = original_state_dict.pop(
        "time_in.out_layer.bias"
    )

    ## vec_embed <- vector_in
    converted_state_dict["vec_embed.in_layer.weight"] = original_state_dict.pop(
        "vector_in.in_layer.weight"
    )
    converted_state_dict["vec_embed.in_layer.bias"] = original_state_dict.pop(
        "vector_in.in_layer.bias"
    )
    converted_state_dict["vec_embed.out_layer.weight"] = original_state_dict.pop(
        "vector_in.out_layer.weight"
    )
    converted_state_dict["vec_embed.out_layer.bias"] = original_state_dict.pop(
        "vector_in.out_layer.bias"
    )

    # context_embedder
    converted_state_dict["context_embedder.weight"] = original_state_dict.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = original_state_dict.pop("txt_in.bias")

    # x_embedder
    converted_state_dict["x_embedder.weight"] = original_state_dict.pop("img_in.weight")
    converted_state_dict["x_embedder.bias"] = original_state_dict.pop("img_in.bias")

    # connector
    remaining_key = list(original_state_dict.keys())
    for key in remaining_key:
        if 'connector' in key:
            converted_state_dict[key] = original_state_dict.pop(key)

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms.
        ## norm1
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.lin.bias"
        )
        ## norm1_context
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.lin.bias"
        )
        # Q, K, V
        sample_q, sample_k, sample_v = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0
        )
        context_q, context_k, context_v = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
        )
        context_q_bias, context_k_bias, context_v_bias = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )

        # ff img_mlp
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.2.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear  <- single_blocks.0.modulation.lin
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        q, k, v, mlp = torch.split(original_state_dict.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        q_bias, k_bias, v_bias, mlp_bias = torch.split(
            original_state_dict.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )

        # output projections.
        converted_state_dict[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.weight"
        )
        converted_state_dict[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.bias"
        )

    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.bias")
    )

    print(original_state_dict.keys())

    return converted_state_dict


def convert_step1x_edit_vae_checkpoint_to_diffusers(original_state_dict):
    converted_state_dict = {}

    # encoder.conv_in
    converted_state_dict["encoder.conv_in.weight"] = original_state_dict.pop(
        "encoder.conv_in.weight"
    )
    converted_state_dict["encoder.conv_in.bias"] = original_state_dict.pop(
        "encoder.conv_in.bias"
    )

    # encoder.conv_out
    converted_state_dict["encoder.conv_out.weight"] = original_state_dict.pop(
        "encoder.conv_out.weight"
    )
    converted_state_dict["encoder.conv_out.bias"] = original_state_dict.pop(
        "encoder.conv_out.bias"
    )

    # encoder.norm_out
    converted_state_dict["encoder.conv_norm_out.weight"] = original_state_dict.pop(
        "encoder.norm_out.weight"
    )
    converted_state_dict["encoder.conv_norm_out.bias"] = original_state_dict.pop(
        "encoder.norm_out.bias"
    )

    # encoder.down
    for i in range(4):
        # conv & norm
        for j in range(2):
            for k in range(1, 3):
                converted_state_dict[f"encoder.down_blocks.{i}.resnets.{j}.conv{k}.weight"] = original_state_dict.pop(
                    f"encoder.down.{i}.block.{j}.conv{k}.weight"
                )
                converted_state_dict[f"encoder.down_blocks.{i}.resnets.{j}.conv{k}.bias"] = original_state_dict.pop(
                    f"encoder.down.{i}.block.{j}.conv{k}.bias"
                )
                converted_state_dict[f"encoder.down_blocks.{i}.resnets.{j}.norm{k}.weight"] = original_state_dict.pop(
                    f"encoder.down.{i}.block.{j}.norm{k}.weight"
                )
                converted_state_dict[f"encoder.down_blocks.{i}.resnets.{j}.norm{k}.bias"] = original_state_dict.pop(
                    f"encoder.down.{i}.block.{j}.norm{k}.bias"
                )
        
        # downsample
        if i != 3 :
            converted_state_dict[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = original_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            converted_state_dict[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = original_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        # shortcut
        if i == 1 or i == 2:
            converted_state_dict[f"encoder.down_blocks.{i}.resnets.0.conv_shortcut.weight"] = original_state_dict.pop(
                f"encoder.down.{i}.block.0.nin_shortcut.weight"
            )
            converted_state_dict[f"encoder.down_blocks.{i}.resnets.0.conv_shortcut.bias"] = original_state_dict.pop(
                f"encoder.down.{i}.block.0.nin_shortcut.bias"
            )
        
    # encoder.mid
    converted_state_dict["encoder.mid_block.attentions.0.to_q.weight"] = original_state_dict.pop(
        "encoder.mid.attn_1.q.weight"
    ).squeeze()
    converted_state_dict["encoder.mid_block.attentions.0.to_q.bias"] = original_state_dict.pop(
        "encoder.mid.attn_1.q.bias"
    )
    converted_state_dict["encoder.mid_block.attentions.0.to_k.weight"] = original_state_dict.pop(
        "encoder.mid.attn_1.k.weight"
    ).squeeze()
    converted_state_dict["encoder.mid_block.attentions.0.to_k.bias"] = original_state_dict.pop(
        "encoder.mid.attn_1.k.bias"
    )
    converted_state_dict["encoder.mid_block.attentions.0.to_v.weight"] = original_state_dict.pop(
        "encoder.mid.attn_1.v.weight"
    ).squeeze()
    converted_state_dict["encoder.mid_block.attentions.0.to_v.bias"] = original_state_dict.pop(
        "encoder.mid.attn_1.v.bias"
    )
    converted_state_dict["encoder.mid_block.attentions.0.group_norm.weight"] = original_state_dict.pop(
        "encoder.mid.attn_1.norm.weight"
    )
    converted_state_dict["encoder.mid_block.attentions.0.group_norm.bias"] = original_state_dict.pop(
        "encoder.mid.attn_1.norm.bias"
    )
    converted_state_dict["encoder.mid_block.attentions.0.to_out.0.weight"] = original_state_dict.pop(
        "encoder.mid.attn_1.proj_out.weight"
    ).squeeze()
    converted_state_dict["encoder.mid_block.attentions.0.to_out.0.bias"] = original_state_dict.pop(
        "encoder.mid.attn_1.proj_out.bias"
    )

    # encoder.mid_block
    for i in range(2):
        for j in range(2):
            # conv
            converted_state_dict[f"encoder.mid_block.resnets.{i}.conv{j+1}.weight"] = original_state_dict.pop(
                f"encoder.mid.block_{i+1}.conv{j+1}.weight"
            )
            converted_state_dict[f"encoder.mid_block.resnets.{i}.conv{j+1}.bias"] = original_state_dict.pop(
                f"encoder.mid.block_{i+1}.conv{j+1}.bias"
            )

            # norm
            converted_state_dict[f"encoder.mid_block.resnets.{i}.norm{j+1}.weight"] = original_state_dict.pop(
                f"encoder.mid.block_{i+1}.norm{j+1}.weight"
            )
            converted_state_dict[f"encoder.mid_block.resnets.{i}.norm{j+1}.bias"] = original_state_dict.pop(
                f"encoder.mid.block_{i+1}.norm{j+1}.bias"
            )

    # decoder.conv_in
    converted_state_dict["decoder.conv_in.weight"] = original_state_dict.pop(
        "decoder.conv_in.weight"
    )
    converted_state_dict["decoder.conv_in.bias"] = original_state_dict.pop(
        "decoder.conv_in.bias"
    )

    # decoder.conv_out
    converted_state_dict["decoder.conv_out.weight"] = original_state_dict.pop(
        "decoder.conv_out.weight"
    )
    converted_state_dict["decoder.conv_out.bias"] = original_state_dict.pop(
        "decoder.conv_out.bias"
    )

    # decoder.norm_out
    converted_state_dict["decoder.conv_norm_out.weight"] = original_state_dict.pop(
        "decoder.norm_out.weight"
    )
    converted_state_dict["decoder.conv_norm_out.bias"] = original_state_dict.pop(
        "decoder.norm_out.bias"
    )
    
    # decoder.mid
    converted_state_dict["decoder.mid_block.attentions.0.to_q.weight"] = original_state_dict.pop(
        "decoder.mid.attn_1.q.weight"
    ).squeeze()
    converted_state_dict["decoder.mid_block.attentions.0.to_q.bias"] = original_state_dict.pop(
        "decoder.mid.attn_1.q.bias"
    )
    converted_state_dict["decoder.mid_block.attentions.0.to_k.weight"] = original_state_dict.pop(
        "decoder.mid.attn_1.k.weight"
    ).squeeze()
    converted_state_dict["decoder.mid_block.attentions.0.to_k.bias"] = original_state_dict.pop(
        "decoder.mid.attn_1.k.bias"
    )
    converted_state_dict["decoder.mid_block.attentions.0.to_v.weight"] = original_state_dict.pop(
        "decoder.mid.attn_1.v.weight"
    ).squeeze()
    converted_state_dict["decoder.mid_block.attentions.0.to_v.bias"] = original_state_dict.pop(
        "decoder.mid.attn_1.v.bias"
    )
    converted_state_dict["decoder.mid_block.attentions.0.group_norm.weight"] = original_state_dict.pop(
        "decoder.mid.attn_1.norm.weight"
    )
    converted_state_dict["decoder.mid_block.attentions.0.group_norm.bias"] = original_state_dict.pop(
        "decoder.mid.attn_1.norm.bias"
    )
    converted_state_dict["decoder.mid_block.attentions.0.to_out.0.weight"] = original_state_dict.pop(
        "decoder.mid.attn_1.proj_out.weight"
    ).squeeze()
    converted_state_dict["decoder.mid_block.attentions.0.to_out.0.bias"] = original_state_dict.pop(
        "decoder.mid.attn_1.proj_out.bias"
    )
    
    # decoder.mid_block
    for i in range(2):
        for j in range(2):
            # conv
            converted_state_dict[f"decoder.mid_block.resnets.{i}.conv{j+1}.weight"] = original_state_dict.pop(
                f"decoder.mid.block_{i+1}.conv{j+1}.weight"
            )
            converted_state_dict[f"decoder.mid_block.resnets.{i}.conv{j+1}.bias"] = original_state_dict.pop(
                f"decoder.mid.block_{i+1}.conv{j+1}.bias"
            )

            # norm
            converted_state_dict[f"decoder.mid_block.resnets.{i}.norm{j+1}.weight"] = original_state_dict.pop(
                f"decoder.mid.block_{i+1}.norm{j+1}.weight"
            )
            converted_state_dict[f"decoder.mid_block.resnets.{i}.norm{j+1}.bias"] = original_state_dict.pop(
                f"decoder.mid.block_{i+1}.norm{j+1}.bias"
            )
    
    # decoder.up
    for i in range(4):
        # conv & norm
        for j in range(3):
            for k in range(1, 3):
                converted_state_dict[f"decoder.up_blocks.{3-i}.resnets.{j}.conv{k}.weight"] = original_state_dict.pop(
                    f"decoder.up.{i}.block.{j}.conv{k}.weight"
                )
                converted_state_dict[f"decoder.up_blocks.{3-i}.resnets.{j}.conv{k}.bias"] = original_state_dict.pop(
                    f"decoder.up.{i}.block.{j}.conv{k}.bias"
                )
                converted_state_dict[f"decoder.up_blocks.{3-i}.resnets.{j}.norm{k}.weight"] = original_state_dict.pop(
                    f"decoder.up.{i}.block.{j}.norm{k}.weight"
                )
                converted_state_dict[f"decoder.up_blocks.{3-i}.resnets.{j}.norm{k}.bias"] = original_state_dict.pop(
                    f"decoder.up.{i}.block.{j}.norm{k}.bias"
                )
        
        # downsample
        if i != 0 :
            converted_state_dict[f"decoder.up_blocks.{3-i}.upsamplers.0.conv.weight"] = original_state_dict.pop(
                f"decoder.up.{i}.upsample.conv.weight"
            )
            converted_state_dict[f"decoder.up_blocks.{3-i}.upsamplers.0.conv.bias"] = original_state_dict.pop(
                f"decoder.up.{i}.upsample.conv.bias"
            )

        # shortcut
        if i == 0 or i == 1:
            converted_state_dict[f"decoder.up_blocks.{3-i}.resnets.0.conv_shortcut.weight"] = original_state_dict.pop(
                f"decoder.up.{i}.block.0.nin_shortcut.weight"
            )
            converted_state_dict[f"decoder.up_blocks.{3-i}.resnets.0.conv_shortcut.bias"] = original_state_dict.pop(
                f"decoder.up.{i}.block.0.nin_shortcut.bias"
            )
    
    return converted_state_dict


def main(args):
    
    if args.transformer:
        original_ckpt = load_original_checkpoint(args)
        
        num_layers = 19
        num_single_layers = 38
        inner_dim = 3072
        mlp_ratio = 4.0

        converted_transformer_state_dict = convert_step1x_edit_transformer_checkpoint_to_diffusers(
            original_ckpt, num_layers, num_single_layers, inner_dim, mlp_ratio=mlp_ratio
        )
        transformer = Step1XEditTransformer2DModel(
            in_channels=args.in_channels, out_channels=args.out_channels
        )
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

    if args.vae:
        original_ckpt = load_original_checkpoint(args)
        converted_vae_state_dict = convert_step1x_edit_vae_checkpoint_to_diffusers(original_ckpt)
        vae = AutoencoderKL(
            in_channels = 3,
            out_channels = 3,
            down_block_types = [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ],
            up_block_types = [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            block_out_channels = [
                128,
                256,
                512,
                512
            ],
            layers_per_block = 2,
            act_fn = "silu",
            latent_channels = 16,
            norm_num_groups = 32,
            sample_size = 1024,
            scaling_factor = 0.3611,
            shift_factor = 0.1159,
            latents_mean  = None,
            latents_std = None,
            force_upcast = True,
            use_quant_conv = False,
            use_post_quant_conv = False,
            mid_block_add_attention = True,
        )
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")

    if args.text_encoder:
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.original_state_dict_repo_id,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        image_encoder = AutoProcessor.from_pretrained(args.original_state_dict_repo_id, min_pixels = 256 * 28 * 28, max_pixels = 324 * 28 * 28)
        text_encoder.save_pretrained(f"{args.output_path}/text_encoder")
        image_encoder.save_pretrained(f"{args.output_path}/processor")
        
    if args.scheduler:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.original_state_dict_repo_id,
            subfolder="scheduler"
        )
        scheduler.save_pretrained(f"{args.output_path}/scheduler")


if __name__ == "__main__":
    main(args)
