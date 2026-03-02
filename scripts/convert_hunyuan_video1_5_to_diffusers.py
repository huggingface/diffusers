import argparse
import json
import os
import pathlib

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import (
    AutoModel,
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKLHunyuanVideo15,
    ClassifierFreeGuidance,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideo15ImageToVideoPipeline,
    HunyuanVideo15Pipeline,
    HunyuanVideo15Transformer3DModel,
)


# to convert only transformer
"""
python scripts/convert_hunyuan_video1_5_to_diffusers.py \
    --original_state_dict_repo_id tencent/HunyuanVideo-1.5\
    --output_path /fsx/yiyi/HunyuanVideo-1.5-Diffusers/transformer\
    --transformer_type 480p_t2v
"""

# to convert full pipeline
"""
python scripts/convert_hunyuan_video1_5_to_diffusers.py \
    --original_state_dict_repo_id tencent/HunyuanVideo-1.5\
    --output_path /fsx/yiyi/HunyuanVideo-1.5-Diffusers \
    --save_pipeline \
    --byt5_path /fsx/yiyi/hy15/text_encoder/Glyph-SDXL-v2\
    --transformer_type 480p_t2v
"""


TRANSFORMER_CONFIGS = {
    "480p_t2v": {
        "target_size": 640,
        "task_type": "i2v",
    },
    "720p_t2v": {
        "target_size": 960,
        "task_type": "t2v",
    },
    "720p_i2v": {
        "target_size": 960,
        "task_type": "i2v",
    },
    "480p_t2v_distilled": {
        "target_size": 640,
        "task_type": "t2v",
    },
    "480p_i2v_distilled": {
        "target_size": 640,
        "task_type": "i2v",
    },
    "720p_i2v_distilled": {
        "target_size": 960,
        "task_type": "i2v",
    },
    "480p_i2v_step_distilled": {
        "target_size": 640,
        "task_type": "i2v",
        "use_meanflow": True,
    },
}

SCHEDULER_CONFIGS = {
    "480p_t2v": {
        "shift": 5.0,
    },
    "480p_i2v": {
        "shift": 5.0,
    },
    "720p_t2v": {
        "shift": 9.0,
    },
    "720p_i2v": {
        "shift": 7.0,
    },
    "480p_t2v_distilled": {
        "shift": 5.0,
    },
    "480p_i2v_distilled": {
        "shift": 5.0,
    },
    "720p_i2v_distilled": {
        "shift": 7.0,
    },
    "480p_i2v_step_distilled": {
        "shift": 7.0,
    },
}

GUIDANCE_CONFIGS = {
    "480p_t2v": {
        "guidance_scale": 6.0,
    },
    "480p_i2v": {
        "guidance_scale": 6.0,
    },
    "720p_t2v": {
        "guidance_scale": 6.0,
    },
    "720p_i2v": {
        "guidance_scale": 6.0,
    },
    "480p_t2v_distilled": {
        "guidance_scale": 1.0,
    },
    "480p_i2v_distilled": {
        "guidance_scale": 1.0,
    },
    "720p_i2v_distilled": {
        "guidance_scale": 1.0,
    },
    "480p_i2v_step_distilled": {
        "guidance_scale": 1.0,
    },
}


def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_hyvideo15_transformer_to_diffusers(original_state_dict, config=None):
    """
    Convert HunyuanVideo 1.5 original checkpoint to Diffusers format.
    """
    converted_state_dict = {}

    # 1. time_embed.timestep_embedder <- time_in
    converted_state_dict["time_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "time_in.mlp.0.weight"
    )
    converted_state_dict["time_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop("time_in.mlp.0.bias")
    converted_state_dict["time_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "time_in.mlp.2.weight"
    )
    converted_state_dict["time_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop("time_in.mlp.2.bias")

    if config.use_meanflow:
        converted_state_dict["time_embed.timestep_embedder_r.linear_1.weight"] = original_state_dict.pop(
            "time_r_in.mlp.0.weight"
        )
        converted_state_dict["time_embed.timestep_embedder_r.linear_1.bias"] = original_state_dict.pop(
            "time_r_in.mlp.0.bias"
        )
        converted_state_dict["time_embed.timestep_embedder_r.linear_2.weight"] = original_state_dict.pop(
            "time_r_in.mlp.2.weight"
        )
        converted_state_dict["time_embed.timestep_embedder_r.linear_2.bias"] = original_state_dict.pop(
            "time_r_in.mlp.2.bias"
        )

    # 2. context_embedder.time_text_embed.timestep_embedder <- txt_in.t_embedder
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

    # 3. context_embedder.time_text_embed.text_embedder <- txt_in.c_embedder
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

    # 4. context_embedder.proj_in <- txt_in.input_embedder
    converted_state_dict["context_embedder.proj_in.weight"] = original_state_dict.pop("txt_in.input_embedder.weight")
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
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.bias")
    )
    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")

    return converted_state_dict


def convert_hunyuan_video_15_vae_checkpoint_to_diffusers(
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

    return converted


def load_sharded_safetensors(dir: pathlib.Path):
    file_paths = list(dir.glob("diffusion_pytorch_model*.safetensors"))
    state_dict = {}
    for path in file_paths:
        state_dict.update(load_file(path))
    return state_dict


def load_original_transformer_state_dict(args):
    if args.original_state_dict_repo_id is not None:
        model_dir = snapshot_download(
            args.original_state_dict_repo_id,
            repo_type="model",
            allow_patterns="transformer/" + args.transformer_type + "/*",
        )
    elif args.original_state_dict_folder is not None:
        model_dir = pathlib.Path(args.original_state_dict_folder)
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or `original_state_dict_folder`")
    model_dir = pathlib.Path(model_dir)
    model_dir = model_dir / "transformer" / args.transformer_type
    return load_sharded_safetensors(model_dir)


def load_original_vae_state_dict(args):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(
            repo_id=args.original_state_dict_repo_id, filename="vae/diffusion_pytorch_model.safetensors"
        )
    elif args.original_state_dict_folder is not None:
        model_dir = pathlib.Path(args.original_state_dict_folder)
        ckpt_path = model_dir / "vae/diffusion_pytorch_model.safetensors"
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or `original_state_dict_folder`")

    original_state_dict = load_file(ckpt_path)
    return original_state_dict


def convert_transformer(args):
    original_state_dict = load_original_transformer_state_dict(args)

    config = TRANSFORMER_CONFIGS[args.transformer_type]
    with init_empty_weights():
        transformer = HunyuanVideo15Transformer3DModel(**config)
    state_dict = convert_hyvideo15_transformer_to_diffusers(original_state_dict, config=transformer.config)
    transformer.load_state_dict(state_dict, strict=True, assign=True)

    return transformer


def convert_vae(args):
    original_state_dict = load_original_vae_state_dict(args)
    with init_empty_weights():
        vae = AutoencoderKLHunyuanVideo15()
    state_dict = convert_hunyuan_video_15_vae_checkpoint_to_diffusers(original_state_dict)
    vae.load_state_dict(state_dict, strict=True, assign=True)
    return vae


def load_mllm():
    print(" loading from Qwen/Qwen2.5-VL-7B-Instruct")
    text_encoder = AutoModel.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if hasattr(text_encoder, "language_model"):
        text_encoder = text_encoder.language_model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", padding_side="right")
    return text_encoder, tokenizer


# copied from https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/910da2a829c484ea28982e8cff3bbc2cacdf1681/hyvideo/models/text_encoders/byT5/__init__.py#L89
def add_special_token(
    tokenizer,
    text_encoder,
    add_color=True,
    add_font=True,
    multilingual=True,
    color_ann_path="assets/color_idx.json",
    font_ann_path="assets/multilingual_10-lang_idx.json",
):
    """
    Add special tokens for color and font to tokenizer and text encoder.

    Args:
        tokenizer: Huggingface tokenizer.
        text_encoder: Huggingface T5 encoder.
        add_color (bool): Whether to add color tokens.
        add_font (bool): Whether to add font tokens.
        color_ann_path (str): Path to color annotation JSON.
        font_ann_path (str): Path to font annotation JSON.
        multilingual (bool): Whether to use multilingual font tokens.
    """
    with open(font_ann_path, "r") as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, "r") as f:
        idx_color_dict = json.load(f)

    if multilingual:
        font_token = [f"<{font_code[:2]}-font-{idx_font_dict[font_code]}>" for font_code in idx_font_dict]
    else:
        font_token = [f"<font-{i}>" for i in range(len(idx_font_dict))]
    color_token = [f"<color-{i}>" for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token

    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    # Set mean_resizing=False to avoid PyTorch LAPACK dependency
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_byt5(args):
    """
    Load ByT5 encoder with Glyph-SDXL-v2 weights and save in HuggingFace format.
    """

    # 1. Load base tokenizer and encoder
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Load as T5EncoderModel
    encoder = T5EncoderModel.from_pretrained("google/byt5-small")

    byt5_checkpoint_path = os.path.join(args.byt5_path, "checkpoints/byt5_model.pt")
    color_ann_path = os.path.join(args.byt5_path, "assets/color_idx.json")
    font_ann_path = os.path.join(args.byt5_path, "assets/multilingual_10-lang_idx.json")

    # 2. Add special tokens
    add_special_token(
        tokenizer=tokenizer,
        text_encoder=encoder,
        add_color=True,
        add_font=True,
        color_ann_path=color_ann_path,
        font_ann_path=font_ann_path,
        multilingual=True,
    )

    # 3. Load Glyph-SDXL-v2 checkpoint
    print(f"\n3. Loading Glyph-SDXL-v2 checkpoint: {byt5_checkpoint_path}")
    checkpoint = torch.load(byt5_checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # add 'encoder.' prefix to the keys
    # Remove 'module.text_tower.encoder.' prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.text_tower.encoder."):
            new_key = "encoder." + key[len("module.text_tower.encoder.") :]
            cleaned_state_dict[new_key] = value
        else:
            new_key = "encoder." + key
            cleaned_state_dict[new_key] = value

    # 4. Load weights
    missing_keys, unexpected_keys = encoder.load_state_dict(cleaned_state_dict, strict=False)
    if unexpected_keys:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")
    if "shared.weight" in missing_keys:
        print("  Missing shared.weight as expected")
        missing_keys.remove("shared.weight")
    if missing_keys:
        raise ValueError(f"Missing keys: {missing_keys}")

    return encoder, tokenizer


def load_siglip():
    image_encoder = SiglipVisionModel.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev", subfolder="image_encoder", torch_dtype=torch.bfloat16
    )
    feature_extractor = SiglipImageProcessor.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev", subfolder="feature_extractor"
    )
    return image_encoder, feature_extractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_state_dict_repo_id", type=str, default=None, help="Path to original hub_id for the model"
    )
    parser.add_argument(
        "--original_state_dict_folder", type=str, default=None, help="Local folder name of the original state dict"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model(s) should be saved")
    parser.add_argument("--transformer_type", type=str, default="480p_i2v", choices=list(TRANSFORMER_CONFIGS.keys()))
    parser.add_argument(
        "--byt5_path",
        type=str,
        default=None,
        help=(
            "path to the downloaded byt5 checkpoint & assets. "
            "Note: They use Glyph-SDXL-v2 as byt5 encoder. You can download from modelscope like: "
            "`modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2` "
            "or manually download following the instructions on "
            "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/910da2a829c484ea28982e8cff3bbc2cacdf1681/checkpoints-download.md. "
            "The path should point to the Glyph-SDXL-v2 folder which should contain an `assets` folder and a `checkpoints` folder, "
            "like: Glyph-SDXL-v2/assets/... and Glyph-SDXL-v2/checkpoints/byt5_model.pt"
        ),
    )
    parser.add_argument("--save_pipeline", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.save_pipeline and args.byt5_path is None:
        raise ValueError("Please provide --byt5_path when saving pipeline")

    transformer = None

    transformer = convert_transformer(args)
    if not args.save_pipeline:
        transformer.save_pretrained(args.output_path, safe_serialization=True)
    else:
        task_type = transformer.config.task_type

        vae = convert_vae(args)

        text_encoder, tokenizer = load_mllm()
        text_encoder_2, tokenizer_2 = load_byt5(args)

        flow_shift = SCHEDULER_CONFIGS[args.transformer_type]["shift"]
        scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

        guidance_scale = GUIDANCE_CONFIGS[args.transformer_type]["guidance_scale"]
        guider = ClassifierFreeGuidance(guidance_scale=guidance_scale)

        if task_type == "i2v":
            image_encoder, feature_extractor = load_siglip()
            pipeline = HunyuanVideo15ImageToVideoPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
                guider=guider,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
        elif task_type == "t2v":
            pipeline = HunyuanVideo15Pipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
                guider=guider,
                scheduler=scheduler,
            )
        else:
            raise ValueError(f"Task type {task_type} is not supported")

        pipeline.save_pretrained(args.output_path, safe_serialization=True)
