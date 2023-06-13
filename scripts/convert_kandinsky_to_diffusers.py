import argparse
import os
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch

from diffusers import UNet2DConditionModel
from diffusers.models.prior_transformer import PriorTransformer
from diffusers.models.vq_model import VQModel


"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget https://huggingface.co/ai-forever/Kandinsky_2.1/blob/main/prior_fp16.ckpt
```

Convert the model:
```sh
python scripts/convert_kandinsky_to_diffusers.py \
      --prior_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/prior_fp16.ckpt \
      --clip_stat_path  /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/ViT-L-14_stats.th \
      --text2img_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/decoder_fp16.ckpt \
      --inpaint_text2img_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/inpainting_fp16.ckpt \
      --movq_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/movq_final.ckpt \
      --dump_path /home/yiyi_huggingface_co/dump \
      --debug decoder
```
"""


# prior

PRIOR_ORIGINAL_PREFIX = "model"

# Uses default arguments
PRIOR_CONFIG = {}


def prior_model_from_original_config():
    model = PriorTransformer(**PRIOR_CONFIG)

    return model


def prior_original_checkpoint_to_diffusers_checkpoint(model, checkpoint, clip_stats_checkpoint):
    diffusers_checkpoint = {}

    # <original>.time_embed.0 -> <diffusers>.time_embedding.linear_1
    diffusers_checkpoint.update(
        {
            "time_embedding.linear_1.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.0.weight"],
            "time_embedding.linear_1.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.0.bias"],
        }
    )

    # <original>.clip_img_proj -> <diffusers>.proj_in
    diffusers_checkpoint.update(
        {
            "proj_in.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.clip_img_proj.weight"],
            "proj_in.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.clip_img_proj.bias"],
        }
    )

    # <original>.text_emb_proj -> <diffusers>.embedding_proj
    diffusers_checkpoint.update(
        {
            "embedding_proj.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.text_emb_proj.weight"],
            "embedding_proj.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.text_emb_proj.bias"],
        }
    )

    # <original>.text_enc_proj -> <diffusers>.encoder_hidden_states_proj
    diffusers_checkpoint.update(
        {
            "encoder_hidden_states_proj.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.text_enc_proj.weight"],
            "encoder_hidden_states_proj.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.text_enc_proj.bias"],
        }
    )

    # <original>.positional_embedding -> <diffusers>.positional_embedding
    diffusers_checkpoint.update({"positional_embedding": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.positional_embedding"]})

    # <original>.prd_emb -> <diffusers>.prd_embedding
    diffusers_checkpoint.update({"prd_embedding": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.prd_emb"]})

    # <original>.time_embed.2 -> <diffusers>.time_embedding.linear_2
    diffusers_checkpoint.update(
        {
            "time_embedding.linear_2.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.2.weight"],
            "time_embedding.linear_2.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.2.bias"],
        }
    )

    # <original>.resblocks.<x> -> <diffusers>.transformer_blocks.<x>
    for idx in range(len(model.transformer_blocks)):
        diffusers_transformer_prefix = f"transformer_blocks.{idx}"
        original_transformer_prefix = f"{PRIOR_ORIGINAL_PREFIX}.transformer.resblocks.{idx}"

        # <original>.attn -> <diffusers>.attn1
        diffusers_attention_prefix = f"{diffusers_transformer_prefix}.attn1"
        original_attention_prefix = f"{original_transformer_prefix}.attn"
        diffusers_checkpoint.update(
            prior_attention_to_diffusers(
                checkpoint,
                diffusers_attention_prefix=diffusers_attention_prefix,
                original_attention_prefix=original_attention_prefix,
                attention_head_dim=model.attention_head_dim,
            )
        )

        # <original>.mlp -> <diffusers>.ff
        diffusers_ff_prefix = f"{diffusers_transformer_prefix}.ff"
        original_ff_prefix = f"{original_transformer_prefix}.mlp"
        diffusers_checkpoint.update(
            prior_ff_to_diffusers(
                checkpoint, diffusers_ff_prefix=diffusers_ff_prefix, original_ff_prefix=original_ff_prefix
            )
        )

        # <original>.ln_1 -> <diffusers>.norm1
        diffusers_checkpoint.update(
            {
                f"{diffusers_transformer_prefix}.norm1.weight": checkpoint[
                    f"{original_transformer_prefix}.ln_1.weight"
                ],
                f"{diffusers_transformer_prefix}.norm1.bias": checkpoint[f"{original_transformer_prefix}.ln_1.bias"],
            }
        )

        # <original>.ln_2 -> <diffusers>.norm3
        diffusers_checkpoint.update(
            {
                f"{diffusers_transformer_prefix}.norm3.weight": checkpoint[
                    f"{original_transformer_prefix}.ln_2.weight"
                ],
                f"{diffusers_transformer_prefix}.norm3.bias": checkpoint[f"{original_transformer_prefix}.ln_2.bias"],
            }
        )

    # <original>.final_ln -> <diffusers>.norm_out
    diffusers_checkpoint.update(
        {
            "norm_out.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.final_ln.weight"],
            "norm_out.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.final_ln.bias"],
        }
    )

    # <original>.out_proj -> <diffusers>.proj_to_clip_embeddings
    diffusers_checkpoint.update(
        {
            "proj_to_clip_embeddings.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.out_proj.weight"],
            "proj_to_clip_embeddings.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.out_proj.bias"],
        }
    )

    # clip stats
    clip_mean, clip_std = clip_stats_checkpoint
    clip_mean = clip_mean[None, :]
    clip_std = clip_std[None, :]

    diffusers_checkpoint.update({"clip_mean": clip_mean, "clip_std": clip_std})

    return diffusers_checkpoint


def prior_attention_to_diffusers(
    checkpoint, *, diffusers_attention_prefix, original_attention_prefix, attention_head_dim
):
    diffusers_checkpoint = {}

    # <original>.c_qkv -> <diffusers>.{to_q, to_k, to_v}
    [q_weight, k_weight, v_weight], [q_bias, k_bias, v_bias] = split_attentions(
        weight=checkpoint[f"{original_attention_prefix}.c_qkv.weight"],
        bias=checkpoint[f"{original_attention_prefix}.c_qkv.bias"],
        split=3,
        chunk_size=attention_head_dim,
    )

    diffusers_checkpoint.update(
        {
            f"{diffusers_attention_prefix}.to_q.weight": q_weight,
            f"{diffusers_attention_prefix}.to_q.bias": q_bias,
            f"{diffusers_attention_prefix}.to_k.weight": k_weight,
            f"{diffusers_attention_prefix}.to_k.bias": k_bias,
            f"{diffusers_attention_prefix}.to_v.weight": v_weight,
            f"{diffusers_attention_prefix}.to_v.bias": v_bias,
        }
    )

    # <original>.c_proj -> <diffusers>.to_out.0
    diffusers_checkpoint.update(
        {
            f"{diffusers_attention_prefix}.to_out.0.weight": checkpoint[f"{original_attention_prefix}.c_proj.weight"],
            f"{diffusers_attention_prefix}.to_out.0.bias": checkpoint[f"{original_attention_prefix}.c_proj.bias"],
        }
    )

    return diffusers_checkpoint


def prior_ff_to_diffusers(checkpoint, *, diffusers_ff_prefix, original_ff_prefix):
    diffusers_checkpoint = {
        # <original>.c_fc -> <diffusers>.net.0.proj
        f"{diffusers_ff_prefix}.net.{0}.proj.weight": checkpoint[f"{original_ff_prefix}.c_fc.weight"],
        f"{diffusers_ff_prefix}.net.{0}.proj.bias": checkpoint[f"{original_ff_prefix}.c_fc.bias"],
        # <original>.c_proj -> <diffusers>.net.2
        f"{diffusers_ff_prefix}.net.{2}.weight": checkpoint[f"{original_ff_prefix}.c_proj.weight"],
        f"{diffusers_ff_prefix}.net.{2}.bias": checkpoint[f"{original_ff_prefix}.c_proj.bias"],
    }

    return diffusers_checkpoint


# done prior

# unet

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.

UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_image",
    "addition_embed_type_num_heads": 64,
    "attention_head_dim": 64,
    "block_out_channels": [384, 768, 1152, 1536],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": False,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 768,
    "cross_attention_norm": None,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": 1024,
    "encoder_hid_dim_type": "text_image_proj",
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 3,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 8,
    "projection_class_embeddings_input_dim": None,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "scale_shift",
    "sample_size": 64,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "up_block_types": [
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "upcast_attention": False,
    "use_linear_projection": False,
}


def unet_model_from_original_config():
    model = UNet2DConditionModel(**UNET_CONFIG)

    return model


def unet_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    num_head_channels = UNET_CONFIG["attention_head_dim"]

    diffusers_checkpoint.update(unet_time_embeddings(checkpoint))
    diffusers_checkpoint.update(unet_conv_in(checkpoint))
    diffusers_checkpoint.update(unet_add_embedding(checkpoint))
    diffusers_checkpoint.update(unet_encoder_hid_proj(checkpoint))

    # <original>.input_blocks -> <diffusers>.down_blocks

    original_down_block_idx = 1

    for diffusers_down_block_idx in range(len(model.down_blocks)):
        checkpoint_update, num_original_down_blocks = unet_downblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            diffusers_down_block_idx=diffusers_down_block_idx,
            original_down_block_idx=original_down_block_idx,
            num_head_channels=num_head_channels,
        )

        original_down_block_idx += num_original_down_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.input_blocks -> <diffusers>.down_blocks

    diffusers_checkpoint.update(
        unet_midblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            num_head_channels=num_head_channels,
        )
    )

    # <original>.output_blocks -> <diffusers>.up_blocks

    original_up_block_idx = 0

    for diffusers_up_block_idx in range(len(model.up_blocks)):
        checkpoint_update, num_original_up_blocks = unet_upblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            diffusers_up_block_idx=diffusers_up_block_idx,
            original_up_block_idx=original_up_block_idx,
            num_head_channels=num_head_channels,
        )

        original_up_block_idx += num_original_up_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.output_blocks -> <diffusers>.up_blocks

    diffusers_checkpoint.update(unet_conv_norm_out(checkpoint))
    diffusers_checkpoint.update(unet_conv_out(checkpoint))

    return diffusers_checkpoint


# done unet

# inpaint unet

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.

INPAINT_UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_image",
    "addition_embed_type_num_heads": 64,
    "attention_head_dim": 64,
    "block_out_channels": [384, 768, 1152, 1536],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": None,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 768,
    "cross_attention_norm": None,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": 1024,
    "encoder_hid_dim_type": "text_image_proj",
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 9,
    "layers_per_block": 3,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 8,
    "projection_class_embeddings_input_dim": None,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "scale_shift",
    "sample_size": 64,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "up_block_types": [
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "upcast_attention": False,
    "use_linear_projection": False,
}


def inpaint_unet_model_from_original_config():
    model = UNet2DConditionModel(**INPAINT_UNET_CONFIG)

    return model


def inpaint_unet_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    num_head_channels = INPAINT_UNET_CONFIG["attention_head_dim"]

    diffusers_checkpoint.update(unet_time_embeddings(checkpoint))
    diffusers_checkpoint.update(unet_conv_in(checkpoint))
    diffusers_checkpoint.update(unet_add_embedding(checkpoint))
    diffusers_checkpoint.update(unet_encoder_hid_proj(checkpoint))

    # <original>.input_blocks -> <diffusers>.down_blocks

    original_down_block_idx = 1

    for diffusers_down_block_idx in range(len(model.down_blocks)):
        checkpoint_update, num_original_down_blocks = unet_downblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            diffusers_down_block_idx=diffusers_down_block_idx,
            original_down_block_idx=original_down_block_idx,
            num_head_channels=num_head_channels,
        )

        original_down_block_idx += num_original_down_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.input_blocks -> <diffusers>.down_blocks

    diffusers_checkpoint.update(
        unet_midblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            num_head_channels=num_head_channels,
        )
    )

    # <original>.output_blocks -> <diffusers>.up_blocks

    original_up_block_idx = 0

    for diffusers_up_block_idx in range(len(model.up_blocks)):
        checkpoint_update, num_original_up_blocks = unet_upblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            diffusers_up_block_idx=diffusers_up_block_idx,
            original_up_block_idx=original_up_block_idx,
            num_head_channels=num_head_channels,
        )

        original_up_block_idx += num_original_up_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.output_blocks -> <diffusers>.up_blocks

    diffusers_checkpoint.update(unet_conv_norm_out(checkpoint))
    diffusers_checkpoint.update(unet_conv_out(checkpoint))

    return diffusers_checkpoint


# done inpaint unet


# unet utils


# <original>.time_embed -> <diffusers>.time_embedding
def unet_time_embeddings(checkpoint):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "time_embedding.linear_1.weight": checkpoint["time_embed.0.weight"],
            "time_embedding.linear_1.bias": checkpoint["time_embed.0.bias"],
            "time_embedding.linear_2.weight": checkpoint["time_embed.2.weight"],
            "time_embedding.linear_2.bias": checkpoint["time_embed.2.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.input_blocks.0 -> <diffusers>.conv_in
def unet_conv_in(checkpoint):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "conv_in.weight": checkpoint["input_blocks.0.0.weight"],
            "conv_in.bias": checkpoint["input_blocks.0.0.bias"],
        }
    )

    return diffusers_checkpoint


def unet_add_embedding(checkpoint):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "add_embedding.text_norm.weight": checkpoint["ln_model_n.weight"],
            "add_embedding.text_norm.bias": checkpoint["ln_model_n.bias"],
            "add_embedding.text_proj.weight": checkpoint["proj_n.weight"],
            "add_embedding.text_proj.bias": checkpoint["proj_n.bias"],
            "add_embedding.image_proj.weight": checkpoint["img_layer.weight"],
            "add_embedding.image_proj.bias": checkpoint["img_layer.bias"],
        }
    )

    return diffusers_checkpoint


def unet_encoder_hid_proj(checkpoint):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "encoder_hid_proj.image_embeds.weight": checkpoint["clip_to_seq.weight"],
            "encoder_hid_proj.image_embeds.bias": checkpoint["clip_to_seq.bias"],
            "encoder_hid_proj.text_proj.weight": checkpoint["to_model_dim_n.weight"],
            "encoder_hid_proj.text_proj.bias": checkpoint["to_model_dim_n.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.out.0 -> <diffusers>.conv_norm_out
def unet_conv_norm_out(checkpoint):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "conv_norm_out.weight": checkpoint["out.0.weight"],
            "conv_norm_out.bias": checkpoint["out.0.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.out.2 -> <diffusers>.conv_out
def unet_conv_out(checkpoint):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "conv_out.weight": checkpoint["out.2.weight"],
            "conv_out.bias": checkpoint["out.2.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.input_blocks -> <diffusers>.down_blocks
def unet_downblock_to_diffusers_checkpoint(
    model, checkpoint, *, diffusers_down_block_idx, original_down_block_idx, num_head_channels
):
    diffusers_checkpoint = {}

    diffusers_resnet_prefix = f"down_blocks.{diffusers_down_block_idx}.resnets"
    original_down_block_prefix = "input_blocks"

    down_block = model.down_blocks[diffusers_down_block_idx]

    num_resnets = len(down_block.resnets)

    if down_block.downsamplers is None:
        downsampler = False
    else:
        assert len(down_block.downsamplers) == 1
        downsampler = True
        # The downsample block is also a resnet
        num_resnets += 1

    for resnet_idx_inc in range(num_resnets):
        full_resnet_prefix = f"{original_down_block_prefix}.{original_down_block_idx + resnet_idx_inc}.0"

        if downsampler and resnet_idx_inc == num_resnets - 1:
            # this is a downsample block
            full_diffusers_resnet_prefix = f"down_blocks.{diffusers_down_block_idx}.downsamplers.0"
        else:
            # this is a regular resnet block
            full_diffusers_resnet_prefix = f"{diffusers_resnet_prefix}.{resnet_idx_inc}"

        diffusers_checkpoint.update(
            resnet_to_diffusers_checkpoint(
                checkpoint, resnet_prefix=full_resnet_prefix, diffusers_resnet_prefix=full_diffusers_resnet_prefix
            )
        )

    if hasattr(down_block, "attentions"):
        num_attentions = len(down_block.attentions)
        diffusers_attention_prefix = f"down_blocks.{diffusers_down_block_idx}.attentions"

        for attention_idx_inc in range(num_attentions):
            full_attention_prefix = f"{original_down_block_prefix}.{original_down_block_idx + attention_idx_inc}.1"
            full_diffusers_attention_prefix = f"{diffusers_attention_prefix}.{attention_idx_inc}"

            diffusers_checkpoint.update(
                attention_to_diffusers_checkpoint(
                    checkpoint,
                    attention_prefix=full_attention_prefix,
                    diffusers_attention_prefix=full_diffusers_attention_prefix,
                    num_head_channels=num_head_channels,
                )
            )

    num_original_down_blocks = num_resnets

    return diffusers_checkpoint, num_original_down_blocks


# <original>.middle_block -> <diffusers>.mid_block
def unet_midblock_to_diffusers_checkpoint(model, checkpoint, *, num_head_channels):
    diffusers_checkpoint = {}

    # block 0

    original_block_idx = 0

    diffusers_checkpoint.update(
        resnet_to_diffusers_checkpoint(
            checkpoint,
            diffusers_resnet_prefix="mid_block.resnets.0",
            resnet_prefix=f"middle_block.{original_block_idx}",
        )
    )

    original_block_idx += 1

    # optional block 1

    if hasattr(model.mid_block, "attentions") and model.mid_block.attentions[0] is not None:
        diffusers_checkpoint.update(
            attention_to_diffusers_checkpoint(
                checkpoint,
                diffusers_attention_prefix="mid_block.attentions.0",
                attention_prefix=f"middle_block.{original_block_idx}",
                num_head_channels=num_head_channels,
            )
        )
        original_block_idx += 1

    # block 1 or block 2

    diffusers_checkpoint.update(
        resnet_to_diffusers_checkpoint(
            checkpoint,
            diffusers_resnet_prefix="mid_block.resnets.1",
            resnet_prefix=f"middle_block.{original_block_idx}",
        )
    )

    return diffusers_checkpoint


# <original>.output_blocks -> <diffusers>.up_blocks
def unet_upblock_to_diffusers_checkpoint(
    model, checkpoint, *, diffusers_up_block_idx, original_up_block_idx, num_head_channels
):
    diffusers_checkpoint = {}

    diffusers_resnet_prefix = f"up_blocks.{diffusers_up_block_idx}.resnets"
    original_up_block_prefix = "output_blocks"

    up_block = model.up_blocks[diffusers_up_block_idx]

    num_resnets = len(up_block.resnets)

    if up_block.upsamplers is None:
        upsampler = False
    else:
        assert len(up_block.upsamplers) == 1
        upsampler = True
        # The upsample block is also a resnet
        num_resnets += 1

    has_attentions = hasattr(up_block, "attentions")

    for resnet_idx_inc in range(num_resnets):
        if upsampler and resnet_idx_inc == num_resnets - 1:
            # this is an upsample block
            if has_attentions:
                # There is a middle attention block that we skip
                original_resnet_block_idx = 2
            else:
                original_resnet_block_idx = 1

            # we add the `minus 1` because the last two resnets are stuck together in the same output block
            full_resnet_prefix = (
                f"{original_up_block_prefix}.{original_up_block_idx + resnet_idx_inc - 1}.{original_resnet_block_idx}"
            )

            full_diffusers_resnet_prefix = f"up_blocks.{diffusers_up_block_idx}.upsamplers.0"
        else:
            # this is a regular resnet block
            full_resnet_prefix = f"{original_up_block_prefix}.{original_up_block_idx + resnet_idx_inc}.0"
            full_diffusers_resnet_prefix = f"{diffusers_resnet_prefix}.{resnet_idx_inc}"

        diffusers_checkpoint.update(
            resnet_to_diffusers_checkpoint(
                checkpoint, resnet_prefix=full_resnet_prefix, diffusers_resnet_prefix=full_diffusers_resnet_prefix
            )
        )

    if has_attentions:
        num_attentions = len(up_block.attentions)
        diffusers_attention_prefix = f"up_blocks.{diffusers_up_block_idx}.attentions"

        for attention_idx_inc in range(num_attentions):
            full_attention_prefix = f"{original_up_block_prefix}.{original_up_block_idx + attention_idx_inc}.1"
            full_diffusers_attention_prefix = f"{diffusers_attention_prefix}.{attention_idx_inc}"

            diffusers_checkpoint.update(
                attention_to_diffusers_checkpoint(
                    checkpoint,
                    attention_prefix=full_attention_prefix,
                    diffusers_attention_prefix=full_diffusers_attention_prefix,
                    num_head_channels=num_head_channels,
                )
            )

    num_original_down_blocks = num_resnets - 1 if upsampler else num_resnets

    return diffusers_checkpoint, num_original_down_blocks


def resnet_to_diffusers_checkpoint(checkpoint, *, diffusers_resnet_prefix, resnet_prefix):
    diffusers_checkpoint = {
        f"{diffusers_resnet_prefix}.norm1.weight": checkpoint[f"{resnet_prefix}.in_layers.0.weight"],
        f"{diffusers_resnet_prefix}.norm1.bias": checkpoint[f"{resnet_prefix}.in_layers.0.bias"],
        f"{diffusers_resnet_prefix}.conv1.weight": checkpoint[f"{resnet_prefix}.in_layers.2.weight"],
        f"{diffusers_resnet_prefix}.conv1.bias": checkpoint[f"{resnet_prefix}.in_layers.2.bias"],
        f"{diffusers_resnet_prefix}.time_emb_proj.weight": checkpoint[f"{resnet_prefix}.emb_layers.1.weight"],
        f"{diffusers_resnet_prefix}.time_emb_proj.bias": checkpoint[f"{resnet_prefix}.emb_layers.1.bias"],
        f"{diffusers_resnet_prefix}.norm2.weight": checkpoint[f"{resnet_prefix}.out_layers.0.weight"],
        f"{diffusers_resnet_prefix}.norm2.bias": checkpoint[f"{resnet_prefix}.out_layers.0.bias"],
        f"{diffusers_resnet_prefix}.conv2.weight": checkpoint[f"{resnet_prefix}.out_layers.3.weight"],
        f"{diffusers_resnet_prefix}.conv2.bias": checkpoint[f"{resnet_prefix}.out_layers.3.bias"],
    }

    skip_connection_prefix = f"{resnet_prefix}.skip_connection"

    if f"{skip_connection_prefix}.weight" in checkpoint:
        diffusers_checkpoint.update(
            {
                f"{diffusers_resnet_prefix}.conv_shortcut.weight": checkpoint[f"{skip_connection_prefix}.weight"],
                f"{diffusers_resnet_prefix}.conv_shortcut.bias": checkpoint[f"{skip_connection_prefix}.bias"],
            }
        )

    return diffusers_checkpoint


def attention_to_diffusers_checkpoint(checkpoint, *, diffusers_attention_prefix, attention_prefix, num_head_channels):
    diffusers_checkpoint = {}

    # <original>.norm -> <diffusers>.group_norm
    diffusers_checkpoint.update(
        {
            f"{diffusers_attention_prefix}.group_norm.weight": checkpoint[f"{attention_prefix}.norm.weight"],
            f"{diffusers_attention_prefix}.group_norm.bias": checkpoint[f"{attention_prefix}.norm.bias"],
        }
    )

    # <original>.qkv -> <diffusers>.{query, key, value}
    [q_weight, k_weight, v_weight], [q_bias, k_bias, v_bias] = split_attentions(
        weight=checkpoint[f"{attention_prefix}.qkv.weight"][:, :, 0],
        bias=checkpoint[f"{attention_prefix}.qkv.bias"],
        split=3,
        chunk_size=num_head_channels,
    )

    diffusers_checkpoint.update(
        {
            f"{diffusers_attention_prefix}.to_q.weight": q_weight,
            f"{diffusers_attention_prefix}.to_q.bias": q_bias,
            f"{diffusers_attention_prefix}.to_k.weight": k_weight,
            f"{diffusers_attention_prefix}.to_k.bias": k_bias,
            f"{diffusers_attention_prefix}.to_v.weight": v_weight,
            f"{diffusers_attention_prefix}.to_v.bias": v_bias,
        }
    )

    # <original>.encoder_kv -> <diffusers>.{context_key, context_value}
    [encoder_k_weight, encoder_v_weight], [encoder_k_bias, encoder_v_bias] = split_attentions(
        weight=checkpoint[f"{attention_prefix}.encoder_kv.weight"][:, :, 0],
        bias=checkpoint[f"{attention_prefix}.encoder_kv.bias"],
        split=2,
        chunk_size=num_head_channels,
    )

    diffusers_checkpoint.update(
        {
            f"{diffusers_attention_prefix}.add_k_proj.weight": encoder_k_weight,
            f"{diffusers_attention_prefix}.add_k_proj.bias": encoder_k_bias,
            f"{diffusers_attention_prefix}.add_v_proj.weight": encoder_v_weight,
            f"{diffusers_attention_prefix}.add_v_proj.bias": encoder_v_bias,
        }
    )

    # <original>.proj_out (1d conv) -> <diffusers>.proj_attn (linear)
    diffusers_checkpoint.update(
        {
            f"{diffusers_attention_prefix}.to_out.0.weight": checkpoint[f"{attention_prefix}.proj_out.weight"][
                :, :, 0
            ],
            f"{diffusers_attention_prefix}.to_out.0.bias": checkpoint[f"{attention_prefix}.proj_out.bias"],
        }
    )

    return diffusers_checkpoint


# TODO maybe document and/or can do more efficiently (build indices in for loop and extract once for each split?)
def split_attentions(*, weight, bias, split, chunk_size):
    weights = [None] * split
    biases = [None] * split

    weights_biases_idx = 0

    for starting_row_index in range(0, weight.shape[0], chunk_size):
        row_indices = torch.arange(starting_row_index, starting_row_index + chunk_size)

        weight_rows = weight[row_indices, :]
        bias_rows = bias[row_indices]

        if weights[weights_biases_idx] is None:
            assert weights[weights_biases_idx] is None
            weights[weights_biases_idx] = weight_rows
            biases[weights_biases_idx] = bias_rows
        else:
            assert weights[weights_biases_idx] is not None
            weights[weights_biases_idx] = torch.concat([weights[weights_biases_idx], weight_rows])
            biases[weights_biases_idx] = torch.concat([biases[weights_biases_idx], bias_rows])

        weights_biases_idx = (weights_biases_idx + 1) % split

    return weights, biases


# done unet utils


def prior(*, args, checkpoint_map_location):
    print("loading prior")

    prior_checkpoint = torch.load(args.prior_checkpoint_path, map_location=checkpoint_map_location)

    clip_stats_checkpoint = torch.load(args.clip_stat_path, map_location=checkpoint_map_location)

    prior_model = prior_model_from_original_config()

    prior_diffusers_checkpoint = prior_original_checkpoint_to_diffusers_checkpoint(
        prior_model, prior_checkpoint, clip_stats_checkpoint
    )

    del prior_checkpoint
    del clip_stats_checkpoint

    load_checkpoint_to_model(prior_diffusers_checkpoint, prior_model, strict=True)

    print("done loading prior")

    return prior_model


def text2img(*, args, checkpoint_map_location):
    print("loading text2img")

    text2img_checkpoint = torch.load(args.text2img_checkpoint_path, map_location=checkpoint_map_location)

    unet_model = unet_model_from_original_config()

    unet_diffusers_checkpoint = unet_original_checkpoint_to_diffusers_checkpoint(unet_model, text2img_checkpoint)

    del text2img_checkpoint

    load_checkpoint_to_model(unet_diffusers_checkpoint, unet_model, strict=True)

    print("done loading text2img")

    return unet_model


def inpaint_text2img(*, args, checkpoint_map_location):
    print("loading inpaint text2img")

    inpaint_text2img_checkpoint = torch.load(
        args.inpaint_text2img_checkpoint_path, map_location=checkpoint_map_location
    )

    inpaint_unet_model = inpaint_unet_model_from_original_config()

    inpaint_unet_diffusers_checkpoint = inpaint_unet_original_checkpoint_to_diffusers_checkpoint(
        inpaint_unet_model, inpaint_text2img_checkpoint
    )

    del inpaint_text2img_checkpoint

    load_checkpoint_to_model(inpaint_unet_diffusers_checkpoint, inpaint_unet_model, strict=True)

    print("done loading inpaint text2img")

    return inpaint_unet_model


# movq

MOVQ_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "AttnDownEncoderBlock2D"),
    "up_block_types": ("AttnUpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    "num_vq_embeddings": 16384,
    "block_out_channels": (128, 256, 256, 512),
    "vq_embed_dim": 4,
    "layers_per_block": 2,
    "norm_type": "spatial",
}


def movq_model_from_original_config():
    movq = VQModel(**MOVQ_CONFIG)
    return movq


def movq_encoder_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    # conv_in
    diffusers_checkpoint.update(
        {
            "encoder.conv_in.weight": checkpoint["encoder.conv_in.weight"],
            "encoder.conv_in.bias": checkpoint["encoder.conv_in.bias"],
        }
    )

    # down_blocks
    for down_block_idx, down_block in enumerate(model.encoder.down_blocks):
        diffusers_down_block_prefix = f"encoder.down_blocks.{down_block_idx}"
        down_block_prefix = f"encoder.down.{down_block_idx}"

        # resnets
        for resnet_idx, resnet in enumerate(down_block.resnets):
            diffusers_resnet_prefix = f"{diffusers_down_block_prefix}.resnets.{resnet_idx}"
            resnet_prefix = f"{down_block_prefix}.block.{resnet_idx}"

            diffusers_checkpoint.update(
                movq_resnet_to_diffusers_checkpoint(
                    resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
                )
            )

        # downsample

        # do not include the downsample when on the last down block
        # There is no downsample on the last down block
        if down_block_idx != len(model.encoder.down_blocks) - 1:
            # There's a single downsample in the original checkpoint but a list of downsamples
            # in the diffusers model.
            diffusers_downsample_prefix = f"{diffusers_down_block_prefix}.downsamplers.0.conv"
            downsample_prefix = f"{down_block_prefix}.downsample.conv"
            diffusers_checkpoint.update(
                {
                    f"{diffusers_downsample_prefix}.weight": checkpoint[f"{downsample_prefix}.weight"],
                    f"{diffusers_downsample_prefix}.bias": checkpoint[f"{downsample_prefix}.bias"],
                }
            )

        # attentions

        if hasattr(down_block, "attentions"):
            for attention_idx, _ in enumerate(down_block.attentions):
                diffusers_attention_prefix = f"{diffusers_down_block_prefix}.attentions.{attention_idx}"
                attention_prefix = f"{down_block_prefix}.attn.{attention_idx}"
                diffusers_checkpoint.update(
                    movq_attention_to_diffusers_checkpoint(
                        checkpoint,
                        diffusers_attention_prefix=diffusers_attention_prefix,
                        attention_prefix=attention_prefix,
                    )
                )

    # mid block

    # mid block attentions

    # There is a single hardcoded attention block in the middle of the VQ-diffusion encoder
    diffusers_attention_prefix = "encoder.mid_block.attentions.0"
    attention_prefix = "encoder.mid.attn_1"
    diffusers_checkpoint.update(
        movq_attention_to_diffusers_checkpoint(
            checkpoint, diffusers_attention_prefix=diffusers_attention_prefix, attention_prefix=attention_prefix
        )
    )

    # mid block resnets

    for diffusers_resnet_idx, resnet in enumerate(model.encoder.mid_block.resnets):
        diffusers_resnet_prefix = f"encoder.mid_block.resnets.{diffusers_resnet_idx}"

        # the hardcoded prefixes to `block_` are 1 and 2
        orig_resnet_idx = diffusers_resnet_idx + 1
        # There are two hardcoded resnets in the middle of the VQ-diffusion encoder
        resnet_prefix = f"encoder.mid.block_{orig_resnet_idx}"

        diffusers_checkpoint.update(
            movq_resnet_to_diffusers_checkpoint(
                resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
            )
        )

    diffusers_checkpoint.update(
        {
            # conv_norm_out
            "encoder.conv_norm_out.weight": checkpoint["encoder.norm_out.weight"],
            "encoder.conv_norm_out.bias": checkpoint["encoder.norm_out.bias"],
            # conv_out
            "encoder.conv_out.weight": checkpoint["encoder.conv_out.weight"],
            "encoder.conv_out.bias": checkpoint["encoder.conv_out.bias"],
        }
    )

    return diffusers_checkpoint


def movq_decoder_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    # conv in
    diffusers_checkpoint.update(
        {
            "decoder.conv_in.weight": checkpoint["decoder.conv_in.weight"],
            "decoder.conv_in.bias": checkpoint["decoder.conv_in.bias"],
        }
    )

    # up_blocks

    for diffusers_up_block_idx, up_block in enumerate(model.decoder.up_blocks):
        # up_blocks are stored in reverse order in the VQ-diffusion checkpoint
        orig_up_block_idx = len(model.decoder.up_blocks) - 1 - diffusers_up_block_idx

        diffusers_up_block_prefix = f"decoder.up_blocks.{diffusers_up_block_idx}"
        up_block_prefix = f"decoder.up.{orig_up_block_idx}"

        # resnets
        for resnet_idx, resnet in enumerate(up_block.resnets):
            diffusers_resnet_prefix = f"{diffusers_up_block_prefix}.resnets.{resnet_idx}"
            resnet_prefix = f"{up_block_prefix}.block.{resnet_idx}"

            diffusers_checkpoint.update(
                movq_resnet_to_diffusers_checkpoint_spatial_norm(
                    resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
                )
            )

        # upsample

        # there is no up sample on the last up block
        if diffusers_up_block_idx != len(model.decoder.up_blocks) - 1:
            # There's a single upsample in the VQ-diffusion checkpoint but a list of downsamples
            # in the diffusers model.
            diffusers_downsample_prefix = f"{diffusers_up_block_prefix}.upsamplers.0.conv"
            downsample_prefix = f"{up_block_prefix}.upsample.conv"
            diffusers_checkpoint.update(
                {
                    f"{diffusers_downsample_prefix}.weight": checkpoint[f"{downsample_prefix}.weight"],
                    f"{diffusers_downsample_prefix}.bias": checkpoint[f"{downsample_prefix}.bias"],
                }
            )

        # attentions

        if hasattr(up_block, "attentions"):
            for attention_idx, _ in enumerate(up_block.attentions):
                diffusers_attention_prefix = f"{diffusers_up_block_prefix}.attentions.{attention_idx}"
                attention_prefix = f"{up_block_prefix}.attn.{attention_idx}"
                diffusers_checkpoint.update(
                    movq_attention_to_diffusers_checkpoint_spatial_norm(
                        checkpoint,
                        diffusers_attention_prefix=diffusers_attention_prefix,
                        attention_prefix=attention_prefix,
                    )
                )

    # mid block

    # mid block attentions

    # There is a single hardcoded attention block in the middle of the VQ-diffusion decoder
    diffusers_attention_prefix = "decoder.mid_block.attentions.0"
    attention_prefix = "decoder.mid.attn_1"
    diffusers_checkpoint.update(
        movq_attention_to_diffusers_checkpoint_spatial_norm(
            checkpoint, diffusers_attention_prefix=diffusers_attention_prefix, attention_prefix=attention_prefix
        )
    )

    # mid block resnets

    for diffusers_resnet_idx, resnet in enumerate(model.encoder.mid_block.resnets):
        diffusers_resnet_prefix = f"decoder.mid_block.resnets.{diffusers_resnet_idx}"

        # the hardcoded prefixes to `block_` are 1 and 2
        orig_resnet_idx = diffusers_resnet_idx + 1
        # There are two hardcoded resnets in the middle of the VQ-diffusion decoder
        resnet_prefix = f"decoder.mid.block_{orig_resnet_idx}"

        diffusers_checkpoint.update(
            movq_resnet_to_diffusers_checkpoint_spatial_norm(
                resnet, checkpoint, diffusers_resnet_prefix=diffusers_resnet_prefix, resnet_prefix=resnet_prefix
            )
        )

    diffusers_checkpoint.update(
        {
            # conv_norm_out
            "decoder.conv_norm_out.norm_layer.weight": checkpoint["decoder.norm_out.norm_layer.weight"],
            "decoder.conv_norm_out.norm_layer.bias": checkpoint["decoder.norm_out.norm_layer.bias"],
            "decoder.conv_norm_out.conv_y.weight": checkpoint["decoder.norm_out.conv_y.weight"],
            "decoder.conv_norm_out.conv_y.bias": checkpoint["decoder.norm_out.conv_y.bias"],
            "decoder.conv_norm_out.conv_b.weight": checkpoint["decoder.norm_out.conv_b.weight"],
            "decoder.conv_norm_out.conv_b.bias": checkpoint["decoder.norm_out.conv_b.bias"],
            # conv_out
            "decoder.conv_out.weight": checkpoint["decoder.conv_out.weight"],
            "decoder.conv_out.bias": checkpoint["decoder.conv_out.bias"],
        }
    )

    return diffusers_checkpoint


def movq_resnet_to_diffusers_checkpoint(resnet, checkpoint, *, diffusers_resnet_prefix, resnet_prefix):
    rv = {
        # norm1
        f"{diffusers_resnet_prefix}.norm1.weight": checkpoint[f"{resnet_prefix}.norm1.weight"],
        f"{diffusers_resnet_prefix}.norm1.bias": checkpoint[f"{resnet_prefix}.norm1.bias"],
        # conv1
        f"{diffusers_resnet_prefix}.conv1.weight": checkpoint[f"{resnet_prefix}.conv1.weight"],
        f"{diffusers_resnet_prefix}.conv1.bias": checkpoint[f"{resnet_prefix}.conv1.bias"],
        # norm2
        f"{diffusers_resnet_prefix}.norm2.weight": checkpoint[f"{resnet_prefix}.norm2.weight"],
        f"{diffusers_resnet_prefix}.norm2.bias": checkpoint[f"{resnet_prefix}.norm2.bias"],
        # conv2
        f"{diffusers_resnet_prefix}.conv2.weight": checkpoint[f"{resnet_prefix}.conv2.weight"],
        f"{diffusers_resnet_prefix}.conv2.bias": checkpoint[f"{resnet_prefix}.conv2.bias"],
    }

    if resnet.conv_shortcut is not None:
        rv.update(
            {
                f"{diffusers_resnet_prefix}.conv_shortcut.weight": checkpoint[f"{resnet_prefix}.nin_shortcut.weight"],
                f"{diffusers_resnet_prefix}.conv_shortcut.bias": checkpoint[f"{resnet_prefix}.nin_shortcut.bias"],
            }
        )

    return rv


def movq_resnet_to_diffusers_checkpoint_spatial_norm(resnet, checkpoint, *, diffusers_resnet_prefix, resnet_prefix):
    rv = {
        # norm1
        f"{diffusers_resnet_prefix}.norm1.norm_layer.weight": checkpoint[f"{resnet_prefix}.norm1.norm_layer.weight"],
        f"{diffusers_resnet_prefix}.norm1.norm_layer.bias": checkpoint[f"{resnet_prefix}.norm1.norm_layer.bias"],
        f"{diffusers_resnet_prefix}.norm1.conv_y.weight": checkpoint[f"{resnet_prefix}.norm1.conv_y.weight"],
        f"{diffusers_resnet_prefix}.norm1.conv_y.bias": checkpoint[f"{resnet_prefix}.norm1.conv_y.bias"],
        f"{diffusers_resnet_prefix}.norm1.conv_b.weight": checkpoint[f"{resnet_prefix}.norm1.conv_b.weight"],
        f"{diffusers_resnet_prefix}.norm1.conv_b.bias": checkpoint[f"{resnet_prefix}.norm1.conv_b.bias"],
        # conv1
        f"{diffusers_resnet_prefix}.conv1.weight": checkpoint[f"{resnet_prefix}.conv1.weight"],
        f"{diffusers_resnet_prefix}.conv1.bias": checkpoint[f"{resnet_prefix}.conv1.bias"],
        # norm2
        f"{diffusers_resnet_prefix}.norm2.norm_layer.weight": checkpoint[f"{resnet_prefix}.norm2.norm_layer.weight"],
        f"{diffusers_resnet_prefix}.norm2.norm_layer.bias": checkpoint[f"{resnet_prefix}.norm2.norm_layer.bias"],
        f"{diffusers_resnet_prefix}.norm2.conv_y.weight": checkpoint[f"{resnet_prefix}.norm2.conv_y.weight"],
        f"{diffusers_resnet_prefix}.norm2.conv_y.bias": checkpoint[f"{resnet_prefix}.norm2.conv_y.bias"],
        f"{diffusers_resnet_prefix}.norm2.conv_b.weight": checkpoint[f"{resnet_prefix}.norm2.conv_b.weight"],
        f"{diffusers_resnet_prefix}.norm2.conv_b.bias": checkpoint[f"{resnet_prefix}.norm2.conv_b.bias"],
        # conv2
        f"{diffusers_resnet_prefix}.conv2.weight": checkpoint[f"{resnet_prefix}.conv2.weight"],
        f"{diffusers_resnet_prefix}.conv2.bias": checkpoint[f"{resnet_prefix}.conv2.bias"],
    }

    if resnet.conv_shortcut is not None:
        rv.update(
            {
                f"{diffusers_resnet_prefix}.conv_shortcut.weight": checkpoint[f"{resnet_prefix}.nin_shortcut.weight"],
                f"{diffusers_resnet_prefix}.conv_shortcut.bias": checkpoint[f"{resnet_prefix}.nin_shortcut.bias"],
            }
        )

    return rv


def movq_attention_to_diffusers_checkpoint(checkpoint, *, diffusers_attention_prefix, attention_prefix):
    return {
        # norm
        f"{diffusers_attention_prefix}.group_norm.weight": checkpoint[f"{attention_prefix}.norm.weight"],
        f"{diffusers_attention_prefix}.group_norm.bias": checkpoint[f"{attention_prefix}.norm.bias"],
        # query
        f"{diffusers_attention_prefix}.to_q.weight": checkpoint[f"{attention_prefix}.q.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_q.bias": checkpoint[f"{attention_prefix}.q.bias"],
        # key
        f"{diffusers_attention_prefix}.to_k.weight": checkpoint[f"{attention_prefix}.k.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_k.bias": checkpoint[f"{attention_prefix}.k.bias"],
        # value
        f"{diffusers_attention_prefix}.to_v.weight": checkpoint[f"{attention_prefix}.v.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_v.bias": checkpoint[f"{attention_prefix}.v.bias"],
        # proj_attn
        f"{diffusers_attention_prefix}.to_out.0.weight": checkpoint[f"{attention_prefix}.proj_out.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_out.0.bias": checkpoint[f"{attention_prefix}.proj_out.bias"],
    }


def movq_attention_to_diffusers_checkpoint_spatial_norm(checkpoint, *, diffusers_attention_prefix, attention_prefix):
    return {
        # norm
        f"{diffusers_attention_prefix}.spatial_norm.norm_layer.weight": checkpoint[
            f"{attention_prefix}.norm.norm_layer.weight"
        ],
        f"{diffusers_attention_prefix}.spatial_norm.norm_layer.bias": checkpoint[
            f"{attention_prefix}.norm.norm_layer.bias"
        ],
        f"{diffusers_attention_prefix}.spatial_norm.conv_y.weight": checkpoint[
            f"{attention_prefix}.norm.conv_y.weight"
        ],
        f"{diffusers_attention_prefix}.spatial_norm.conv_y.bias": checkpoint[f"{attention_prefix}.norm.conv_y.bias"],
        f"{diffusers_attention_prefix}.spatial_norm.conv_b.weight": checkpoint[
            f"{attention_prefix}.norm.conv_b.weight"
        ],
        f"{diffusers_attention_prefix}.spatial_norm.conv_b.bias": checkpoint[f"{attention_prefix}.norm.conv_b.bias"],
        # query
        f"{diffusers_attention_prefix}.to_q.weight": checkpoint[f"{attention_prefix}.q.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_q.bias": checkpoint[f"{attention_prefix}.q.bias"],
        # key
        f"{diffusers_attention_prefix}.to_k.weight": checkpoint[f"{attention_prefix}.k.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_k.bias": checkpoint[f"{attention_prefix}.k.bias"],
        # value
        f"{diffusers_attention_prefix}.to_v.weight": checkpoint[f"{attention_prefix}.v.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_v.bias": checkpoint[f"{attention_prefix}.v.bias"],
        # proj_attn
        f"{diffusers_attention_prefix}.to_out.0.weight": checkpoint[f"{attention_prefix}.proj_out.weight"][:, :, 0, 0],
        f"{diffusers_attention_prefix}.to_out.0.bias": checkpoint[f"{attention_prefix}.proj_out.bias"],
    }


def movq_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update(movq_encoder_to_diffusers_checkpoint(model, checkpoint))

    # quant_conv

    diffusers_checkpoint.update(
        {
            "quant_conv.weight": checkpoint["quant_conv.weight"],
            "quant_conv.bias": checkpoint["quant_conv.bias"],
        }
    )

    # quantize
    diffusers_checkpoint.update({"quantize.embedding.weight": checkpoint["quantize.embedding.weight"]})

    # post_quant_conv
    diffusers_checkpoint.update(
        {
            "post_quant_conv.weight": checkpoint["post_quant_conv.weight"],
            "post_quant_conv.bias": checkpoint["post_quant_conv.bias"],
        }
    )

    # decoder
    diffusers_checkpoint.update(movq_decoder_to_diffusers_checkpoint(model, checkpoint))

    return diffusers_checkpoint


def movq(*, args, checkpoint_map_location):
    print("loading movq")

    movq_checkpoint = torch.load(args.movq_checkpoint_path, map_location=checkpoint_map_location)

    movq_model = movq_model_from_original_config()

    movq_diffusers_checkpoint = movq_original_checkpoint_to_diffusers_checkpoint(movq_model, movq_checkpoint)

    del movq_checkpoint

    load_checkpoint_to_model(movq_diffusers_checkpoint, movq_model, strict=True)

    print("done loading movq")

    return movq_model


def load_checkpoint_to_model(checkpoint, model, strict=False):
    with tempfile.NamedTemporaryFile(delete=False) as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        if strict:
            model.load_state_dict(torch.load(file.name), strict=True)
        else:
            load_checkpoint_and_dispatch(model, file.name, device_map="auto")
    os.remove(file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--prior_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the prior checkpoint to convert.",
    )
    parser.add_argument(
        "--clip_stat_path",
        default=None,
        type=str,
        required=False,
        help="Path to the clip stats checkpoint to convert.",
    )
    parser.add_argument(
        "--text2img_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--movq_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--inpaint_text2img_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the inpaint text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--checkpoint_load_device",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading checkpoints.",
    )

    parser.add_argument(
        "--debug",
        default=None,
        type=str,
        required=False,
        help="Only run a specific stage of the convert script. Used for debugging",
    )

    args = parser.parse_args()

    print(f"loading checkpoints to {args.checkpoint_load_device}")

    checkpoint_map_location = torch.device(args.checkpoint_load_device)

    if args.debug is not None:
        print(f"debug: only executing {args.debug}")

    if args.debug is None:
        print("to-do")
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
        prior_model.save_pretrained(args.dump_path)
    elif args.debug == "text2img":
        unet_model = text2img(args=args, checkpoint_map_location=checkpoint_map_location)
        unet_model.save_pretrained(f"{args.dump_path}/unet")
    elif args.debug == "inpaint_text2img":
        inpaint_unet_model = inpaint_text2img(args=args, checkpoint_map_location=checkpoint_map_location)
        inpaint_unet_model.save_pretrained(f"{args.dump_path}/inpaint_unet")
    elif args.debug == "decoder":
        decoder = movq(args=args, checkpoint_map_location=checkpoint_map_location)
        decoder.save_pretrained(f"{args.dump_path}/decoder")
    else:
        raise ValueError(f"unknown debug value : {args.debug}")
