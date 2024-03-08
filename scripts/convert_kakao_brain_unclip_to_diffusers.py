import argparse
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import UnCLIPPipeline, UNet2DConditionModel, UNet2DModel
from diffusers.models.transformers.prior_transformer import PriorTransformer
from diffusers.pipelines.unclip.text_proj import UnCLIPTextProjModel
from diffusers.schedulers.scheduling_unclip import UnCLIPScheduler


r"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/efdf6206d8ed593961593dc029a8affa/decoder-ckpt-step%3D01000000-of-01000000.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/4226b831ae0279020d134281f3c31590/improved-sr-ckpt-step%3D1.2M.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/85626483eaca9f581e2a78d31ff905ca/prior-ckpt-step%3D01000000-of-01000000.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th
```

Convert the model:
```sh
$ python scripts/convert_kakao_brain_unclip_to_diffusers.py \
      --decoder_checkpoint_path ./decoder-ckpt-step\=01000000-of-01000000.ckpt \
      --super_res_unet_checkpoint_path ./improved-sr-ckpt-step\=1.2M.ckpt \
      --prior_checkpoint_path ./prior-ckpt-step\=01000000-of-01000000.ckpt \
      --clip_stat_path ./ViT-L-14_stats.th \
      --dump_path <path where to save model>
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


# decoder

DECODER_ORIGINAL_PREFIX = "model"

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.
DECODER_CONFIG = {
    "sample_size": 64,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ),
    "up_block_types": (
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 3,
    "out_channels": 6,
    "cross_attention_dim": 1536,
    "class_embed_type": "identity",
    "attention_head_dim": 64,
    "resnet_time_scale_shift": "scale_shift",
}


def decoder_model_from_original_config():
    model = UNet2DConditionModel(**DECODER_CONFIG)

    return model


def decoder_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    original_unet_prefix = DECODER_ORIGINAL_PREFIX
    num_head_channels = DECODER_CONFIG["attention_head_dim"]

    diffusers_checkpoint.update(unet_time_embeddings(checkpoint, original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_in(checkpoint, original_unet_prefix))

    # <original>.input_blocks -> <diffusers>.down_blocks

    original_down_block_idx = 1

    for diffusers_down_block_idx in range(len(model.down_blocks)):
        checkpoint_update, num_original_down_blocks = unet_downblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            diffusers_down_block_idx=diffusers_down_block_idx,
            original_down_block_idx=original_down_block_idx,
            original_unet_prefix=original_unet_prefix,
            num_head_channels=num_head_channels,
        )

        original_down_block_idx += num_original_down_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.input_blocks -> <diffusers>.down_blocks

    diffusers_checkpoint.update(
        unet_midblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            original_unet_prefix=original_unet_prefix,
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
            original_unet_prefix=original_unet_prefix,
            num_head_channels=num_head_channels,
        )

        original_up_block_idx += num_original_up_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.output_blocks -> <diffusers>.up_blocks

    diffusers_checkpoint.update(unet_conv_norm_out(checkpoint, original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_out(checkpoint, original_unet_prefix))

    return diffusers_checkpoint


# done decoder

# text proj


def text_proj_from_original_config():
    # From the conditional unet constructor where the dimension of the projected time embeddings is
    # constructed
    time_embed_dim = DECODER_CONFIG["block_out_channels"][0] * 4

    cross_attention_dim = DECODER_CONFIG["cross_attention_dim"]

    model = UnCLIPTextProjModel(time_embed_dim=time_embed_dim, cross_attention_dim=cross_attention_dim)

    return model


# Note that the input checkpoint is the original decoder checkpoint
def text_proj_original_checkpoint_to_diffusers_checkpoint(checkpoint):
    diffusers_checkpoint = {
        # <original>.text_seq_proj.0 -> <diffusers>.encoder_hidden_states_proj
        "encoder_hidden_states_proj.weight": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.text_seq_proj.0.weight"],
        "encoder_hidden_states_proj.bias": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.text_seq_proj.0.bias"],
        # <original>.text_seq_proj.1 -> <diffusers>.text_encoder_hidden_states_norm
        "text_encoder_hidden_states_norm.weight": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.text_seq_proj.1.weight"],
        "text_encoder_hidden_states_norm.bias": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.text_seq_proj.1.bias"],
        # <original>.clip_tok_proj -> <diffusers>.clip_extra_context_tokens_proj
        "clip_extra_context_tokens_proj.weight": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.clip_tok_proj.weight"],
        "clip_extra_context_tokens_proj.bias": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.clip_tok_proj.bias"],
        # <original>.text_feat_proj -> <diffusers>.embedding_proj
        "embedding_proj.weight": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.text_feat_proj.weight"],
        "embedding_proj.bias": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.text_feat_proj.bias"],
        # <original>.cf_param -> <diffusers>.learned_classifier_free_guidance_embeddings
        "learned_classifier_free_guidance_embeddings": checkpoint[f"{DECODER_ORIGINAL_PREFIX}.cf_param"],
        # <original>.clip_emb -> <diffusers>.clip_image_embeddings_project_to_time_embeddings
        "clip_image_embeddings_project_to_time_embeddings.weight": checkpoint[
            f"{DECODER_ORIGINAL_PREFIX}.clip_emb.weight"
        ],
        "clip_image_embeddings_project_to_time_embeddings.bias": checkpoint[
            f"{DECODER_ORIGINAL_PREFIX}.clip_emb.bias"
        ],
    }

    return diffusers_checkpoint


# done text proj

# super res unet first steps

SUPER_RES_UNET_FIRST_STEPS_PREFIX = "model_first_steps"

SUPER_RES_UNET_FIRST_STEPS_CONFIG = {
    "sample_size": 256,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
    ),
    "up_block_types": (
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 6,
    "out_channels": 3,
    "add_attention": False,
}


def super_res_unet_first_steps_model_from_original_config():
    model = UNet2DModel(**SUPER_RES_UNET_FIRST_STEPS_CONFIG)

    return model


def super_res_unet_first_steps_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    original_unet_prefix = SUPER_RES_UNET_FIRST_STEPS_PREFIX

    diffusers_checkpoint.update(unet_time_embeddings(checkpoint, original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_in(checkpoint, original_unet_prefix))

    # <original>.input_blocks -> <diffusers>.down_blocks

    original_down_block_idx = 1

    for diffusers_down_block_idx in range(len(model.down_blocks)):
        checkpoint_update, num_original_down_blocks = unet_downblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            diffusers_down_block_idx=diffusers_down_block_idx,
            original_down_block_idx=original_down_block_idx,
            original_unet_prefix=original_unet_prefix,
            num_head_channels=None,
        )

        original_down_block_idx += num_original_down_blocks

        diffusers_checkpoint.update(checkpoint_update)

    diffusers_checkpoint.update(
        unet_midblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            original_unet_prefix=original_unet_prefix,
            num_head_channels=None,
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
            original_unet_prefix=original_unet_prefix,
            num_head_channels=None,
        )

        original_up_block_idx += num_original_up_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.output_blocks -> <diffusers>.up_blocks

    diffusers_checkpoint.update(unet_conv_norm_out(checkpoint, original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_out(checkpoint, original_unet_prefix))

    return diffusers_checkpoint


# done super res unet first steps

# super res unet last step

SUPER_RES_UNET_LAST_STEP_PREFIX = "model_last_step"

SUPER_RES_UNET_LAST_STEP_CONFIG = {
    "sample_size": 256,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
    ),
    "up_block_types": (
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 6,
    "out_channels": 3,
    "add_attention": False,
}


def super_res_unet_last_step_model_from_original_config():
    model = UNet2DModel(**SUPER_RES_UNET_LAST_STEP_CONFIG)

    return model


def super_res_unet_last_step_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    original_unet_prefix = SUPER_RES_UNET_LAST_STEP_PREFIX

    diffusers_checkpoint.update(unet_time_embeddings(checkpoint, original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_in(checkpoint, original_unet_prefix))

    # <original>.input_blocks -> <diffusers>.down_blocks

    original_down_block_idx = 1

    for diffusers_down_block_idx in range(len(model.down_blocks)):
        checkpoint_update, num_original_down_blocks = unet_downblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            diffusers_down_block_idx=diffusers_down_block_idx,
            original_down_block_idx=original_down_block_idx,
            original_unet_prefix=original_unet_prefix,
            num_head_channels=None,
        )

        original_down_block_idx += num_original_down_blocks

        diffusers_checkpoint.update(checkpoint_update)

    diffusers_checkpoint.update(
        unet_midblock_to_diffusers_checkpoint(
            model,
            checkpoint,
            original_unet_prefix=original_unet_prefix,
            num_head_channels=None,
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
            original_unet_prefix=original_unet_prefix,
            num_head_channels=None,
        )

        original_up_block_idx += num_original_up_blocks

        diffusers_checkpoint.update(checkpoint_update)

    # done <original>.output_blocks -> <diffusers>.up_blocks

    diffusers_checkpoint.update(unet_conv_norm_out(checkpoint, original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_out(checkpoint, original_unet_prefix))

    return diffusers_checkpoint


# done super res unet last step


# unet utils


# <original>.time_embed -> <diffusers>.time_embedding
def unet_time_embeddings(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "time_embedding.linear_1.weight": checkpoint[f"{original_unet_prefix}.time_embed.0.weight"],
            "time_embedding.linear_1.bias": checkpoint[f"{original_unet_prefix}.time_embed.0.bias"],
            "time_embedding.linear_2.weight": checkpoint[f"{original_unet_prefix}.time_embed.2.weight"],
            "time_embedding.linear_2.bias": checkpoint[f"{original_unet_prefix}.time_embed.2.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.input_blocks.0 -> <diffusers>.conv_in
def unet_conv_in(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "conv_in.weight": checkpoint[f"{original_unet_prefix}.input_blocks.0.0.weight"],
            "conv_in.bias": checkpoint[f"{original_unet_prefix}.input_blocks.0.0.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.out.0 -> <diffusers>.conv_norm_out
def unet_conv_norm_out(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "conv_norm_out.weight": checkpoint[f"{original_unet_prefix}.out.0.weight"],
            "conv_norm_out.bias": checkpoint[f"{original_unet_prefix}.out.0.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.out.2 -> <diffusers>.conv_out
def unet_conv_out(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update(
        {
            "conv_out.weight": checkpoint[f"{original_unet_prefix}.out.2.weight"],
            "conv_out.bias": checkpoint[f"{original_unet_prefix}.out.2.bias"],
        }
    )

    return diffusers_checkpoint


# <original>.input_blocks -> <diffusers>.down_blocks
def unet_downblock_to_diffusers_checkpoint(
    model, checkpoint, *, diffusers_down_block_idx, original_down_block_idx, original_unet_prefix, num_head_channels
):
    diffusers_checkpoint = {}

    diffusers_resnet_prefix = f"down_blocks.{diffusers_down_block_idx}.resnets"
    original_down_block_prefix = f"{original_unet_prefix}.input_blocks"

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
def unet_midblock_to_diffusers_checkpoint(model, checkpoint, *, original_unet_prefix, num_head_channels):
    diffusers_checkpoint = {}

    # block 0

    original_block_idx = 0

    diffusers_checkpoint.update(
        resnet_to_diffusers_checkpoint(
            checkpoint,
            diffusers_resnet_prefix="mid_block.resnets.0",
            resnet_prefix=f"{original_unet_prefix}.middle_block.{original_block_idx}",
        )
    )

    original_block_idx += 1

    # optional block 1

    if hasattr(model.mid_block, "attentions") and model.mid_block.attentions[0] is not None:
        diffusers_checkpoint.update(
            attention_to_diffusers_checkpoint(
                checkpoint,
                diffusers_attention_prefix="mid_block.attentions.0",
                attention_prefix=f"{original_unet_prefix}.middle_block.{original_block_idx}",
                num_head_channels=num_head_channels,
            )
        )
        original_block_idx += 1

    # block 1 or block 2

    diffusers_checkpoint.update(
        resnet_to_diffusers_checkpoint(
            checkpoint,
            diffusers_resnet_prefix="mid_block.resnets.1",
            resnet_prefix=f"{original_unet_prefix}.middle_block.{original_block_idx}",
        )
    )

    return diffusers_checkpoint


# <original>.output_blocks -> <diffusers>.up_blocks
def unet_upblock_to_diffusers_checkpoint(
    model, checkpoint, *, diffusers_up_block_idx, original_up_block_idx, original_unet_prefix, num_head_channels
):
    diffusers_checkpoint = {}

    diffusers_resnet_prefix = f"up_blocks.{diffusers_up_block_idx}.resnets"
    original_up_block_prefix = f"{original_unet_prefix}.output_blocks"

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


# Driver functions


def text_encoder():
    print("loading CLIP text encoder")

    clip_name = "openai/clip-vit-large-patch14"

    # sets pad_value to 0
    pad_token = "!"

    tokenizer_model = CLIPTokenizer.from_pretrained(clip_name, pad_token=pad_token, device_map="auto")

    assert tokenizer_model.convert_tokens_to_ids(pad_token) == 0

    text_encoder_model = CLIPTextModelWithProjection.from_pretrained(
        clip_name,
        # `CLIPTextModel` does not support device_map="auto"
        # device_map="auto"
    )

    print("done loading CLIP text encoder")

    return text_encoder_model, tokenizer_model


def prior(*, args, checkpoint_map_location):
    print("loading prior")

    prior_checkpoint = torch.load(args.prior_checkpoint_path, map_location=checkpoint_map_location)
    prior_checkpoint = prior_checkpoint["state_dict"]

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


def decoder(*, args, checkpoint_map_location):
    print("loading decoder")

    decoder_checkpoint = torch.load(args.decoder_checkpoint_path, map_location=checkpoint_map_location)
    decoder_checkpoint = decoder_checkpoint["state_dict"]

    decoder_model = decoder_model_from_original_config()

    decoder_diffusers_checkpoint = decoder_original_checkpoint_to_diffusers_checkpoint(
        decoder_model, decoder_checkpoint
    )

    # text proj interlude

    # The original decoder implementation includes a set of parameters that are used
    # for creating the `encoder_hidden_states` which are what the U-net is conditioned
    # on. The diffusers conditional unet directly takes the encoder_hidden_states. We pull
    # the parameters into the UnCLIPTextProjModel class
    text_proj_model = text_proj_from_original_config()

    text_proj_checkpoint = text_proj_original_checkpoint_to_diffusers_checkpoint(decoder_checkpoint)

    load_checkpoint_to_model(text_proj_checkpoint, text_proj_model, strict=True)

    # done text proj interlude

    del decoder_checkpoint

    load_checkpoint_to_model(decoder_diffusers_checkpoint, decoder_model, strict=True)

    print("done loading decoder")

    return decoder_model, text_proj_model


def super_res_unet(*, args, checkpoint_map_location):
    print("loading super resolution unet")

    super_res_checkpoint = torch.load(args.super_res_unet_checkpoint_path, map_location=checkpoint_map_location)
    super_res_checkpoint = super_res_checkpoint["state_dict"]

    # model_first_steps

    super_res_first_model = super_res_unet_first_steps_model_from_original_config()

    super_res_first_steps_checkpoint = super_res_unet_first_steps_original_checkpoint_to_diffusers_checkpoint(
        super_res_first_model, super_res_checkpoint
    )

    # model_last_step
    super_res_last_model = super_res_unet_last_step_model_from_original_config()

    super_res_last_step_checkpoint = super_res_unet_last_step_original_checkpoint_to_diffusers_checkpoint(
        super_res_last_model, super_res_checkpoint
    )

    del super_res_checkpoint

    load_checkpoint_to_model(super_res_first_steps_checkpoint, super_res_first_model, strict=True)

    load_checkpoint_to_model(super_res_last_step_checkpoint, super_res_last_model, strict=True)

    print("done loading super resolution unet")

    return super_res_first_model, super_res_last_model


def load_checkpoint_to_model(checkpoint, model, strict=False):
    with tempfile.NamedTemporaryFile() as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        if strict:
            model.load_state_dict(torch.load(file.name), strict=True)
        else:
            load_checkpoint_and_dispatch(model, file.name, device_map="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--prior_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the prior checkpoint to convert.",
    )

    parser.add_argument(
        "--decoder_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the decoder checkpoint to convert.",
    )

    parser.add_argument(
        "--super_res_unet_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the super resolution checkpoint to convert.",
    )

    parser.add_argument(
        "--clip_stat_path", default=None, type=str, required=True, help="Path to the clip stats checkpoint to convert."
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
        text_encoder_model, tokenizer_model = text_encoder()

        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)

        decoder_model, text_proj_model = decoder(args=args, checkpoint_map_location=checkpoint_map_location)

        super_res_first_model, super_res_last_model = super_res_unet(
            args=args, checkpoint_map_location=checkpoint_map_location
        )

        prior_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample_range=5.0,
        )

        decoder_scheduler = UnCLIPScheduler(
            variance_type="learned_range",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        super_res_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        print(f"saving Kakao Brain unCLIP to {args.dump_path}")

        pipe = UnCLIPPipeline(
            prior=prior_model,
            decoder=decoder_model,
            text_proj=text_proj_model,
            tokenizer=tokenizer_model,
            text_encoder=text_encoder_model,
            super_res_first=super_res_first_model,
            super_res_last=super_res_last_model,
            prior_scheduler=prior_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )
        pipe.save_pretrained(args.dump_path)

        print("done writing Kakao Brain unCLIP")
    elif args.debug == "text_encoder":
        text_encoder_model, tokenizer_model = text_encoder()
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
    elif args.debug == "decoder":
        decoder_model, text_proj_model = decoder(args=args, checkpoint_map_location=checkpoint_map_location)
    elif args.debug == "super_res_unet":
        super_res_first_model, super_res_last_model = super_res_unet(
            args=args, checkpoint_map_location=checkpoint_map_location
        )
    else:
        raise ValueError(f"unknown debug value : {args.debug}")
