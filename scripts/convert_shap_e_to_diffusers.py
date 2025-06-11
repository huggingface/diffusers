import argparse
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch

from diffusers.models.transformers.prior_transformer import PriorTransformer
from diffusers.pipelines.shap_e import ShapERenderer


"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget  "https://openaipublic.azureedge.net/main/shap-e/text_cond.pt"
```

Convert the model:
```sh
$ python scripts/convert_shap_e_to_diffusers.py \
      --prior_checkpoint_path  /home/yiyi_huggingface_co/shap-e/shap_e_model_cache/text_cond.pt \
      --prior_image_checkpoint_path /home/yiyi_huggingface_co/shap-e/shap_e_model_cache/image_cond.pt \
      --transmitter_checkpoint_path /home/yiyi_huggingface_co/shap-e/shap_e_model_cache/transmitter.pt\
      --dump_path /home/yiyi_huggingface_co/model_repo/shap-e-img2img/shap_e_renderer\
      --debug renderer
```
"""


# prior

PRIOR_ORIGINAL_PREFIX = "wrapped"

PRIOR_CONFIG = {
    "num_attention_heads": 16,
    "attention_head_dim": 1024 // 16,
    "num_layers": 24,
    "embedding_dim": 1024,
    "num_embeddings": 1024,
    "additional_embeddings": 0,
    "time_embed_act_fn": "gelu",
    "norm_in_type": "layer",
    "encoder_hid_proj_type": None,
    "added_emb_type": None,
    "time_embed_dim": 1024 * 4,
    "embedding_proj_dim": 768,
    "clip_embed_dim": 1024 * 2,
}


def prior_model_from_original_config():
    model = PriorTransformer(**PRIOR_CONFIG)

    return model


def prior_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    # <original>.time_embed.c_fc -> <diffusers>.time_embedding.linear_1
    diffusers_checkpoint.update(
        {
            "time_embedding.linear_1.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.c_fc.weight"],
            "time_embedding.linear_1.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.c_fc.bias"],
        }
    )

    # <original>.time_embed.c_proj -> <diffusers>.time_embedding.linear_2
    diffusers_checkpoint.update(
        {
            "time_embedding.linear_2.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.c_proj.weight"],
            "time_embedding.linear_2.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.time_embed.c_proj.bias"],
        }
    )

    # <original>.input_proj -> <diffusers>.proj_in
    diffusers_checkpoint.update(
        {
            "proj_in.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.input_proj.weight"],
            "proj_in.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.input_proj.bias"],
        }
    )

    # <original>.clip_emb -> <diffusers>.embedding_proj
    diffusers_checkpoint.update(
        {
            "embedding_proj.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.clip_embed.weight"],
            "embedding_proj.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.clip_embed.bias"],
        }
    )

    # <original>.pos_emb -> <diffusers>.positional_embedding
    diffusers_checkpoint.update({"positional_embedding": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.pos_emb"][None, :]})

    # <original>.ln_pre -> <diffusers>.norm_in
    diffusers_checkpoint.update(
        {
            "norm_in.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_pre.weight"],
            "norm_in.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_pre.bias"],
        }
    )

    # <original>.backbone.resblocks.<x> -> <diffusers>.transformer_blocks.<x>
    for idx in range(len(model.transformer_blocks)):
        diffusers_transformer_prefix = f"transformer_blocks.{idx}"
        original_transformer_prefix = f"{PRIOR_ORIGINAL_PREFIX}.backbone.resblocks.{idx}"

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

    # <original>.ln_post -> <diffusers>.norm_out
    diffusers_checkpoint.update(
        {
            "norm_out.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_post.weight"],
            "norm_out.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_post.bias"],
        }
    )

    # <original>.output_proj -> <diffusers>.proj_to_clip_embeddings
    diffusers_checkpoint.update(
        {
            "proj_to_clip_embeddings.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.output_proj.weight"],
            "proj_to_clip_embeddings.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.output_proj.bias"],
        }
    )

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


# prior_image (only slightly different from prior)


PRIOR_IMAGE_ORIGINAL_PREFIX = "wrapped"

# Uses default arguments
PRIOR_IMAGE_CONFIG = {
    "num_attention_heads": 8,
    "attention_head_dim": 1024 // 8,
    "num_layers": 24,
    "embedding_dim": 1024,
    "num_embeddings": 1024,
    "additional_embeddings": 0,
    "time_embed_act_fn": "gelu",
    "norm_in_type": "layer",
    "embedding_proj_norm_type": "layer",
    "encoder_hid_proj_type": None,
    "added_emb_type": None,
    "time_embed_dim": 1024 * 4,
    "embedding_proj_dim": 1024,
    "clip_embed_dim": 1024 * 2,
}


def prior_image_model_from_original_config():
    model = PriorTransformer(**PRIOR_IMAGE_CONFIG)

    return model


def prior_image_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}

    # <original>.time_embed.c_fc -> <diffusers>.time_embedding.linear_1
    diffusers_checkpoint.update(
        {
            "time_embedding.linear_1.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_fc.weight"],
            "time_embedding.linear_1.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_fc.bias"],
        }
    )

    # <original>.time_embed.c_proj -> <diffusers>.time_embedding.linear_2
    diffusers_checkpoint.update(
        {
            "time_embedding.linear_2.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_proj.weight"],
            "time_embedding.linear_2.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_proj.bias"],
        }
    )

    # <original>.input_proj -> <diffusers>.proj_in
    diffusers_checkpoint.update(
        {
            "proj_in.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.input_proj.weight"],
            "proj_in.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.input_proj.bias"],
        }
    )

    # <original>.clip_embed.0 -> <diffusers>.embedding_proj_norm
    diffusers_checkpoint.update(
        {
            "embedding_proj_norm.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.0.weight"],
            "embedding_proj_norm.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.0.bias"],
        }
    )

    # <original>..clip_embed.1 -> <diffusers>.embedding_proj
    diffusers_checkpoint.update(
        {
            "embedding_proj.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.1.weight"],
            "embedding_proj.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.1.bias"],
        }
    )

    # <original>.pos_emb -> <diffusers>.positional_embedding
    diffusers_checkpoint.update(
        {"positional_embedding": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.pos_emb"][None, :]}
    )

    # <original>.ln_pre -> <diffusers>.norm_in
    diffusers_checkpoint.update(
        {
            "norm_in.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_pre.weight"],
            "norm_in.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_pre.bias"],
        }
    )

    # <original>.backbone.resblocks.<x> -> <diffusers>.transformer_blocks.<x>
    for idx in range(len(model.transformer_blocks)):
        diffusers_transformer_prefix = f"transformer_blocks.{idx}"
        original_transformer_prefix = f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.backbone.resblocks.{idx}"

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

    # <original>.ln_post -> <diffusers>.norm_out
    diffusers_checkpoint.update(
        {
            "norm_out.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_post.weight"],
            "norm_out.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_post.bias"],
        }
    )

    # <original>.output_proj -> <diffusers>.proj_to_clip_embeddings
    diffusers_checkpoint.update(
        {
            "proj_to_clip_embeddings.weight": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.output_proj.weight"],
            "proj_to_clip_embeddings.bias": checkpoint[f"{PRIOR_IMAGE_ORIGINAL_PREFIX}.output_proj.bias"],
        }
    )

    return diffusers_checkpoint


# done prior_image


# renderer

## create the lookup table for marching cubes method used in MeshDecoder

MC_TABLE = [
    [],
    [[0, 1, 0, 2, 0, 4]],
    [[1, 0, 1, 5, 1, 3]],
    [[0, 4, 1, 5, 0, 2], [1, 5, 1, 3, 0, 2]],
    [[2, 0, 2, 3, 2, 6]],
    [[0, 1, 2, 3, 0, 4], [2, 3, 2, 6, 0, 4]],
    [[1, 0, 1, 5, 1, 3], [2, 6, 0, 2, 3, 2]],
    [[3, 2, 2, 6, 3, 1], [3, 1, 2, 6, 1, 5], [1, 5, 2, 6, 0, 4]],
    [[3, 1, 3, 7, 3, 2]],
    [[0, 2, 0, 4, 0, 1], [3, 7, 2, 3, 1, 3]],
    [[1, 5, 3, 7, 1, 0], [3, 7, 3, 2, 1, 0]],
    [[2, 0, 0, 4, 2, 3], [2, 3, 0, 4, 3, 7], [3, 7, 0, 4, 1, 5]],
    [[2, 0, 3, 1, 2, 6], [3, 1, 3, 7, 2, 6]],
    [[1, 3, 3, 7, 1, 0], [1, 0, 3, 7, 0, 4], [0, 4, 3, 7, 2, 6]],
    [[0, 1, 1, 5, 0, 2], [0, 2, 1, 5, 2, 6], [2, 6, 1, 5, 3, 7]],
    [[0, 4, 1, 5, 3, 7], [0, 4, 3, 7, 2, 6]],
    [[4, 0, 4, 6, 4, 5]],
    [[0, 2, 4, 6, 0, 1], [4, 6, 4, 5, 0, 1]],
    [[1, 5, 1, 3, 1, 0], [4, 6, 5, 4, 0, 4]],
    [[5, 1, 1, 3, 5, 4], [5, 4, 1, 3, 4, 6], [4, 6, 1, 3, 0, 2]],
    [[2, 0, 2, 3, 2, 6], [4, 5, 0, 4, 6, 4]],
    [[6, 4, 4, 5, 6, 2], [6, 2, 4, 5, 2, 3], [2, 3, 4, 5, 0, 1]],
    [[2, 6, 2, 0, 3, 2], [1, 0, 1, 5, 3, 1], [6, 4, 5, 4, 0, 4]],
    [[1, 3, 5, 4, 1, 5], [1, 3, 4, 6, 5, 4], [1, 3, 3, 2, 4, 6], [3, 2, 2, 6, 4, 6]],
    [[3, 1, 3, 7, 3, 2], [6, 4, 5, 4, 0, 4]],
    [[4, 5, 0, 1, 4, 6], [0, 1, 0, 2, 4, 6], [7, 3, 2, 3, 1, 3]],
    [[3, 2, 1, 0, 3, 7], [1, 0, 1, 5, 3, 7], [6, 4, 5, 4, 0, 4]],
    [[3, 7, 3, 2, 1, 5], [3, 2, 6, 4, 1, 5], [1, 5, 6, 4, 5, 4], [3, 2, 2, 0, 6, 4]],
    [[3, 7, 2, 6, 3, 1], [2, 6, 2, 0, 3, 1], [5, 4, 0, 4, 6, 4]],
    [[1, 0, 1, 3, 5, 4], [1, 3, 2, 6, 5, 4], [1, 3, 3, 7, 2, 6], [5, 4, 2, 6, 4, 6]],
    [[0, 1, 1, 5, 0, 2], [0, 2, 1, 5, 2, 6], [2, 6, 1, 5, 3, 7], [4, 5, 0, 4, 4, 6]],
    [[6, 2, 4, 6, 4, 5], [4, 5, 5, 1, 6, 2], [6, 2, 5, 1, 7, 3]],
    [[5, 1, 5, 4, 5, 7]],
    [[0, 1, 0, 2, 0, 4], [5, 7, 1, 5, 4, 5]],
    [[1, 0, 5, 4, 1, 3], [5, 4, 5, 7, 1, 3]],
    [[4, 5, 5, 7, 4, 0], [4, 0, 5, 7, 0, 2], [0, 2, 5, 7, 1, 3]],
    [[2, 0, 2, 3, 2, 6], [7, 5, 1, 5, 4, 5]],
    [[2, 6, 0, 4, 2, 3], [0, 4, 0, 1, 2, 3], [7, 5, 1, 5, 4, 5]],
    [[5, 7, 1, 3, 5, 4], [1, 3, 1, 0, 5, 4], [6, 2, 0, 2, 3, 2]],
    [[3, 1, 3, 2, 7, 5], [3, 2, 0, 4, 7, 5], [3, 2, 2, 6, 0, 4], [7, 5, 0, 4, 5, 4]],
    [[3, 7, 3, 2, 3, 1], [5, 4, 7, 5, 1, 5]],
    [[0, 4, 0, 1, 2, 0], [3, 1, 3, 7, 2, 3], [4, 5, 7, 5, 1, 5]],
    [[7, 3, 3, 2, 7, 5], [7, 5, 3, 2, 5, 4], [5, 4, 3, 2, 1, 0]],
    [[0, 4, 2, 3, 0, 2], [0, 4, 3, 7, 2, 3], [0, 4, 4, 5, 3, 7], [4, 5, 5, 7, 3, 7]],
    [[2, 0, 3, 1, 2, 6], [3, 1, 3, 7, 2, 6], [4, 5, 7, 5, 1, 5]],
    [[1, 3, 3, 7, 1, 0], [1, 0, 3, 7, 0, 4], [0, 4, 3, 7, 2, 6], [5, 7, 1, 5, 5, 4]],
    [[2, 6, 2, 0, 3, 7], [2, 0, 4, 5, 3, 7], [3, 7, 4, 5, 7, 5], [2, 0, 0, 1, 4, 5]],
    [[4, 0, 5, 4, 5, 7], [5, 7, 7, 3, 4, 0], [4, 0, 7, 3, 6, 2]],
    [[4, 6, 5, 7, 4, 0], [5, 7, 5, 1, 4, 0]],
    [[1, 0, 0, 2, 1, 5], [1, 5, 0, 2, 5, 7], [5, 7, 0, 2, 4, 6]],
    [[0, 4, 4, 6, 0, 1], [0, 1, 4, 6, 1, 3], [1, 3, 4, 6, 5, 7]],
    [[0, 2, 4, 6, 5, 7], [0, 2, 5, 7, 1, 3]],
    [[5, 1, 4, 0, 5, 7], [4, 0, 4, 6, 5, 7], [3, 2, 6, 2, 0, 2]],
    [[2, 3, 2, 6, 0, 1], [2, 6, 7, 5, 0, 1], [0, 1, 7, 5, 1, 5], [2, 6, 6, 4, 7, 5]],
    [[0, 4, 4, 6, 0, 1], [0, 1, 4, 6, 1, 3], [1, 3, 4, 6, 5, 7], [2, 6, 0, 2, 2, 3]],
    [[3, 1, 2, 3, 2, 6], [2, 6, 6, 4, 3, 1], [3, 1, 6, 4, 7, 5]],
    [[4, 6, 5, 7, 4, 0], [5, 7, 5, 1, 4, 0], [2, 3, 1, 3, 7, 3]],
    [[1, 0, 0, 2, 1, 5], [1, 5, 0, 2, 5, 7], [5, 7, 0, 2, 4, 6], [3, 2, 1, 3, 3, 7]],
    [[0, 1, 0, 4, 2, 3], [0, 4, 5, 7, 2, 3], [0, 4, 4, 6, 5, 7], [2, 3, 5, 7, 3, 7]],
    [[7, 5, 3, 7, 3, 2], [3, 2, 2, 0, 7, 5], [7, 5, 2, 0, 6, 4]],
    [[0, 4, 4, 6, 5, 7], [0, 4, 5, 7, 1, 5], [0, 2, 1, 3, 3, 7], [3, 7, 2, 6, 0, 2]],
    [
        [3, 1, 7, 3, 6, 2],
        [6, 2, 0, 1, 3, 1],
        [6, 4, 0, 1, 6, 2],
        [6, 4, 5, 1, 0, 1],
        [6, 4, 7, 5, 5, 1],
    ],
    [
        [4, 0, 6, 4, 7, 5],
        [7, 5, 1, 0, 4, 0],
        [7, 3, 1, 0, 7, 5],
        [7, 3, 2, 0, 1, 0],
        [7, 3, 6, 2, 2, 0],
    ],
    [[7, 3, 6, 2, 6, 4], [7, 5, 7, 3, 6, 4]],
    [[6, 2, 6, 7, 6, 4]],
    [[0, 4, 0, 1, 0, 2], [6, 7, 4, 6, 2, 6]],
    [[1, 0, 1, 5, 1, 3], [7, 6, 4, 6, 2, 6]],
    [[1, 3, 0, 2, 1, 5], [0, 2, 0, 4, 1, 5], [7, 6, 4, 6, 2, 6]],
    [[2, 3, 6, 7, 2, 0], [6, 7, 6, 4, 2, 0]],
    [[4, 0, 0, 1, 4, 6], [4, 6, 0, 1, 6, 7], [6, 7, 0, 1, 2, 3]],
    [[6, 4, 2, 0, 6, 7], [2, 0, 2, 3, 6, 7], [5, 1, 3, 1, 0, 1]],
    [[1, 5, 1, 3, 0, 4], [1, 3, 7, 6, 0, 4], [0, 4, 7, 6, 4, 6], [1, 3, 3, 2, 7, 6]],
    [[3, 2, 3, 1, 3, 7], [6, 4, 2, 6, 7, 6]],
    [[3, 7, 3, 2, 1, 3], [0, 2, 0, 4, 1, 0], [7, 6, 4, 6, 2, 6]],
    [[1, 5, 3, 7, 1, 0], [3, 7, 3, 2, 1, 0], [4, 6, 2, 6, 7, 6]],
    [[2, 0, 0, 4, 2, 3], [2, 3, 0, 4, 3, 7], [3, 7, 0, 4, 1, 5], [6, 4, 2, 6, 6, 7]],
    [[7, 6, 6, 4, 7, 3], [7, 3, 6, 4, 3, 1], [3, 1, 6, 4, 2, 0]],
    [[0, 1, 4, 6, 0, 4], [0, 1, 6, 7, 4, 6], [0, 1, 1, 3, 6, 7], [1, 3, 3, 7, 6, 7]],
    [[0, 2, 0, 1, 4, 6], [0, 1, 3, 7, 4, 6], [0, 1, 1, 5, 3, 7], [4, 6, 3, 7, 6, 7]],
    [[7, 3, 6, 7, 6, 4], [6, 4, 4, 0, 7, 3], [7, 3, 4, 0, 5, 1]],
    [[4, 0, 6, 2, 4, 5], [6, 2, 6, 7, 4, 5]],
    [[2, 6, 6, 7, 2, 0], [2, 0, 6, 7, 0, 1], [0, 1, 6, 7, 4, 5]],
    [[6, 7, 4, 5, 6, 2], [4, 5, 4, 0, 6, 2], [3, 1, 0, 1, 5, 1]],
    [[2, 0, 2, 6, 3, 1], [2, 6, 4, 5, 3, 1], [2, 6, 6, 7, 4, 5], [3, 1, 4, 5, 1, 5]],
    [[0, 2, 2, 3, 0, 4], [0, 4, 2, 3, 4, 5], [4, 5, 2, 3, 6, 7]],
    [[0, 1, 2, 3, 6, 7], [0, 1, 6, 7, 4, 5]],
    [[0, 2, 2, 3, 0, 4], [0, 4, 2, 3, 4, 5], [4, 5, 2, 3, 6, 7], [1, 3, 0, 1, 1, 5]],
    [[5, 4, 1, 5, 1, 3], [1, 3, 3, 2, 5, 4], [5, 4, 3, 2, 7, 6]],
    [[4, 0, 6, 2, 4, 5], [6, 2, 6, 7, 4, 5], [1, 3, 7, 3, 2, 3]],
    [[2, 6, 6, 7, 2, 0], [2, 0, 6, 7, 0, 1], [0, 1, 6, 7, 4, 5], [3, 7, 2, 3, 3, 1]],
    [[0, 1, 1, 5, 3, 7], [0, 1, 3, 7, 2, 3], [0, 4, 2, 6, 6, 7], [6, 7, 4, 5, 0, 4]],
    [
        [6, 2, 7, 6, 5, 4],
        [5, 4, 0, 2, 6, 2],
        [5, 1, 0, 2, 5, 4],
        [5, 1, 3, 2, 0, 2],
        [5, 1, 7, 3, 3, 2],
    ],
    [[3, 1, 3, 7, 2, 0], [3, 7, 5, 4, 2, 0], [2, 0, 5, 4, 0, 4], [3, 7, 7, 6, 5, 4]],
    [[1, 0, 3, 1, 3, 7], [3, 7, 7, 6, 1, 0], [1, 0, 7, 6, 5, 4]],
    [
        [1, 0, 5, 1, 7, 3],
        [7, 3, 2, 0, 1, 0],
        [7, 6, 2, 0, 7, 3],
        [7, 6, 4, 0, 2, 0],
        [7, 6, 5, 4, 4, 0],
    ],
    [[7, 6, 5, 4, 5, 1], [7, 3, 7, 6, 5, 1]],
    [[5, 7, 5, 1, 5, 4], [6, 2, 7, 6, 4, 6]],
    [[0, 2, 0, 4, 1, 0], [5, 4, 5, 7, 1, 5], [2, 6, 7, 6, 4, 6]],
    [[1, 0, 5, 4, 1, 3], [5, 4, 5, 7, 1, 3], [2, 6, 7, 6, 4, 6]],
    [[4, 5, 5, 7, 4, 0], [4, 0, 5, 7, 0, 2], [0, 2, 5, 7, 1, 3], [6, 7, 4, 6, 6, 2]],
    [[2, 3, 6, 7, 2, 0], [6, 7, 6, 4, 2, 0], [1, 5, 4, 5, 7, 5]],
    [[4, 0, 0, 1, 4, 6], [4, 6, 0, 1, 6, 7], [6, 7, 0, 1, 2, 3], [5, 1, 4, 5, 5, 7]],
    [[0, 2, 2, 3, 6, 7], [0, 2, 6, 7, 4, 6], [0, 1, 4, 5, 5, 7], [5, 7, 1, 3, 0, 1]],
    [
        [5, 4, 7, 5, 3, 1],
        [3, 1, 0, 4, 5, 4],
        [3, 2, 0, 4, 3, 1],
        [3, 2, 6, 4, 0, 4],
        [3, 2, 7, 6, 6, 4],
    ],
    [[5, 4, 5, 7, 1, 5], [3, 7, 3, 2, 1, 3], [4, 6, 2, 6, 7, 6]],
    [[1, 0, 0, 2, 0, 4], [1, 5, 5, 4, 5, 7], [3, 2, 1, 3, 3, 7], [2, 6, 7, 6, 4, 6]],
    [[7, 3, 3, 2, 7, 5], [7, 5, 3, 2, 5, 4], [5, 4, 3, 2, 1, 0], [6, 2, 7, 6, 6, 4]],
    [
        [0, 4, 2, 3, 0, 2],
        [0, 4, 3, 7, 2, 3],
        [0, 4, 4, 5, 3, 7],
        [4, 5, 5, 7, 3, 7],
        [6, 7, 4, 6, 2, 6],
    ],
    [[7, 6, 6, 4, 7, 3], [7, 3, 6, 4, 3, 1], [3, 1, 6, 4, 2, 0], [5, 4, 7, 5, 5, 1]],
    [
        [0, 1, 4, 6, 0, 4],
        [0, 1, 6, 7, 4, 6],
        [0, 1, 1, 3, 6, 7],
        [1, 3, 3, 7, 6, 7],
        [5, 7, 1, 5, 4, 5],
    ],
    [
        [6, 7, 4, 6, 0, 2],
        [0, 2, 3, 7, 6, 7],
        [0, 1, 3, 7, 0, 2],
        [0, 1, 5, 7, 3, 7],
        [0, 1, 4, 5, 5, 7],
    ],
    [[4, 0, 6, 7, 4, 6], [4, 0, 7, 3, 6, 7], [4, 0, 5, 7, 7, 3], [4, 5, 5, 7, 4, 0]],
    [[7, 5, 5, 1, 7, 6], [7, 6, 5, 1, 6, 2], [6, 2, 5, 1, 4, 0]],
    [[0, 2, 1, 5, 0, 1], [0, 2, 5, 7, 1, 5], [0, 2, 2, 6, 5, 7], [2, 6, 6, 7, 5, 7]],
    [[1, 3, 1, 0, 5, 7], [1, 0, 2, 6, 5, 7], [5, 7, 2, 6, 7, 6], [1, 0, 0, 4, 2, 6]],
    [[2, 0, 6, 2, 6, 7], [6, 7, 7, 5, 2, 0], [2, 0, 7, 5, 3, 1]],
    [[0, 4, 0, 2, 1, 5], [0, 2, 6, 7, 1, 5], [0, 2, 2, 3, 6, 7], [1, 5, 6, 7, 5, 7]],
    [[7, 6, 5, 7, 5, 1], [5, 1, 1, 0, 7, 6], [7, 6, 1, 0, 3, 2]],
    [
        [2, 0, 3, 2, 7, 6],
        [7, 6, 4, 0, 2, 0],
        [7, 5, 4, 0, 7, 6],
        [7, 5, 1, 0, 4, 0],
        [7, 5, 3, 1, 1, 0],
    ],
    [[7, 5, 3, 1, 3, 2], [7, 6, 7, 5, 3, 2]],
    [[7, 5, 5, 1, 7, 6], [7, 6, 5, 1, 6, 2], [6, 2, 5, 1, 4, 0], [3, 1, 7, 3, 3, 2]],
    [
        [0, 2, 1, 5, 0, 1],
        [0, 2, 5, 7, 1, 5],
        [0, 2, 2, 6, 5, 7],
        [2, 6, 6, 7, 5, 7],
        [3, 7, 2, 3, 1, 3],
    ],
    [
        [3, 7, 2, 3, 0, 1],
        [0, 1, 5, 7, 3, 7],
        [0, 4, 5, 7, 0, 1],
        [0, 4, 6, 7, 5, 7],
        [0, 4, 2, 6, 6, 7],
    ],
    [[2, 0, 3, 7, 2, 3], [2, 0, 7, 5, 3, 7], [2, 0, 6, 7, 7, 5], [2, 6, 6, 7, 2, 0]],
    [
        [5, 7, 1, 5, 0, 4],
        [0, 4, 6, 7, 5, 7],
        [0, 2, 6, 7, 0, 4],
        [0, 2, 3, 7, 6, 7],
        [0, 2, 1, 3, 3, 7],
    ],
    [[1, 0, 5, 7, 1, 5], [1, 0, 7, 6, 5, 7], [1, 0, 3, 7, 7, 6], [1, 3, 3, 7, 1, 0]],
    [[0, 2, 0, 1, 0, 4], [3, 7, 6, 7, 5, 7]],
    [[7, 5, 7, 3, 7, 6]],
    [[7, 3, 7, 5, 7, 6]],
    [[0, 1, 0, 2, 0, 4], [6, 7, 3, 7, 5, 7]],
    [[1, 3, 1, 0, 1, 5], [7, 6, 3, 7, 5, 7]],
    [[0, 4, 1, 5, 0, 2], [1, 5, 1, 3, 0, 2], [6, 7, 3, 7, 5, 7]],
    [[2, 6, 2, 0, 2, 3], [7, 5, 6, 7, 3, 7]],
    [[0, 1, 2, 3, 0, 4], [2, 3, 2, 6, 0, 4], [5, 7, 6, 7, 3, 7]],
    [[1, 5, 1, 3, 0, 1], [2, 3, 2, 6, 0, 2], [5, 7, 6, 7, 3, 7]],
    [[3, 2, 2, 6, 3, 1], [3, 1, 2, 6, 1, 5], [1, 5, 2, 6, 0, 4], [7, 6, 3, 7, 7, 5]],
    [[3, 1, 7, 5, 3, 2], [7, 5, 7, 6, 3, 2]],
    [[7, 6, 3, 2, 7, 5], [3, 2, 3, 1, 7, 5], [4, 0, 1, 0, 2, 0]],
    [[5, 7, 7, 6, 5, 1], [5, 1, 7, 6, 1, 0], [1, 0, 7, 6, 3, 2]],
    [[2, 3, 2, 0, 6, 7], [2, 0, 1, 5, 6, 7], [2, 0, 0, 4, 1, 5], [6, 7, 1, 5, 7, 5]],
    [[6, 2, 2, 0, 6, 7], [6, 7, 2, 0, 7, 5], [7, 5, 2, 0, 3, 1]],
    [[0, 4, 0, 1, 2, 6], [0, 1, 5, 7, 2, 6], [2, 6, 5, 7, 6, 7], [0, 1, 1, 3, 5, 7]],
    [[1, 5, 0, 2, 1, 0], [1, 5, 2, 6, 0, 2], [1, 5, 5, 7, 2, 6], [5, 7, 7, 6, 2, 6]],
    [[5, 1, 7, 5, 7, 6], [7, 6, 6, 2, 5, 1], [5, 1, 6, 2, 4, 0]],
    [[4, 5, 4, 0, 4, 6], [7, 3, 5, 7, 6, 7]],
    [[0, 2, 4, 6, 0, 1], [4, 6, 4, 5, 0, 1], [3, 7, 5, 7, 6, 7]],
    [[4, 6, 4, 5, 0, 4], [1, 5, 1, 3, 0, 1], [6, 7, 3, 7, 5, 7]],
    [[5, 1, 1, 3, 5, 4], [5, 4, 1, 3, 4, 6], [4, 6, 1, 3, 0, 2], [7, 3, 5, 7, 7, 6]],
    [[2, 3, 2, 6, 0, 2], [4, 6, 4, 5, 0, 4], [3, 7, 5, 7, 6, 7]],
    [[6, 4, 4, 5, 6, 2], [6, 2, 4, 5, 2, 3], [2, 3, 4, 5, 0, 1], [7, 5, 6, 7, 7, 3]],
    [[0, 1, 1, 5, 1, 3], [0, 2, 2, 3, 2, 6], [4, 5, 0, 4, 4, 6], [5, 7, 6, 7, 3, 7]],
    [
        [1, 3, 5, 4, 1, 5],
        [1, 3, 4, 6, 5, 4],
        [1, 3, 3, 2, 4, 6],
        [3, 2, 2, 6, 4, 6],
        [7, 6, 3, 7, 5, 7],
    ],
    [[3, 1, 7, 5, 3, 2], [7, 5, 7, 6, 3, 2], [0, 4, 6, 4, 5, 4]],
    [[1, 0, 0, 2, 4, 6], [1, 0, 4, 6, 5, 4], [1, 3, 5, 7, 7, 6], [7, 6, 3, 2, 1, 3]],
    [[5, 7, 7, 6, 5, 1], [5, 1, 7, 6, 1, 0], [1, 0, 7, 6, 3, 2], [4, 6, 5, 4, 4, 0]],
    [
        [7, 5, 6, 7, 2, 3],
        [2, 3, 1, 5, 7, 5],
        [2, 0, 1, 5, 2, 3],
        [2, 0, 4, 5, 1, 5],
        [2, 0, 6, 4, 4, 5],
    ],
    [[6, 2, 2, 0, 6, 7], [6, 7, 2, 0, 7, 5], [7, 5, 2, 0, 3, 1], [4, 0, 6, 4, 4, 5]],
    [
        [4, 6, 5, 4, 1, 0],
        [1, 0, 2, 6, 4, 6],
        [1, 3, 2, 6, 1, 0],
        [1, 3, 7, 6, 2, 6],
        [1, 3, 5, 7, 7, 6],
    ],
    [
        [1, 5, 0, 2, 1, 0],
        [1, 5, 2, 6, 0, 2],
        [1, 5, 5, 7, 2, 6],
        [5, 7, 7, 6, 2, 6],
        [4, 6, 5, 4, 0, 4],
    ],
    [[5, 1, 4, 6, 5, 4], [5, 1, 6, 2, 4, 6], [5, 1, 7, 6, 6, 2], [5, 7, 7, 6, 5, 1]],
    [[5, 4, 7, 6, 5, 1], [7, 6, 7, 3, 5, 1]],
    [[7, 3, 5, 1, 7, 6], [5, 1, 5, 4, 7, 6], [2, 0, 4, 0, 1, 0]],
    [[3, 1, 1, 0, 3, 7], [3, 7, 1, 0, 7, 6], [7, 6, 1, 0, 5, 4]],
    [[0, 2, 0, 4, 1, 3], [0, 4, 6, 7, 1, 3], [1, 3, 6, 7, 3, 7], [0, 4, 4, 5, 6, 7]],
    [[5, 4, 7, 6, 5, 1], [7, 6, 7, 3, 5, 1], [0, 2, 3, 2, 6, 2]],
    [[1, 5, 5, 4, 7, 6], [1, 5, 7, 6, 3, 7], [1, 0, 3, 2, 2, 6], [2, 6, 0, 4, 1, 0]],
    [[3, 1, 1, 0, 3, 7], [3, 7, 1, 0, 7, 6], [7, 6, 1, 0, 5, 4], [2, 0, 3, 2, 2, 6]],
    [
        [2, 3, 6, 2, 4, 0],
        [4, 0, 1, 3, 2, 3],
        [4, 5, 1, 3, 4, 0],
        [4, 5, 7, 3, 1, 3],
        [4, 5, 6, 7, 7, 3],
    ],
    [[1, 5, 5, 4, 1, 3], [1, 3, 5, 4, 3, 2], [3, 2, 5, 4, 7, 6]],
    [[1, 5, 5, 4, 1, 3], [1, 3, 5, 4, 3, 2], [3, 2, 5, 4, 7, 6], [0, 4, 1, 0, 0, 2]],
    [[1, 0, 5, 4, 7, 6], [1, 0, 7, 6, 3, 2]],
    [[2, 3, 0, 2, 0, 4], [0, 4, 4, 5, 2, 3], [2, 3, 4, 5, 6, 7]],
    [[1, 3, 1, 5, 0, 2], [1, 5, 7, 6, 0, 2], [1, 5, 5, 4, 7, 6], [0, 2, 7, 6, 2, 6]],
    [
        [5, 1, 4, 5, 6, 7],
        [6, 7, 3, 1, 5, 1],
        [6, 2, 3, 1, 6, 7],
        [6, 2, 0, 1, 3, 1],
        [6, 2, 4, 0, 0, 1],
    ],
    [[6, 7, 2, 6, 2, 0], [2, 0, 0, 1, 6, 7], [6, 7, 0, 1, 4, 5]],
    [[6, 2, 4, 0, 4, 5], [6, 7, 6, 2, 4, 5]],
    [[6, 7, 7, 3, 6, 4], [6, 4, 7, 3, 4, 0], [4, 0, 7, 3, 5, 1]],
    [[1, 5, 1, 0, 3, 7], [1, 0, 4, 6, 3, 7], [1, 0, 0, 2, 4, 6], [3, 7, 4, 6, 7, 6]],
    [[1, 0, 3, 7, 1, 3], [1, 0, 7, 6, 3, 7], [1, 0, 0, 4, 7, 6], [0, 4, 4, 6, 7, 6]],
    [[6, 4, 7, 6, 7, 3], [7, 3, 3, 1, 6, 4], [6, 4, 3, 1, 2, 0]],
    [[6, 7, 7, 3, 6, 4], [6, 4, 7, 3, 4, 0], [4, 0, 7, 3, 5, 1], [2, 3, 6, 2, 2, 0]],
    [
        [7, 6, 3, 7, 1, 5],
        [1, 5, 4, 6, 7, 6],
        [1, 0, 4, 6, 1, 5],
        [1, 0, 2, 6, 4, 6],
        [1, 0, 3, 2, 2, 6],
    ],
    [
        [1, 0, 3, 7, 1, 3],
        [1, 0, 7, 6, 3, 7],
        [1, 0, 0, 4, 7, 6],
        [0, 4, 4, 6, 7, 6],
        [2, 6, 0, 2, 3, 2],
    ],
    [[3, 1, 7, 6, 3, 7], [3, 1, 6, 4, 7, 6], [3, 1, 2, 6, 6, 4], [3, 2, 2, 6, 3, 1]],
    [[3, 2, 3, 1, 7, 6], [3, 1, 0, 4, 7, 6], [7, 6, 0, 4, 6, 4], [3, 1, 1, 5, 0, 4]],
    [
        [0, 1, 2, 0, 6, 4],
        [6, 4, 5, 1, 0, 1],
        [6, 7, 5, 1, 6, 4],
        [6, 7, 3, 1, 5, 1],
        [6, 7, 2, 3, 3, 1],
    ],
    [[0, 1, 4, 0, 4, 6], [4, 6, 6, 7, 0, 1], [0, 1, 6, 7, 2, 3]],
    [[6, 7, 2, 3, 2, 0], [6, 4, 6, 7, 2, 0]],
    [
        [2, 6, 0, 2, 1, 3],
        [1, 3, 7, 6, 2, 6],
        [1, 5, 7, 6, 1, 3],
        [1, 5, 4, 6, 7, 6],
        [1, 5, 0, 4, 4, 6],
    ],
    [[1, 5, 1, 0, 1, 3], [4, 6, 7, 6, 2, 6]],
    [[0, 1, 2, 6, 0, 2], [0, 1, 6, 7, 2, 6], [0, 1, 4, 6, 6, 7], [0, 4, 4, 6, 0, 1]],
    [[6, 7, 6, 2, 6, 4]],
    [[6, 2, 7, 3, 6, 4], [7, 3, 7, 5, 6, 4]],
    [[7, 5, 6, 4, 7, 3], [6, 4, 6, 2, 7, 3], [1, 0, 2, 0, 4, 0]],
    [[6, 2, 7, 3, 6, 4], [7, 3, 7, 5, 6, 4], [0, 1, 5, 1, 3, 1]],
    [[2, 0, 0, 4, 1, 5], [2, 0, 1, 5, 3, 1], [2, 6, 3, 7, 7, 5], [7, 5, 6, 4, 2, 6]],
    [[3, 7, 7, 5, 3, 2], [3, 2, 7, 5, 2, 0], [2, 0, 7, 5, 6, 4]],
    [[3, 2, 3, 7, 1, 0], [3, 7, 6, 4, 1, 0], [3, 7, 7, 5, 6, 4], [1, 0, 6, 4, 0, 4]],
    [[3, 7, 7, 5, 3, 2], [3, 2, 7, 5, 2, 0], [2, 0, 7, 5, 6, 4], [1, 5, 3, 1, 1, 0]],
    [
        [7, 3, 5, 7, 4, 6],
        [4, 6, 2, 3, 7, 3],
        [4, 0, 2, 3, 4, 6],
        [4, 0, 1, 3, 2, 3],
        [4, 0, 5, 1, 1, 3],
    ],
    [[2, 3, 3, 1, 2, 6], [2, 6, 3, 1, 6, 4], [6, 4, 3, 1, 7, 5]],
    [[2, 3, 3, 1, 2, 6], [2, 6, 3, 1, 6, 4], [6, 4, 3, 1, 7, 5], [0, 1, 2, 0, 0, 4]],
    [[1, 0, 1, 5, 3, 2], [1, 5, 4, 6, 3, 2], [3, 2, 4, 6, 2, 6], [1, 5, 5, 7, 4, 6]],
    [
        [0, 2, 4, 0, 5, 1],
        [5, 1, 3, 2, 0, 2],
        [5, 7, 3, 2, 5, 1],
        [5, 7, 6, 2, 3, 2],
        [5, 7, 4, 6, 6, 2],
    ],
    [[2, 0, 3, 1, 7, 5], [2, 0, 7, 5, 6, 4]],
    [[4, 6, 0, 4, 0, 1], [0, 1, 1, 3, 4, 6], [4, 6, 1, 3, 5, 7]],
    [[0, 2, 1, 0, 1, 5], [1, 5, 5, 7, 0, 2], [0, 2, 5, 7, 4, 6]],
    [[5, 7, 4, 6, 4, 0], [5, 1, 5, 7, 4, 0]],
    [[5, 4, 4, 0, 5, 7], [5, 7, 4, 0, 7, 3], [7, 3, 4, 0, 6, 2]],
    [[0, 1, 0, 2, 4, 5], [0, 2, 3, 7, 4, 5], [4, 5, 3, 7, 5, 7], [0, 2, 2, 6, 3, 7]],
    [[5, 4, 4, 0, 5, 7], [5, 7, 4, 0, 7, 3], [7, 3, 4, 0, 6, 2], [1, 0, 5, 1, 1, 3]],
    [
        [1, 5, 3, 1, 2, 0],
        [2, 0, 4, 5, 1, 5],
        [2, 6, 4, 5, 2, 0],
        [2, 6, 7, 5, 4, 5],
        [2, 6, 3, 7, 7, 5],
    ],
    [[2, 3, 0, 4, 2, 0], [2, 3, 4, 5, 0, 4], [2, 3, 3, 7, 4, 5], [3, 7, 7, 5, 4, 5]],
    [[3, 2, 7, 3, 7, 5], [7, 5, 5, 4, 3, 2], [3, 2, 5, 4, 1, 0]],
    [
        [2, 3, 0, 4, 2, 0],
        [2, 3, 4, 5, 0, 4],
        [2, 3, 3, 7, 4, 5],
        [3, 7, 7, 5, 4, 5],
        [1, 5, 3, 1, 0, 1],
    ],
    [[3, 2, 1, 5, 3, 1], [3, 2, 5, 4, 1, 5], [3, 2, 7, 5, 5, 4], [3, 7, 7, 5, 3, 2]],
    [[2, 6, 2, 3, 0, 4], [2, 3, 7, 5, 0, 4], [2, 3, 3, 1, 7, 5], [0, 4, 7, 5, 4, 5]],
    [
        [3, 2, 1, 3, 5, 7],
        [5, 7, 6, 2, 3, 2],
        [5, 4, 6, 2, 5, 7],
        [5, 4, 0, 2, 6, 2],
        [5, 4, 1, 0, 0, 2],
    ],
    [
        [4, 5, 0, 4, 2, 6],
        [2, 6, 7, 5, 4, 5],
        [2, 3, 7, 5, 2, 6],
        [2, 3, 1, 5, 7, 5],
        [2, 3, 0, 1, 1, 5],
    ],
    [[2, 3, 2, 0, 2, 6], [1, 5, 7, 5, 4, 5]],
    [[5, 7, 4, 5, 4, 0], [4, 0, 0, 2, 5, 7], [5, 7, 0, 2, 1, 3]],
    [[5, 4, 1, 0, 1, 3], [5, 7, 5, 4, 1, 3]],
    [[0, 2, 4, 5, 0, 4], [0, 2, 5, 7, 4, 5], [0, 2, 1, 5, 5, 7], [0, 1, 1, 5, 0, 2]],
    [[5, 4, 5, 1, 5, 7]],
    [[4, 6, 6, 2, 4, 5], [4, 5, 6, 2, 5, 1], [5, 1, 6, 2, 7, 3]],
    [[4, 6, 6, 2, 4, 5], [4, 5, 6, 2, 5, 1], [5, 1, 6, 2, 7, 3], [0, 2, 4, 0, 0, 1]],
    [[3, 7, 3, 1, 2, 6], [3, 1, 5, 4, 2, 6], [3, 1, 1, 0, 5, 4], [2, 6, 5, 4, 6, 4]],
    [
        [6, 4, 2, 6, 3, 7],
        [3, 7, 5, 4, 6, 4],
        [3, 1, 5, 4, 3, 7],
        [3, 1, 0, 4, 5, 4],
        [3, 1, 2, 0, 0, 4],
    ],
    [[2, 0, 2, 3, 6, 4], [2, 3, 1, 5, 6, 4], [6, 4, 1, 5, 4, 5], [2, 3, 3, 7, 1, 5]],
    [
        [0, 4, 1, 0, 3, 2],
        [3, 2, 6, 4, 0, 4],
        [3, 7, 6, 4, 3, 2],
        [3, 7, 5, 4, 6, 4],
        [3, 7, 1, 5, 5, 4],
    ],
    [
        [1, 3, 0, 1, 4, 5],
        [4, 5, 7, 3, 1, 3],
        [4, 6, 7, 3, 4, 5],
        [4, 6, 2, 3, 7, 3],
        [4, 6, 0, 2, 2, 3],
    ],
    [[3, 7, 3, 1, 3, 2], [5, 4, 6, 4, 0, 4]],
    [[3, 1, 2, 6, 3, 2], [3, 1, 6, 4, 2, 6], [3, 1, 1, 5, 6, 4], [1, 5, 5, 4, 6, 4]],
    [
        [3, 1, 2, 6, 3, 2],
        [3, 1, 6, 4, 2, 6],
        [3, 1, 1, 5, 6, 4],
        [1, 5, 5, 4, 6, 4],
        [0, 4, 1, 0, 2, 0],
    ],
    [[4, 5, 6, 4, 6, 2], [6, 2, 2, 3, 4, 5], [4, 5, 2, 3, 0, 1]],
    [[2, 3, 6, 4, 2, 6], [2, 3, 4, 5, 6, 4], [2, 3, 0, 4, 4, 5], [2, 0, 0, 4, 2, 3]],
    [[1, 3, 5, 1, 5, 4], [5, 4, 4, 6, 1, 3], [1, 3, 4, 6, 0, 2]],
    [[1, 3, 0, 4, 1, 0], [1, 3, 4, 6, 0, 4], [1, 3, 5, 4, 4, 6], [1, 5, 5, 4, 1, 3]],
    [[4, 6, 0, 2, 0, 1], [4, 5, 4, 6, 0, 1]],
    [[4, 6, 4, 0, 4, 5]],
    [[4, 0, 6, 2, 7, 3], [4, 0, 7, 3, 5, 1]],
    [[1, 5, 0, 1, 0, 2], [0, 2, 2, 6, 1, 5], [1, 5, 2, 6, 3, 7]],
    [[3, 7, 1, 3, 1, 0], [1, 0, 0, 4, 3, 7], [3, 7, 0, 4, 2, 6]],
    [[3, 1, 2, 0, 2, 6], [3, 7, 3, 1, 2, 6]],
    [[0, 4, 2, 0, 2, 3], [2, 3, 3, 7, 0, 4], [0, 4, 3, 7, 1, 5]],
    [[3, 7, 1, 5, 1, 0], [3, 2, 3, 7, 1, 0]],
    [[0, 4, 1, 3, 0, 1], [0, 4, 3, 7, 1, 3], [0, 4, 2, 3, 3, 7], [0, 2, 2, 3, 0, 4]],
    [[3, 7, 3, 1, 3, 2]],
    [[2, 6, 3, 2, 3, 1], [3, 1, 1, 5, 2, 6], [2, 6, 1, 5, 0, 4]],
    [[1, 5, 3, 2, 1, 3], [1, 5, 2, 6, 3, 2], [1, 5, 0, 2, 2, 6], [1, 0, 0, 2, 1, 5]],
    [[2, 3, 0, 1, 0, 4], [2, 6, 2, 3, 0, 4]],
    [[2, 3, 2, 0, 2, 6]],
    [[1, 5, 0, 4, 0, 2], [1, 3, 1, 5, 0, 2]],
    [[1, 5, 1, 0, 1, 3]],
    [[0, 2, 0, 1, 0, 4]],
    [],
]


def create_mc_lookup_table():
    cases = torch.zeros(256, 5, 3, dtype=torch.long)
    masks = torch.zeros(256, 5, dtype=torch.bool)

    edge_to_index = {
        (0, 1): 0,
        (2, 3): 1,
        (4, 5): 2,
        (6, 7): 3,
        (0, 2): 4,
        (1, 3): 5,
        (4, 6): 6,
        (5, 7): 7,
        (0, 4): 8,
        (1, 5): 9,
        (2, 6): 10,
        (3, 7): 11,
    }

    for i, case in enumerate(MC_TABLE):
        for j, tri in enumerate(case):
            for k, (c1, c2) in enumerate(zip(tri[::2], tri[1::2])):
                cases[i, j, k] = edge_to_index[(c1, c2) if c1 < c2 else (c2, c1)]
            masks[i, j] = True
    return cases, masks


RENDERER_CONFIG = {}


def renderer_model_from_original_config():
    model = ShapERenderer(**RENDERER_CONFIG)

    return model


RENDERER_MLP_ORIGINAL_PREFIX = "renderer.nerstf"

RENDERER_PARAMS_PROJ_ORIGINAL_PREFIX = "encoder.params_proj"


def renderer_model_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update(
        {f"mlp.{k}": checkpoint[f"{RENDERER_MLP_ORIGINAL_PREFIX}.{k}"] for k in model.mlp.state_dict().keys()}
    )

    diffusers_checkpoint.update(
        {
            f"params_proj.{k}": checkpoint[f"{RENDERER_PARAMS_PROJ_ORIGINAL_PREFIX}.{k}"]
            for k in model.params_proj.state_dict().keys()
        }
    )

    diffusers_checkpoint.update({"void.background": model.state_dict()["void.background"]})

    cases, masks = create_mc_lookup_table()

    diffusers_checkpoint.update({"mesh_decoder.cases": cases})
    diffusers_checkpoint.update({"mesh_decoder.masks": masks})

    return diffusers_checkpoint


# done renderer


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


def prior(*, args, checkpoint_map_location):
    print("loading prior")

    prior_checkpoint = torch.load(args.prior_checkpoint_path, map_location=checkpoint_map_location)

    prior_model = prior_model_from_original_config()

    prior_diffusers_checkpoint = prior_original_checkpoint_to_diffusers_checkpoint(prior_model, prior_checkpoint)

    del prior_checkpoint

    load_prior_checkpoint_to_model(prior_diffusers_checkpoint, prior_model)

    print("done loading prior")

    return prior_model


def prior_image(*, args, checkpoint_map_location):
    print("loading prior_image")

    print(f"load checkpoint from {args.prior_image_checkpoint_path}")
    prior_checkpoint = torch.load(args.prior_image_checkpoint_path, map_location=checkpoint_map_location)

    prior_model = prior_image_model_from_original_config()

    prior_diffusers_checkpoint = prior_image_original_checkpoint_to_diffusers_checkpoint(prior_model, prior_checkpoint)

    del prior_checkpoint

    load_prior_checkpoint_to_model(prior_diffusers_checkpoint, prior_model)

    print("done loading prior_image")

    return prior_model


def renderer(*, args, checkpoint_map_location):
    print(" loading renderer")

    renderer_checkpoint = torch.load(args.transmitter_checkpoint_path, map_location=checkpoint_map_location)

    renderer_model = renderer_model_from_original_config()

    renderer_diffusers_checkpoint = renderer_model_original_checkpoint_to_diffusers_checkpoint(
        renderer_model, renderer_checkpoint
    )

    del renderer_checkpoint

    load_checkpoint_to_model(renderer_diffusers_checkpoint, renderer_model, strict=True)

    print("done loading renderer")

    return renderer_model


# prior model will expect clip_mean and clip_std, which are missing from the state_dict
PRIOR_EXPECTED_MISSING_KEYS = ["clip_mean", "clip_std"]


def load_prior_checkpoint_to_model(checkpoint, model):
    with tempfile.NamedTemporaryFile() as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(file.name), strict=False)
        missing_keys = list(set(missing_keys) - set(PRIOR_EXPECTED_MISSING_KEYS))

        if len(unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys when loading prior model: {unexpected_keys}")
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys when loading prior model: {missing_keys}")


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
        required=False,
        help="Path to the prior checkpoint to convert.",
    )

    parser.add_argument(
        "--prior_image_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the prior_image checkpoint to convert.",
    )

    parser.add_argument(
        "--transmitter_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the transmitter checkpoint to convert.",
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
        print("YiYi TO-DO")
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
        prior_model.save_pretrained(args.dump_path)
    elif args.debug == "prior_image":
        prior_model = prior_image(args=args, checkpoint_map_location=checkpoint_map_location)
        prior_model.save_pretrained(args.dump_path)
    elif args.debug == "renderer":
        renderer_model = renderer(args=args, checkpoint_map_location=checkpoint_map_location)
        renderer_model.save_pretrained(args.dump_path)
    else:
        raise ValueError(f"unknown debug value : {args.debug}")
