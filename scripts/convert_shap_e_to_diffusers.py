import argparse
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch
from collections import OrderedDict

from diffusers.models.prior_transformer import PriorTransformer
from diffusers.pipelines.shap_e import ShapEParamsProjModel, MLPNeRSTFModel


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
      --transmitter_checkpoint_path /home/yiyi_huggingface_co/shap-e/shap_e_model_cache/transmitter.pt\
      --dump_path /home/yiyi_huggingface_co/model_repo/shape/renderer \
      --debug renderer
```
"""


# prior

PRIOR_ORIGINAL_PREFIX = "wrapped"

# Uses default arguments
PRIOR_CONFIG = {
    "num_attention_heads": 16,
    "attention_head_dim": 1024 // 16,
    "num_layers": 24,
    "embedding_dim": 1024,
    "num_embeddings": 1024,
    "additional_embeddings": 0,
    "act_fn": "gelu",
    "time_embed_dim": 1024 * 4,
    "clip_embedding_dim": 768,
    "out_dim": 1024 * 2,
    "has_pre_norm": True,
    "has_encoder_hidden_states_proj": False,
    "has_prd_embedding": False,
    "has_post_process": False,
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

    # <original>.clip_img_proj -> <diffusers>.proj_in
    diffusers_checkpoint.update(
        {
            "proj_in.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.input_proj.weight"],
            "proj_in.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.input_proj.bias"],
        }
    )

    # <original>.text_emb_proj -> <diffusers>.embedding_proj
    diffusers_checkpoint.update(
        {
            "embedding_proj.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.clip_embed.weight"],
            "embedding_proj.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.clip_embed.bias"],
        }
    )

    # <original>.positional_embedding -> <diffusers>.positional_embedding
    diffusers_checkpoint.update({"positional_embedding": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.pos_emb"][None, :]})

    # <original>.ln_pre -> <diffusers>.norm_in
    diffusers_checkpoint.update(
        {
            "norm_in.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_pre.weight"],
            "norm_in.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_pre.bias"],
        }
    )

    # <original>.resblocks.<x> -> <diffusers>.transformer_blocks.<x>
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

    # <original>.final_ln -> <diffusers>.norm_out
    diffusers_checkpoint.update(
        {
            "norm_out.weight": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_post.weight"],
            "norm_out.bias": checkpoint[f"{PRIOR_ORIGINAL_PREFIX}.ln_post.bias"],
        }
    )

    # <original>.out_proj -> <diffusers>.proj_to_clip_embeddings
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


# params_proj

PARAMS_PROJ_ORIGINAL_PREFIX = "encoder.params_proj"

PARAMS_PROJ_CONFIG = {}

def params_proj_model_from_original_config():
    model = ShapEParamsProjModel(**PARAMS_PROJ_CONFIG)

    return model


def params_proj_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):

    diffusers_checkpoint = {
        k: checkpoint[f"{PARAMS_PROJ_ORIGINAL_PREFIX}.{k}"] for k in model.state_dict().keys()
    }

    return diffusers_checkpoint


# done params_proj


# renderer

RENDERER_ORIGINAL_PREFIX = "renderer.nerstf"

RENDERER_CONFIG = {}

def renderer_model_from_original_config():
    model = MLPNeRSTFModel(**RENDERER_CONFIG)

    return model

def renderer_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {
        k: checkpoint[f"{RENDERER_ORIGINAL_PREFIX}.{k}"] for k in model.state_dict().keys()
    }

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

    load_checkpoint_to_model(prior_diffusers_checkpoint, prior_model, strict=True)

    print("done loading prior")

    return prior_model


def params_proj(*, args, checkpoint_map_location):
    print("loading params_proj")

    params_proj_checkpoint = torch.load(args.transmitter_checkpoint_path, map_location=checkpoint_map_location)
    
    params_proj_model = params_proj_model_from_original_config()

    params_proj_diffusers_checkpoint = params_proj_original_checkpoint_to_diffusers_checkpoint(params_proj_model, params_proj_checkpoint)

    del params_proj_checkpoint

    load_checkpoint_to_model(params_proj_diffusers_checkpoint,params_proj_model, strict=True)

    print("done loading params_proj")

    return params_proj_model

def renderer(*, args, checkpoint_map_location):
    print(" loading renderer")

    renderer_checkpoint = torch.load(args.transmitter_checkpoint_path, map_location=checkpoint_map_location)
    
    renderer_model = renderer_model_from_original_config()

    renderer_diffusers_checkpoint = renderer_original_checkpoint_to_diffusers_checkpoint(renderer_model, renderer_checkpoint)

    del renderer_checkpoint

    load_checkpoint_to_model(renderer_diffusers_checkpoint, renderer_model, strict=True)

    print("done loading renderer")

    return renderer_model


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
    elif args.debug == "params_proj":
        params_proj_model = params_proj(args=args, checkpoint_map_location=checkpoint_map_location)
        params_proj_model.save_pretrained(args.dump_path)
    elif args.debug == "renderer":
        renderer_model = renderer(args=args, checkpoint_map_location=checkpoint_map_location)
        renderer_model.save_pretrained(args.dump_path)
    else:
        raise ValueError(f"unknown debug value : {args.debug}")
