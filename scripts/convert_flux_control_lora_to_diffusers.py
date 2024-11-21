import argparse
from contextlib import nullcontext

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--filename", default="flux-canny-dev-lora.safetensors", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str, default="bf16")

args = parser.parse_args()
dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32


# Adapted from from the original BFL codebase.
def optionally_expand_state_dict(name: str, param: torch.Tensor, state_dict: dict) -> dict:
    if name in state_dict:
        print(f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}.")
        # expand with zeros:
        expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
        # popular with pre-trained param for the first half. Remaining half stays with zeros.
        slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
        expanded_state_dict_weight[slices] = state_dict[name]
        state_dict[name] = expanded_state_dict_weight

    return state_dict


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


def convert_flux_control_lora_checkpoint_to_diffusers(
    original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio=4.0
):
    converted_state_dict = {}

    ## time_text_embed.timestep_embedder <-  time_in
    for lora_key, diffusers_lora_key in zip(["lora_A", "lora_B"], ["lora_A", "lora_B"]):
        converted_state_dict[
            f"time_text_embed.timestep_embedder.linear_1.{diffusers_lora_key}.weight"
        ] = original_state_dict.pop(f"time_in.in_layer.{lora_key}.weight")
        if f"time_in.in_layer.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[
                f"time_text_embed.timestep_embedder.linear_1.{diffusers_lora_key}.bias"
            ] = original_state_dict.pop(f"time_in.in_layer.{lora_key}.bias")

        converted_state_dict[
            f"time_text_embed.timestep_embedder.linear_2.{diffusers_lora_key}.weight"
        ] = original_state_dict.pop(f"time_in.out_layer.{lora_key}.weight")
        if f"time_in.out_layer.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[
                f"time_text_embed.timestep_embedder.linear_2.{diffusers_lora_key}.bias"
            ] = original_state_dict.pop(f"time_in.out_layer.{lora_key}.bias")

        ## time_text_embed.text_embedder <- vector_in
        converted_state_dict[
            f"time_text_embed.text_embedder.linear_1.{diffusers_lora_key}.weight"
        ] = original_state_dict.pop(f"vector_in.in_layer.{lora_key}.weight")
        if f"vector_in.in_layer.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[
                f"time_text_embed.text_embedder.linear_1.{diffusers_lora_key}.bias"
            ] = original_state_dict.pop(f"vector_in.in_layer.{lora_key}.bias")

        converted_state_dict[
            f"time_text_embed.text_embedder.linear_2.{diffusers_lora_key}.weight"
        ] = original_state_dict.pop(f"vector_in.out_layer.{lora_key}.weight")
        if f"vector_in.out_layer.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[
                f"time_text_embed.text_embedder.linear_2.{diffusers_lora_key}.bias"
            ] = original_state_dict.pop(f"vector_in.out_layer.{lora_key}.bias")

        # guidance
        has_guidance = any("guidance" in k for k in original_state_dict)
        if has_guidance:
            converted_state_dict[
                f"time_text_embed.guidance_embedder.linear_1.{diffusers_lora_key}.weight"
            ] = original_state_dict.pop(f"guidance_in.in_layer.{lora_key}.weight")
            if f"guidance_in.in_layer.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"time_text_embed.guidance_embedder.linear_1.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"guidance_in.in_layer.{lora_key}.bias")

            converted_state_dict[
                f"time_text_embed.guidance_embedder.linear_2.{diffusers_lora_key}.weight"
            ] = original_state_dict.pop(f"guidance_in.out_layer.{lora_key}.weight")
            if f"guidance_in.out_layer.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"time_text_embed.guidance_embedder.linear_2.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"guidance_in.out_layer.{lora_key}.bias")

        # context_embedder
        converted_state_dict[f"context_embedder.{diffusers_lora_key}.weight"] = original_state_dict.pop(
            f"txt_in.{lora_key}.weight"
        )
        if f"txt_in.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[f"context_embedder.{diffusers_lora_key}.bias"] = original_state_dict.pop(
                f"txt_in.{lora_key}.bias"
            )

        # x_embedder
        converted_state_dict[f"x_embedder.{diffusers_lora_key}.weight"] = original_state_dict.pop(
            f"img_in.{lora_key}.weight"
        )
        if f"img_in.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[f"x_embedder.{diffusers_lora_key}.bias"] = original_state_dict.pop(
                f"img_in.{lora_key}.bias"
            )

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."

        for lora_key, diffusers_lora_key in zip(["lora_A", "lora_B"], ["lora_A", "lora_B"]):
            # norms
            converted_state_dict[f"{block_prefix}norm1.linear.{diffusers_lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mod.lin.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mod.lin.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"{block_prefix}norm1.linear.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"double_blocks.{i}.img_mod.lin.{lora_key}.bias")

            converted_state_dict[
                f"{block_prefix}norm1_context.linear.{diffusers_lora_key}.weight"
            ] = original_state_dict.pop(f"double_blocks.{i}.txt_mod.lin.{lora_key}.weight")
            if f"double_blocks.{i}.txt_mod.lin.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"{block_prefix}norm1_context.linear.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"double_blocks.{i}.txt_mod.lin.{lora_key}.bias")

            # Q, K, V
            if lora_key == "lora_A":
                sample_lora_weight = original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.to_v.{diffusers_lora_key}.weight"] = torch.cat(
                    [sample_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{diffusers_lora_key}.weight"] = torch.cat(
                    [sample_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.to_k.{diffusers_lora_key}.weight"] = torch.cat(
                    [sample_lora_weight]
                )

                context_lora_weight = original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{diffusers_lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{diffusers_lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{diffusers_lora_key}.weight"] = torch.cat(
                    [context_lora_weight]
                )
            else:
                sample_q, sample_k, sample_v = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.weight"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{diffusers_lora_key}.weight"] = torch.cat([sample_q])
                converted_state_dict[f"{block_prefix}attn.to_k.{diffusers_lora_key}.weight"] = torch.cat([sample_k])
                converted_state_dict[f"{block_prefix}attn.to_v.{diffusers_lora_key}.weight"] = torch.cat([sample_v])

                context_q, context_k, context_v = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.weight"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{diffusers_lora_key}.weight"] = torch.cat(
                    [context_q]
                )
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{diffusers_lora_key}.weight"] = torch.cat(
                    [context_k]
                )
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{diffusers_lora_key}.weight"] = torch.cat(
                    [context_v]
                )

            if f"double_blocks.{i}.img_attn.qkv.{lora_key}.bias" in original_state_dict.keys():
                sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.{lora_key}.bias"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{diffusers_lora_key}.bias"] = torch.cat([sample_q_bias])
                converted_state_dict[f"{block_prefix}attn.to_k.{diffusers_lora_key}.bias"] = torch.cat([sample_k_bias])
                converted_state_dict[f"{block_prefix}attn.to_v.{diffusers_lora_key}.bias"] = torch.cat([sample_v_bias])

            if f"double_blocks.{i}.txt_attn.qkv.{lora_key}.bias" in original_state_dict.keys():
                context_q_bias, context_k_bias, context_v_bias = torch.chunk(
                    original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.{lora_key}.bias"), 3, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.add_q_proj.{diffusers_lora_key}.bias"] = torch.cat(
                    [context_q_bias]
                )
                converted_state_dict[f"{block_prefix}attn.add_k_proj.{diffusers_lora_key}.bias"] = torch.cat(
                    [context_k_bias]
                )
                converted_state_dict[f"{block_prefix}attn.add_v_proj.{diffusers_lora_key}.bias"] = torch.cat(
                    [context_v_bias]
                )

            # ff img_mlp
            converted_state_dict[f"{block_prefix}ff.net.0.proj.{diffusers_lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.0.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mlp.0.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"{block_prefix}ff.net.0.proj{diffusers_lora_key}..bias"
                ] = original_state_dict.pop(f"double_blocks.{i}.img_mlp.0.{lora_key}.bias")

            converted_state_dict[f"{block_prefix}ff.net.2.{diffusers_lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.2.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_mlp.2.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[f"{block_prefix}ff.net.2.{diffusers_lora_key}.bias"] = original_state_dict.pop(
                    f"double_blocks.{i}.img_mlp.2.{lora_key}.bias"
                )

            converted_state_dict[
                f"{block_prefix}ff_context.net.0.proj.{diffusers_lora_key}.weight"
            ] = original_state_dict.pop(f"double_blocks.{i}.txt_mlp.0.{lora_key}.weight")
            if f"double_blocks.{i}.txt_mlp.0.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"{block_prefix}ff_context.net.0.proj.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"double_blocks.{i}.txt_mlp.0.{lora_key}.bias")

            converted_state_dict[
                f"{block_prefix}ff_context.net.2.{diffusers_lora_key}.weight"
            ] = original_state_dict.pop(f"double_blocks.{i}.txt_mlp.2.{lora_key}.weight")
            if f"double_blocks.{i}.txt_mlp.2.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"{block_prefix}ff_context.net.2.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"double_blocks.{i}.txt_mlp.2.{lora_key}.bias")

            # output projections.
            converted_state_dict[f"{block_prefix}attn.to_out.0.{diffusers_lora_key}.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_attn.proj.{lora_key}.weight"
            )
            if f"double_blocks.{i}.img_attn.proj.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"{block_prefix}attn.to_out.0.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"double_blocks.{i}.img_attn.proj.{lora_key}.bias")
            converted_state_dict[
                f"{block_prefix}attn.to_add_out.{diffusers_lora_key}.weight"
            ] = original_state_dict.pop(f"double_blocks.{i}.txt_attn.proj.{lora_key}.weight")
            if f"double_blocks.{i}.txt_attn.proj.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[
                    f"{block_prefix}attn.to_add_out.{diffusers_lora_key}.bias"
                ] = original_state_dict.pop(f"double_blocks.{i}.txt_attn.proj.{lora_key}.bias")

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

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."

        for lora_key, diffusers_lora_key in zip(["lora_A", "lora_B"], ["lora_A", "lora_B"]):
            # norm.linear  <- single_blocks.0.modulation.lin
            converted_state_dict[f"{block_prefix}norm.linear.{diffusers_lora_key}.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.modulation.lin.{lora_key}.weight"
            )
            if f"single_blocks.{i}.modulation.lin.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[f"{block_prefix}norm.linear.{diffusers_lora_key}.bias"] = original_state_dict.pop(
                    f"single_blocks.{i}.modulation.lin.{lora_key}.bias"
                )

            # Q, K, V, mlp
            mlp_hidden_dim = int(inner_dim * mlp_ratio)
            split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)

            if lora_key == "lora_A":
                lora_weight = original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.weight")
                converted_state_dict[f"{block_prefix}attn.to_q.{diffusers_lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_k.{diffusers_lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}attn.to_v.{diffusers_lora_key}.weight"] = torch.cat([lora_weight])
                converted_state_dict[f"{block_prefix}proj_mlp.{diffusers_lora_key}.weight"] = torch.cat([lora_weight])

                if f"single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict.keys():
                    lora_bias = original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.bias")
                    converted_state_dict[f"{block_prefix}attn.to_q.{diffusers_lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{diffusers_lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{diffusers_lora_key}.bias"] = torch.cat([lora_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{diffusers_lora_key}.bias"] = torch.cat([lora_bias])
            else:
                q, k, v, mlp = torch.split(
                    original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.weight"), split_size, dim=0
                )
                converted_state_dict[f"{block_prefix}attn.to_q.{diffusers_lora_key}.weight"] = torch.cat([q])
                converted_state_dict[f"{block_prefix}attn.to_k.{diffusers_lora_key}.weight"] = torch.cat([k])
                converted_state_dict[f"{block_prefix}attn.to_v.{diffusers_lora_key}.weight"] = torch.cat([v])
                converted_state_dict[f"{block_prefix}proj_mlp.{diffusers_lora_key}.weight"] = torch.cat([mlp])

                if f"single_blocks.{i}.linear1.{lora_key}.bias" in original_state_dict.keys():
                    q_bias, k_bias, v_bias, mlp_bias = torch.split(
                        original_state_dict.pop(f"single_blocks.{i}.linear1.{lora_key}.bias"), split_size, dim=0
                    )
                    converted_state_dict[f"{block_prefix}attn.to_q.{diffusers_lora_key}.bias"] = torch.cat([q_bias])
                    converted_state_dict[f"{block_prefix}attn.to_k.{diffusers_lora_key}.bias"] = torch.cat([k_bias])
                    converted_state_dict[f"{block_prefix}attn.to_v.{diffusers_lora_key}.bias"] = torch.cat([v_bias])
                    converted_state_dict[f"{block_prefix}proj_mlp.{diffusers_lora_key}.bias"] = torch.cat([mlp_bias])

            # output projections.
            converted_state_dict[f"{block_prefix}proj_out.{diffusers_lora_key}.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.linear2.{lora_key}.weight"
            )
            if f"single_blocks.{i}.linear2.{lora_key}.bias" in original_state_dict.keys():
                converted_state_dict[f"{block_prefix}proj_out.{diffusers_lora_key}.bias"] = original_state_dict.pop(
                    f"single_blocks.{i}.linear2.{lora_key}.bias"
                )

        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )

    for lora_key, diffusers_lora_key in zip(["lora_A", "lora_B"], ["lora_A", "lora_B"]):
        converted_state_dict[f"proj_out.{diffusers_lora_key}.weight"] = original_state_dict.pop(
            f"final_layer.linear.{lora_key}.weight"
        )
        if f"final_layer.linear.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[f"proj_out.{diffusers_lora_key}.bias"] = original_state_dict.pop(
                f"final_layer.linear.{lora_key}.bias"
            )

        converted_state_dict[f"norm_out.linear.{diffusers_lora_key}.weight"] = swap_scale_shift(
            original_state_dict.pop(f"final_layer.adaLN_modulation.1.{lora_key}.weight")
        )
        if f"final_layer.adaLN_modulation.1.{lora_key}.bias" in original_state_dict.keys():
            converted_state_dict[f"norm_out.linear.{diffusers_lora_key}.bias"] = swap_scale_shift(
                original_state_dict.pop(f"final_layer.adaLN_modulation.1.{lora_key}.bias")
            )

    print("Remaining:", original_state_dict.keys())

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


def main(args):
    original_ckpt = load_original_checkpoint(args)

    num_layers = 19
    num_single_layers = 38
    inner_dim = 3072
    mlp_ratio = 4.0

    converted_control_lora_state_dict = convert_flux_control_lora_checkpoint_to_diffusers(
        original_ckpt, num_layers, num_single_layers, inner_dim, mlp_ratio
    )
    safetensors.torch.save_file(converted_control_lora_state_dict, args.output_path)


if __name__ == "__main__":
    main(args)
