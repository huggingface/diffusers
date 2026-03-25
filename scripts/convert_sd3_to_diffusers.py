import argparse
from contextlib import nullcontext

import safetensors.torch
import torch
from accelerate import init_empty_weights

from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from diffusers.models.model_loading_utils import load_model_dict_into_meta
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available() else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str)

args = parser.parse_args()


def load_original_checkpoint(ckpt_path):
    original_state_dict = safetensors.torch.load_file(ckpt_path)
    keys = list(original_state_dict.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            original_state_dict[k.replace("model.diffusion_model.", "")] = original_state_dict.pop(k)

    return original_state_dict


# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_sd3_transformer_checkpoint_to_diffusers(
    original_state_dict, num_layers, caption_projection_dim, dual_attention_layers, has_qk_norm
):
    converted_state_dict = {}

    # Positional and patch embeddings.
    converted_state_dict["pos_embed.pos_embed"] = original_state_dict.pop("pos_embed")
    converted_state_dict["pos_embed.proj.weight"] = original_state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = original_state_dict.pop("x_embedder.proj.bias")

    # Timestep embeddings.
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
        "t_embedder.mlp.0.bias"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
        "t_embedder.mlp.2.bias"
    )

    # Context projections.
    converted_state_dict["context_embedder.weight"] = original_state_dict.pop("context_embedder.weight")
    converted_state_dict["context_embedder.bias"] = original_state_dict.pop("context_embedder.bias")

    # Pooled context projection.
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop(
        "y_embedder.mlp.0.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop(
        "y_embedder.mlp.0.bias"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop(
        "y_embedder.mlp.2.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop(
        "y_embedder.mlp.2.bias"
    )

    # Transformer blocks 🎸.
    for i in range(num_layers):
        # Q, K, V
        sample_q, sample_k, sample_v = torch.chunk(
            original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.qkv.weight"), 3, dim=0
        )
        context_q, context_k, context_v = torch.chunk(
            original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.qkv.weight"), 3, dim=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
            original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.qkv.bias"), 3, dim=0
        )
        context_q_bias, context_k_bias, context_v_bias = torch.chunk(
            original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.qkv.bias"), 3, dim=0
        )

        converted_state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_q.bias"] = torch.cat([sample_q_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_k.bias"] = torch.cat([sample_k_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = torch.cat([sample_v])
        converted_state_dict[f"transformer_blocks.{i}.attn.to_v.bias"] = torch.cat([sample_v_bias])

        converted_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.weight"] = torch.cat([context_q])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.bias"] = torch.cat([context_q_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.weight"] = torch.cat([context_k])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.bias"] = torch.cat([context_k_bias])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.weight"] = torch.cat([context_v])
        converted_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.bias"] = torch.cat([context_v_bias])

        # qk norm
        if has_qk_norm:
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_q.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.x_block.attn.ln_q.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_k.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.x_block.attn.ln_k.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_added_q.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.attn.ln_q.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_added_k.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.attn.ln_k.weight"
            )

        # output projections.
        converted_state_dict[f"transformer_blocks.{i}.attn.to_out.0.weight"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.attn.to_out.0.bias"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.attn.proj.bias"
        )
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.attn.to_add_out.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.attn.proj.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn.to_add_out.bias"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.attn.proj.bias"
            )

        # attn2
        if i in dual_attention_layers:
            # Q, K, V
            sample_q2, sample_k2, sample_v2 = torch.chunk(
                original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.qkv.weight"), 3, dim=0
            )
            sample_q2_bias, sample_k2_bias, sample_v2_bias = torch.chunk(
                original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.qkv.bias"), 3, dim=0
            )
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.weight"] = torch.cat([sample_q2])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.bias"] = torch.cat([sample_q2_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.weight"] = torch.cat([sample_k2])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.bias"] = torch.cat([sample_k2_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.weight"] = torch.cat([sample_v2])
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.bias"] = torch.cat([sample_v2_bias])

            # qk norm
            if has_qk_norm:
                converted_state_dict[f"transformer_blocks.{i}.attn2.norm_q.weight"] = original_state_dict.pop(
                    f"joint_blocks.{i}.x_block.attn2.ln_q.weight"
                )
                converted_state_dict[f"transformer_blocks.{i}.attn2.norm_k.weight"] = original_state_dict.pop(
                    f"joint_blocks.{i}.x_block.attn2.ln_k.weight"
                )

            # output projections.
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.x_block.attn2.proj.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.bias"] = original_state_dict.pop(
                f"joint_blocks.{i}.x_block.attn2.proj.bias"
            )

        # norms.
        converted_state_dict[f"transformer_blocks.{i}.norm1.linear.weight"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.adaLN_modulation.1.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.norm1.linear.bias"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.adaLN_modulation.1.bias"
        )
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.adaLN_modulation.1.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.bias"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.adaLN_modulation.1.bias"
            )
        else:
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.weight"] = swap_scale_shift(
                original_state_dict.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.weight"),
                dim=caption_projection_dim,
            )
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.bias"] = swap_scale_shift(
                original_state_dict.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.bias"),
                dim=caption_projection_dim,
            )

        # ffs.
        converted_state_dict[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.mlp.fc1.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.net.0.proj.bias"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.mlp.fc1.bias"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.net.2.weight"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.mlp.fc2.weight"
        )
        converted_state_dict[f"transformer_blocks.{i}.ff.net.2.bias"] = original_state_dict.pop(
            f"joint_blocks.{i}.x_block.mlp.fc2.bias"
        )
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.0.proj.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.mlp.fc1.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.0.proj.bias"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.mlp.fc1.bias"
            )
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.2.weight"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.mlp.fc2.weight"
            )
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.2.bias"] = original_state_dict.pop(
                f"joint_blocks.{i}.context_block.mlp.fc2.bias"
            )

    # Final blocks.
    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.weight"), dim=caption_projection_dim
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.bias"), dim=caption_projection_dim
    )

    return converted_state_dict


def is_vae_in_checkpoint(original_state_dict):
    return ("first_stage_model.decoder.conv_in.weight" in original_state_dict) and (
        "first_stage_model.encoder.conv_in.weight" in original_state_dict
    )


def get_attn2_layers(state_dict):
    attn2_layers = []
    for key in state_dict.keys():
        if "attn2." in key:
            # Extract the layer number from the key
            layer_num = int(key.split(".")[1])
            attn2_layers.append(layer_num)
    return tuple(sorted(set(attn2_layers)))


def get_pos_embed_max_size(state_dict):
    num_patches = state_dict["pos_embed"].shape[1]
    pos_embed_max_size = int(num_patches**0.5)
    return pos_embed_max_size


def get_caption_projection_dim(state_dict):
    caption_projection_dim = state_dict["context_embedder.weight"].shape[0]
    return caption_projection_dim


def main(args):
    original_ckpt = load_original_checkpoint(args.checkpoint_path)
    original_dtype = next(iter(original_ckpt.values())).dtype

    # Initialize dtype with a default value
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

    if dtype != original_dtype:
        print(
            f"Checkpoint dtype {original_dtype} does not match requested dtype {dtype}. This can lead to unexpected results, proceed with caution."
        )

    num_layers = list(set(int(k.split(".", 2)[1]) for k in original_ckpt if "joint_blocks" in k))[-1] + 1  # noqa: C401

    caption_projection_dim = get_caption_projection_dim(original_ckpt)

    # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
    attn2_layers = get_attn2_layers(original_ckpt)

    # sd3.5 use qk norm("rms_norm")
    has_qk_norm = any("ln_q" in key for key in original_ckpt.keys())

    # sd3.5 2b use pox_embed_max_size=384 and sd3.0 and sd3.5 8b use 192
    pos_embed_max_size = get_pos_embed_max_size(original_ckpt)

    converted_transformer_state_dict = convert_sd3_transformer_checkpoint_to_diffusers(
        original_ckpt, num_layers, caption_projection_dim, attn2_layers, has_qk_norm
    )

    with CTX():
        transformer = SD3Transformer2DModel(
            sample_size=128,
            patch_size=2,
            in_channels=16,
            joint_attention_dim=4096,
            num_layers=num_layers,
            caption_projection_dim=caption_projection_dim,
            num_attention_heads=num_layers,
            pos_embed_max_size=pos_embed_max_size,
            qk_norm="rms_norm" if has_qk_norm else None,
            dual_attention_layers=attn2_layers,
        )
    if is_accelerate_available():
        load_model_dict_into_meta(transformer, converted_transformer_state_dict)
    else:
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)

    print("Saving SD3 Transformer in Diffusers format.")
    transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

    if is_vae_in_checkpoint(original_ckpt):
        with CTX():
            vae = AutoencoderKL.from_config(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="vae",
                latent_channels=16,
                use_post_quant_conv=False,
                use_quant_conv=False,
                scaling_factor=1.5305,
                shift_factor=0.0609,
            )
        converted_vae_state_dict = convert_ldm_vae_checkpoint(original_ckpt, vae.config)
        if is_accelerate_available():
            load_model_dict_into_meta(vae, converted_vae_state_dict)
        else:
            vae.load_state_dict(converted_vae_state_dict, strict=True)

        print("Saving SD3 Autoencoder in Diffusers format.")
        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")


if __name__ == "__main__":
    main(args)
