# Convert the original UniDiffuser checkpoints into diffusers equivalents.

import argparse
from argparse import Namespace

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    GPT2Tokenizer,
)

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UniDiffuserModel,
    UniDiffuserPipeline,
    UniDiffuserTextDecoder,
)


SCHEDULER_CONFIG = Namespace(
    **{
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "solver_order": 3,
    }
)


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.shave_segments
def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_vae_resnet_paths
def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_vae_attention_paths
def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "to_q.weight")
        new_item = new_item.replace("q.bias", "to_q.bias")

        new_item = new_item.replace("k.weight", "to_k.weight")
        new_item = new_item.replace("k.bias", "to_k.bias")

        new_item = new_item.replace("v.weight", "to_v.weight")
        new_item = new_item.replace("v.bias", "to_v.bias")

        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.conv_attn_to_linear
def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


# Modified from diffusers.pipelines.stable_diffusion.convert_from_ckpt.assign_to_checkpoint
# config.num_head_channels => num_head_channels
def assign_to_checkpoint(
    paths,
    checkpoint,
    old_checkpoint,
    attention_paths_to_split=None,
    additional_replacements=None,
    num_head_channels=1,
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // num_head_channels // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        is_attn_weight = "proj_attn.weight" in new_path or ("attentions" in new_path and "to_" in new_path)
        shape = old_checkpoint[path["old"]].shape
        if is_attn_weight and len(shape) == 3:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        elif is_attn_weight and len(shape) == 4:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def create_vae_diffusers_config(config_type):
    # Hardcoded for now
    if args.config_type == "test":
        vae_config = create_vae_diffusers_config_test()
    elif args.config_type == "big":
        vae_config = create_vae_diffusers_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )
    return vae_config


def create_unidiffuser_unet_config(config_type, version):
    # Hardcoded for now
    if args.config_type == "test":
        unet_config = create_unidiffuser_unet_config_test()
    elif args.config_type == "big":
        unet_config = create_unidiffuser_unet_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )
    # Unidiffuser-v1 uses data type embeddings
    if version == 1:
        unet_config["use_data_type_embedding"] = True
    return unet_config


def create_text_decoder_config(config_type):
    # Hardcoded for now
    if args.config_type == "test":
        text_decoder_config = create_text_decoder_config_test()
    elif args.config_type == "big":
        text_decoder_config = create_text_decoder_config_big()
    else:
        raise NotImplementedError(
            f"Config type {config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )
    return text_decoder_config


# Hardcoded configs for test versions of the UniDiffuser models, corresponding to those in the fast default tests.
def create_vae_diffusers_config_test():
    vae_config = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [32, 64],
        "latent_channels": 4,
        "layers_per_block": 1,
    }
    return vae_config


def create_unidiffuser_unet_config_test():
    unet_config = {
        "text_dim": 32,
        "clip_img_dim": 32,
        "num_text_tokens": 77,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 2,
        "dropout": 0.0,
        "norm_num_groups": 32,
        "attention_bias": False,
        "sample_size": 16,
        "patch_size": 2,
        "activation_fn": "gelu",
        "num_embeds_ada_norm": 1000,
        "norm_type": "layer_norm",
        "block_type": "unidiffuser",
        "pre_layer_norm": False,
        "use_timestep_embedding": False,
        "norm_elementwise_affine": True,
        "use_patch_pos_embed": False,
        "ff_final_dropout": True,
        "use_data_type_embedding": False,
    }
    return unet_config


def create_text_decoder_config_test():
    text_decoder_config = {
        "prefix_length": 77,
        "prefix_inner_dim": 32,
        "prefix_hidden_dim": 32,
        "vocab_size": 1025,  # 1024 + 1 for new EOS token
        "n_positions": 1024,
        "n_embd": 32,
        "n_layer": 5,
        "n_head": 4,
        "n_inner": 37,
        "activation_function": "gelu",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
    }
    return text_decoder_config


# Hardcoded configs for the UniDiffuser V1 model at https://huggingface.co/thu-ml/unidiffuser-v1
# See also https://github.com/thu-ml/unidiffuser/blob/main/configs/sample_unidiffuser_v1.py
def create_vae_diffusers_config_big():
    vae_config = {
        "sample_size": 256,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [128, 256, 512, 512],
        "latent_channels": 4,
        "layers_per_block": 2,
    }
    return vae_config


def create_unidiffuser_unet_config_big():
    unet_config = {
        "text_dim": 64,
        "clip_img_dim": 512,
        "num_text_tokens": 77,
        "num_attention_heads": 24,
        "attention_head_dim": 64,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 30,
        "dropout": 0.0,
        "norm_num_groups": 32,
        "attention_bias": False,
        "sample_size": 64,
        "patch_size": 2,
        "activation_fn": "gelu",
        "num_embeds_ada_norm": 1000,
        "norm_type": "layer_norm",
        "block_type": "unidiffuser",
        "pre_layer_norm": False,
        "use_timestep_embedding": False,
        "norm_elementwise_affine": True,
        "use_patch_pos_embed": False,
        "ff_final_dropout": True,
        "use_data_type_embedding": False,
    }
    return unet_config


# From https://huggingface.co/gpt2/blob/main/config.json, the GPT2 checkpoint used by UniDiffuser
def create_text_decoder_config_big():
    text_decoder_config = {
        "prefix_length": 77,
        "prefix_inner_dim": 768,
        "prefix_hidden_dim": 64,
        "vocab_size": 50258,  # 50257 + 1 for new EOS token
        "n_positions": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_inner": 3072,
        "activation_function": "gelu",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
    }
    return text_decoder_config


# Based on diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_vae_checkpoint
def convert_vae_to_diffusers(ckpt, diffusers_model, num_head_channels=1):
    """
    Converts a UniDiffuser autoencoder_kl.pth checkpoint to a diffusers AutoencoderKL.
    """
    # autoencoder_kl.pth ckpt is a torch state dict
    vae_state_dict = torch.load(ckpt, map_location="cpu")

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            num_head_channels=num_head_channels,  # not used in vae
        )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            num_head_channels=num_head_channels,  # not used in vae
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        num_head_channels=num_head_channels,  # not used in vae
    )
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            num_head_channels=num_head_channels,  # not used in vae
        )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            num_head_channels=num_head_channels,  # not used in vae
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        num_head_channels=num_head_channels,  # not used in vae
    )
    conv_attn_to_linear(new_checkpoint)

    missing_keys, unexpected_keys = diffusers_model.load_state_dict(new_checkpoint)
    for missing_key in missing_keys:
        print(f"Missing key: {missing_key}")
    for unexpected_key in unexpected_keys:
        print(f"Unexpected key: {unexpected_key}")

    return diffusers_model


def convert_uvit_block_to_diffusers_block(
    uvit_state_dict,
    new_state_dict,
    block_prefix,
    new_prefix="transformer.transformer_",
    skip_connection=False,
):
    """
    Maps the keys in a UniDiffuser transformer block (`Block`) to the keys in a diffusers transformer block
    (`UTransformerBlock`/`UniDiffuserBlock`).
    """
    prefix = new_prefix + block_prefix
    if skip_connection:
        new_state_dict[prefix + ".skip.skip_linear.weight"] = uvit_state_dict[block_prefix + ".skip_linear.weight"]
        new_state_dict[prefix + ".skip.skip_linear.bias"] = uvit_state_dict[block_prefix + ".skip_linear.bias"]
        new_state_dict[prefix + ".skip.norm.weight"] = uvit_state_dict[block_prefix + ".norm1.weight"]
        new_state_dict[prefix + ".skip.norm.bias"] = uvit_state_dict[block_prefix + ".norm1.bias"]

        # Create the prefix string for out_blocks.
        prefix += ".block"

    # Split up attention qkv.weight into to_q.weight, to_k.weight, to_v.weight
    qkv = uvit_state_dict[block_prefix + ".attn.qkv.weight"]
    new_attn_keys = [".attn1.to_q.weight", ".attn1.to_k.weight", ".attn1.to_v.weight"]
    new_attn_keys = [prefix + key for key in new_attn_keys]
    shape = qkv.shape[0] // len(new_attn_keys)
    for i, attn_key in enumerate(new_attn_keys):
        new_state_dict[attn_key] = qkv[i * shape : (i + 1) * shape]

    new_state_dict[prefix + ".attn1.to_out.0.weight"] = uvit_state_dict[block_prefix + ".attn.proj.weight"]
    new_state_dict[prefix + ".attn1.to_out.0.bias"] = uvit_state_dict[block_prefix + ".attn.proj.bias"]
    new_state_dict[prefix + ".norm1.weight"] = uvit_state_dict[block_prefix + ".norm2.weight"]
    new_state_dict[prefix + ".norm1.bias"] = uvit_state_dict[block_prefix + ".norm2.bias"]
    new_state_dict[prefix + ".ff.net.0.proj.weight"] = uvit_state_dict[block_prefix + ".mlp.fc1.weight"]
    new_state_dict[prefix + ".ff.net.0.proj.bias"] = uvit_state_dict[block_prefix + ".mlp.fc1.bias"]
    new_state_dict[prefix + ".ff.net.2.weight"] = uvit_state_dict[block_prefix + ".mlp.fc2.weight"]
    new_state_dict[prefix + ".ff.net.2.bias"] = uvit_state_dict[block_prefix + ".mlp.fc2.bias"]
    new_state_dict[prefix + ".norm3.weight"] = uvit_state_dict[block_prefix + ".norm3.weight"]
    new_state_dict[prefix + ".norm3.bias"] = uvit_state_dict[block_prefix + ".norm3.bias"]

    return uvit_state_dict, new_state_dict


def convert_uvit_to_diffusers(ckpt, diffusers_model):
    """
    Converts a UniDiffuser uvit_v*.pth checkpoint to a diffusers UniDiffusersModel.
    """
    # uvit_v*.pth ckpt is a torch state dict
    uvit_state_dict = torch.load(ckpt, map_location="cpu")

    new_state_dict = {}

    # Input layers
    new_state_dict["vae_img_in.proj.weight"] = uvit_state_dict["patch_embed.proj.weight"]
    new_state_dict["vae_img_in.proj.bias"] = uvit_state_dict["patch_embed.proj.bias"]
    new_state_dict["clip_img_in.weight"] = uvit_state_dict["clip_img_embed.weight"]
    new_state_dict["clip_img_in.bias"] = uvit_state_dict["clip_img_embed.bias"]
    new_state_dict["text_in.weight"] = uvit_state_dict["text_embed.weight"]
    new_state_dict["text_in.bias"] = uvit_state_dict["text_embed.bias"]

    new_state_dict["pos_embed"] = uvit_state_dict["pos_embed"]

    # Handle data type token embeddings for UniDiffuser-v1
    if "token_embedding.weight" in uvit_state_dict and diffusers_model.use_data_type_embedding:
        new_state_dict["data_type_pos_embed_token"] = uvit_state_dict["pos_embed_token"]
        new_state_dict["data_type_token_embedding.weight"] = uvit_state_dict["token_embedding.weight"]

    # Also initialize the PatchEmbedding in UTransformer2DModel with the PatchEmbedding from the checkpoint.
    # This isn't used in the current implementation, so might want to remove.
    new_state_dict["transformer.pos_embed.proj.weight"] = uvit_state_dict["patch_embed.proj.weight"]
    new_state_dict["transformer.pos_embed.proj.bias"] = uvit_state_dict["patch_embed.proj.bias"]

    # Output layers
    new_state_dict["transformer.norm_out.weight"] = uvit_state_dict["norm.weight"]
    new_state_dict["transformer.norm_out.bias"] = uvit_state_dict["norm.bias"]

    new_state_dict["vae_img_out.weight"] = uvit_state_dict["decoder_pred.weight"]
    new_state_dict["vae_img_out.bias"] = uvit_state_dict["decoder_pred.bias"]
    new_state_dict["clip_img_out.weight"] = uvit_state_dict["clip_img_out.weight"]
    new_state_dict["clip_img_out.bias"] = uvit_state_dict["clip_img_out.bias"]
    new_state_dict["text_out.weight"] = uvit_state_dict["text_out.weight"]
    new_state_dict["text_out.bias"] = uvit_state_dict["text_out.bias"]

    # in_blocks
    in_blocks_prefixes = {".".join(layer.split(".")[:2]) for layer in uvit_state_dict if "in_blocks" in layer}
    for in_block_prefix in list(in_blocks_prefixes):
        convert_uvit_block_to_diffusers_block(uvit_state_dict, new_state_dict, in_block_prefix)

    # mid_block
    # Assume there's only one mid block
    convert_uvit_block_to_diffusers_block(uvit_state_dict, new_state_dict, "mid_block")

    # out_blocks
    out_blocks_prefixes = {".".join(layer.split(".")[:2]) for layer in uvit_state_dict if "out_blocks" in layer}
    for out_block_prefix in list(out_blocks_prefixes):
        convert_uvit_block_to_diffusers_block(uvit_state_dict, new_state_dict, out_block_prefix, skip_connection=True)

    missing_keys, unexpected_keys = diffusers_model.load_state_dict(new_state_dict)
    for missing_key in missing_keys:
        print(f"Missing key: {missing_key}")
    for unexpected_key in unexpected_keys:
        print(f"Unexpected key: {unexpected_key}")

    return diffusers_model


def convert_caption_decoder_to_diffusers(ckpt, diffusers_model):
    """
    Converts a UniDiffuser caption_decoder.pth checkpoint to a diffusers UniDiffuserTextDecoder.
    """
    # caption_decoder.pth ckpt is a torch state dict
    checkpoint_state_dict = torch.load(ckpt, map_location="cpu")
    decoder_state_dict = {}
    # Remove the "module." prefix, if necessary
    caption_decoder_key = "module."
    for key in checkpoint_state_dict:
        if key.startswith(caption_decoder_key):
            decoder_state_dict[key.replace(caption_decoder_key, "")] = checkpoint_state_dict.get(key)
        else:
            decoder_state_dict[key] = checkpoint_state_dict.get(key)

    new_state_dict = {}

    # Encoder and Decoder
    new_state_dict["encode_prefix.weight"] = decoder_state_dict["encode_prefix.weight"]
    new_state_dict["encode_prefix.bias"] = decoder_state_dict["encode_prefix.bias"]
    new_state_dict["decode_prefix.weight"] = decoder_state_dict["decode_prefix.weight"]
    new_state_dict["decode_prefix.bias"] = decoder_state_dict["decode_prefix.bias"]

    # Internal GPT2LMHeadModel transformer model
    for key, val in decoder_state_dict.items():
        if key.startswith("gpt"):
            suffix = key[len("gpt") :]
            new_state_dict["transformer" + suffix] = val

    missing_keys, unexpected_keys = diffusers_model.load_state_dict(new_state_dict)
    for missing_key in missing_keys:
        print(f"Missing key: {missing_key}")
    for unexpected_key in unexpected_keys:
        print(f"Unexpected key: {unexpected_key}")

    return diffusers_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--caption_decoder_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to caption decoder checkpoint to convert.",
    )
    parser.add_argument(
        "--uvit_checkpoint_path", default=None, type=str, required=False, help="Path to U-ViT checkpoint to convert."
    )
    parser.add_argument(
        "--vae_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to VAE checkpoint to convert.",
    )
    parser.add_argument(
        "--pipeline_output_path",
        default=None,
        type=str,
        required=True,
        help="Path to save the output pipeline to.",
    )
    parser.add_argument(
        "--config_type",
        default="test",
        type=str,
        help=(
            "Config type to use. Should be 'test' to create small models for testing or 'big' to convert a full"
            " checkpoint."
        ),
    )
    parser.add_argument(
        "--version",
        default=0,
        type=int,
        help="The UniDiffuser model type to convert to. Should be 0 for UniDiffuser-v0 and 1 for UniDiffuser-v1.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Whether to use safetensors/safe seialization when saving the pipeline.",
    )

    args = parser.parse_args()

    # Convert the VAE model.
    if args.vae_checkpoint_path is not None:
        vae_config = create_vae_diffusers_config(args.config_type)
        vae = AutoencoderKL(**vae_config)
        vae = convert_vae_to_diffusers(args.vae_checkpoint_path, vae)

    # Convert the U-ViT ("unet") model.
    if args.uvit_checkpoint_path is not None:
        unet_config = create_unidiffuser_unet_config(args.config_type, args.version)
        unet = UniDiffuserModel(**unet_config)
        unet = convert_uvit_to_diffusers(args.uvit_checkpoint_path, unet)

    # Convert the caption decoder ("text_decoder") model.
    if args.caption_decoder_checkpoint_path is not None:
        text_decoder_config = create_text_decoder_config(args.config_type)
        text_decoder = UniDiffuserTextDecoder(**text_decoder_config)
        text_decoder = convert_caption_decoder_to_diffusers(args.caption_decoder_checkpoint_path, text_decoder)

    # Scheduler is the same for both the test and big models.
    scheduler_config = SCHEDULER_CONFIG
    scheduler = DPMSolverMultistepScheduler(
        beta_start=scheduler_config.beta_start,
        beta_end=scheduler_config.beta_end,
        beta_schedule=scheduler_config.beta_schedule,
        solver_order=scheduler_config.solver_order,
    )

    if args.config_type == "test":
        # Make a small random CLIPTextModel
        torch.manual_seed(0)
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(clip_text_encoder_config)
        clip_tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # Make a small random CLIPVisionModel and accompanying CLIPImageProcessor
        torch.manual_seed(0)
        clip_image_encoder_config = CLIPVisionConfig(
            image_size=32,
            patch_size=2,
            num_channels=3,
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            dropout=0.1,
            attention_dropout=0.1,
            initializer_range=0.02,
        )
        image_encoder = CLIPVisionModelWithProjection(clip_image_encoder_config)
        image_processor = CLIPImageProcessor(crop_size=32, size=32)

        # Note that the text_decoder should already have its token embeddings resized.
        text_tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        eos = "<|EOS|>"
        special_tokens_dict = {"eos_token": eos}
        text_tokenizer.add_special_tokens(special_tokens_dict)
    elif args.config_type == "big":
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Note that the text_decoder should already have its token embeddings resized.
        text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        eos = "<|EOS|>"
        special_tokens_dict = {"eos_token": eos}
        text_tokenizer.add_special_tokens(special_tokens_dict)
    else:
        raise NotImplementedError(
            f"Config type {args.config_type} is not implemented, currently only config types"
            " 'test' and 'big' are available."
        )

    pipeline = UniDiffuserPipeline(
        vae=vae,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        clip_image_processor=image_processor,
        clip_tokenizer=clip_tokenizer,
        text_decoder=text_decoder,
        text_tokenizer=text_tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.save_pretrained(args.pipeline_output_path, safe_serialization=args.safe_serialization)
