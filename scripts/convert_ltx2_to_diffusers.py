import argparse
import os
from contextlib import nullcontext
from typing import Any

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2LatentUpsamplePipeline,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2TextConnectors, LTX2Vocoder
from diffusers.utils.import_utils import is_accelerate_available


CTX = init_empty_weights if is_accelerate_available() else nullcontext


LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT = {
    # Input Patchify Projections
    "patchify_proj": "proj_in",
    "audio_patchify_proj": "audio_proj_in",
    # Modulation Parameters
    # Handle adaln_single --> time_embed, audioln_single --> audio_time_embed separately as the original keys are
    # substrings of the other modulation parameters below
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
    # Transformer Blocks
    # Per-Block Cross Attention Modulatin Parameters
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    # Attention QK Norms
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

LTX_2_0_VIDEO_VAE_RENAME_DICT = {
    # Encoder
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    # Decoder
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
    # Common
    # For all 3D ResNets
    "res_blocks": "resnets",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

LTX_2_0_AUDIO_VAE_RENAME_DICT = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

LTX_2_0_VOCODER_RENAME_DICT = {
    "ups": "upsamplers",
    "resblocks": "resnets",
    "conv_pre": "conv_in",
    "conv_post": "conv_out",
}

LTX_2_0_TEXT_ENCODER_RENAME_DICT = {
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    # Attention QK Norms
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}


def update_state_dict_inplace(state_dict: dict[str, Any], old_key: str, new_key: str) -> None:
    state_dict[new_key] = state_dict.pop(old_key)


def remove_keys_inplace(key: str, state_dict: dict[str, Any]) -> None:
    state_dict.pop(key)


def convert_ltx2_transformer_adaln_single(key: str, state_dict: dict[str, Any]) -> None:
    # Skip if not a weight, bias
    if ".weight" not in key and ".bias" not in key:
        return

    if key.startswith("adaln_single."):
        new_key = key.replace("adaln_single.", "time_embed.")
        param = state_dict.pop(key)
        state_dict[new_key] = param

    if key.startswith("audio_adaln_single."):
        new_key = key.replace("audio_adaln_single.", "audio_time_embed.")
        param = state_dict.pop(key)
        state_dict[new_key] = param

    return


def convert_ltx2_audio_vae_per_channel_statistics(key: str, state_dict: dict[str, Any]) -> None:
    if key.startswith("per_channel_statistics"):
        new_key = ".".join(["decoder", key])
        param = state_dict.pop(key)
        state_dict[new_key] = param

    return


LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "video_embeddings_connector": remove_keys_inplace,
    "audio_embeddings_connector": remove_keys_inplace,
    "adaln_single": convert_ltx2_transformer_adaln_single,
}

LTX_2_0_CONNECTORS_KEYS_RENAME_DICT = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    "text_embedding_projection.aggregate_embed": "text_proj_in",
    # Attention QK Norms
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

LTX_2_0_VAE_SPECIAL_KEYS_REMAP = {
    "per_channel_statistics.channel": remove_keys_inplace,
    "per_channel_statistics.mean-of-stds": remove_keys_inplace,
}

LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP = {}

LTX_2_0_VOCODER_SPECIAL_KEYS_REMAP = {}


def split_transformer_and_connector_state_dict(state_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    connector_prefixes = (
        "video_embeddings_connector",
        "audio_embeddings_connector",
        "transformer_1d_blocks",
        "text_embedding_projection.aggregate_embed",
        "connectors.",
        "video_connector",
        "audio_connector",
        "text_proj_in",
    )

    transformer_state_dict, connector_state_dict = {}, {}
    for key, value in state_dict.items():
        if key.startswith(connector_prefixes):
            connector_state_dict[key] = value
        else:
            transformer_state_dict[key] = value

    return transformer_state_dict, connector_state_dict


def get_ltx2_transformer_config(version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "test":
        # Produces a transformer of the same size as used in test_models_transformer_ltx2.py
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "in_channels": 4,
                "out_channels": 4,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 2,
                "attention_head_dim": 8,
                "cross_attention_dim": 16,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "audio_in_channels": 4,
                "audio_out_channels": 4,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 2,
                "audio_attention_head_dim": 4,
                "audio_cross_attention_dim": 8,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "num_layers": 2,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 16,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": False,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1,
            },
        }
        rename_dict = LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif version == "2.0":
        config = {
            "model_id": "diffusers-internal-dev/new-ltx-model",
            "diffusers_config": {
                "in_channels": 128,
                "out_channels": 128,
                "patch_size": 1,
                "patch_size_t": 1,
                "num_attention_heads": 32,
                "attention_head_dim": 128,
                "cross_attention_dim": 4096,
                "vae_scale_factors": (8, 32, 32),
                "pos_embed_max_pos": 20,
                "base_height": 2048,
                "base_width": 2048,
                "audio_in_channels": 128,
                "audio_out_channels": 128,
                "audio_patch_size": 1,
                "audio_patch_size_t": 1,
                "audio_num_attention_heads": 32,
                "audio_attention_head_dim": 64,
                "audio_cross_attention_dim": 2048,
                "audio_scale_factor": 4,
                "audio_pos_embed_max_pos": 20,
                "audio_sampling_rate": 16000,
                "audio_hop_length": 160,
                "num_layers": 48,
                "activation_fn": "gelu-approximate",
                "qk_norm": "rms_norm_across_heads",
                "norm_elementwise_affine": False,
                "norm_eps": 1e-6,
                "caption_channels": 3840,
                "attention_bias": True,
                "attention_out_bias": True,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_offset": 1,
                "timestep_scale_multiplier": 1000,
                "cross_attn_timestep_scale_multiplier": 1000,
                "rope_type": "split",
            },
        }
        rename_dict = LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT
        special_keys_remap = LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def get_ltx2_connectors_config(version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "test":
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "caption_channels": 16,
                "text_proj_in_factor": 3,
                "video_connector_num_attention_heads": 4,
                "video_connector_attention_head_dim": 8,
                "video_connector_num_layers": 1,
                "video_connector_num_learnable_registers": None,
                "audio_connector_num_attention_heads": 4,
                "audio_connector_attention_head_dim": 8,
                "audio_connector_num_layers": 1,
                "audio_connector_num_learnable_registers": None,
                "connector_rope_base_seq_len": 32,
                "rope_theta": 10000.0,
                "rope_double_precision": False,
                "causal_temporal_positioning": False,
            },
        }
    elif version == "2.0":
        config = {
            "model_id": "diffusers-internal-dev/new-ltx-model",
            "diffusers_config": {
                "caption_channels": 3840,
                "text_proj_in_factor": 49,
                "video_connector_num_attention_heads": 30,
                "video_connector_attention_head_dim": 128,
                "video_connector_num_layers": 2,
                "video_connector_num_learnable_registers": 128,
                "audio_connector_num_attention_heads": 30,
                "audio_connector_attention_head_dim": 128,
                "audio_connector_num_layers": 2,
                "audio_connector_num_learnable_registers": 128,
                "connector_rope_base_seq_len": 4096,
                "rope_theta": 10000.0,
                "rope_double_precision": True,
                "causal_temporal_positioning": False,
                "rope_type": "split",
            },
        }

    rename_dict = LTX_2_0_CONNECTORS_KEYS_RENAME_DICT
    special_keys_remap = {}

    return config, rename_dict, special_keys_remap


def convert_ltx2_transformer(original_state_dict: dict[str, Any], version: str) -> dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_transformer_config(version)
    diffusers_config = config["diffusers_config"]

    transformer_state_dict, _ = split_transformer_and_connector_state_dict(original_state_dict)

    with init_empty_weights():
        transformer = LTX2VideoTransformer3DModel.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(transformer_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(transformer_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(transformer_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, transformer_state_dict)

    transformer.load_state_dict(transformer_state_dict, strict=True, assign=True)
    return transformer


def convert_ltx2_connectors(original_state_dict: dict[str, Any], version: str) -> LTX2TextConnectors:
    config, rename_dict, special_keys_remap = get_ltx2_connectors_config(version)
    diffusers_config = config["diffusers_config"]

    _, connector_state_dict = split_transformer_and_connector_state_dict(original_state_dict)
    if len(connector_state_dict) == 0:
        raise ValueError("No connector weights found in the provided state dict.")

    with init_empty_weights():
        connectors = LTX2TextConnectors.from_config(diffusers_config)

    for key in list(connector_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(connector_state_dict, key, new_key)

    for key in list(connector_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, connector_state_dict)

    connectors.load_state_dict(connector_state_dict, strict=True, assign=True)
    return connectors


def get_ltx2_video_vae_config(
    version: str, timestep_conditioning: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "test":
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 2048),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 1024),
                "layers_per_block": (4, 6, 6, 2, 2),
                "decoder_layers_per_block": (5, 5, 5, 5),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True),
                "decoder_inject_noise": (False, False, False, False),
                "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                "upsample_residual": (True, True, True),
                "upsample_factor": (2, 2, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "reflect",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
        rename_dict = LTX_2_0_VIDEO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_VAE_SPECIAL_KEYS_REMAP
    elif version == "2.0":
        config = {
            "model_id": "diffusers-internal-dev/dummy-ltx2",
            "diffusers_config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "block_out_channels": (256, 512, 1024, 2048),
                "down_block_types": (
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                    "LTX2VideoDownBlock3D",
                ),
                "decoder_block_out_channels": (256, 512, 1024),
                "layers_per_block": (4, 6, 6, 2, 2),
                "decoder_layers_per_block": (5, 5, 5, 5),
                "spatio_temporal_scaling": (True, True, True, True),
                "decoder_spatio_temporal_scaling": (True, True, True),
                "decoder_inject_noise": (False, False, False, False),
                "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                "upsample_residual": (True, True, True),
                "upsample_factor": (2, 2, 2),
                "timestep_conditioning": timestep_conditioning,
                "patch_size": 4,
                "patch_size_t": 1,
                "resnet_norm_eps": 1e-6,
                "encoder_causal": True,
                "decoder_causal": False,
                "encoder_spatial_padding_mode": "zeros",
                "decoder_spatial_padding_mode": "reflect",
                "spatial_compression_ratio": 32,
                "temporal_compression_ratio": 8,
            },
        }
        rename_dict = LTX_2_0_VIDEO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_VAE_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def convert_ltx2_video_vae(
    original_state_dict: dict[str, Any], version: str, timestep_conditioning: bool
) -> dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_video_vae_config(version, timestep_conditioning)
    diffusers_config = config["diffusers_config"]

    with init_empty_weights():
        vae = AutoencoderKLLTX2Video.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def get_ltx2_audio_vae_config(version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "2.0":
        config = {
            "model_id": "diffusers-internal-dev/new-ltx-model",
            "diffusers_config": {
                "base_channels": 128,
                "output_channels": 2,
                "ch_mult": (1, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": None,
                "in_channels": 2,
                "resolution": 256,
                "latent_channels": 8,
                "norm_type": "pixel",
                "causality_axis": "height",
                "dropout": 0.0,
                "mid_block_add_attention": False,
                "sample_rate": 16000,
                "mel_hop_length": 160,
                "is_causal": True,
                "mel_bins": 64,
                "double_z": True,
            },
        }
        rename_dict = LTX_2_0_AUDIO_VAE_RENAME_DICT
        special_keys_remap = LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def convert_ltx2_audio_vae(original_state_dict: dict[str, Any], version: str) -> dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_audio_vae_config(version)
    diffusers_config = config["diffusers_config"]

    with init_empty_weights():
        vae = AutoencoderKLLTX2Audio.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def get_ltx2_vocoder_config(version: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if version == "2.0":
        config = {
            "model_id": "diffusers-internal-dev/new-ltx-model",
            "diffusers_config": {
                "in_channels": 128,
                "hidden_channels": 1024,
                "out_channels": 2,
                "upsample_kernel_sizes": [16, 15, 8, 4, 4],
                "upsample_factors": [6, 5, 2, 2, 2],
                "resnet_kernel_sizes": [3, 7, 11],
                "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "leaky_relu_negative_slope": 0.1,
                "output_sampling_rate": 24000,
            },
        }
        rename_dict = LTX_2_0_VOCODER_RENAME_DICT
        special_keys_remap = LTX_2_0_VOCODER_SPECIAL_KEYS_REMAP
    return config, rename_dict, special_keys_remap


def convert_ltx2_vocoder(original_state_dict: dict[str, Any], version: str) -> dict[str, Any]:
    config, rename_dict, special_keys_remap = get_ltx2_vocoder_config(version)
    diffusers_config = config["diffusers_config"]

    with init_empty_weights():
        vocoder = LTX2Vocoder.from_config(diffusers_config)

    # Handle official code --> diffusers key remapping via the remap dict
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    # Handle any special logic which can't be expressed by a simple 1:1 remapping with the handlers in
    # special_keys_remap
    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vocoder.load_state_dict(original_state_dict, strict=True, assign=True)
    return vocoder


def get_ltx2_spatial_latent_upsampler_config(version: str):
    if version == "2.0":
        config = {
            "in_channels": 128,
            "mid_channels": 1024,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
            "rational_spatial_scale": 2.0,
        }
    else:
        raise ValueError(f"Unsupported version: {version}")
    return config


def convert_ltx2_spatial_latent_upsampler(
    original_state_dict: dict[str, Any], config: dict[str, Any], dtype: torch.dtype
):
    with init_empty_weights():
        latent_upsampler = LTX2LatentUpsamplerModel(**config)

    latent_upsampler.load_state_dict(original_state_dict, strict=True, assign=True)
    latent_upsampler.to(dtype)
    return latent_upsampler


def load_original_checkpoint(args, filename: str | None) -> dict[str, Any]:
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


def load_hub_or_local_checkpoint(repo_id: str | None = None, filename: str | None = None) -> dict[str, Any]:
    if repo_id is None and filename is None:
        raise ValueError("Please supply at least one of `repo_id` or `filename`")

    if repo_id is not None:
        if filename is None:
            raise ValueError("If repo_id is specified, filename must also be specified.")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        ckpt_path = filename

    _, ext = os.path.splitext(ckpt_path)
    if ext in [".safetensors", ".sft"]:
        state_dict = safetensors.torch.load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    return state_dict


def get_model_state_dict_from_combined_ckpt(combined_ckpt: dict[str, Any], prefix: str) -> dict[str, Any]:
    # Ensure that the key prefix ends with a dot (.)
    if not prefix.endswith("."):
        prefix = prefix + "."

    model_state_dict = {}
    for param_name, param in combined_ckpt.items():
        if param_name.startswith(prefix):
            model_state_dict[param_name.replace(prefix, "")] = param

    if prefix == "model.diffusion_model.":
        # Some checkpoints store the text connector projection outside the diffusion model prefix.
        connector_key = "text_embedding_projection.aggregate_embed.weight"
        if connector_key in combined_ckpt and connector_key not in model_state_dict:
            model_state_dict[connector_key] = combined_ckpt[connector_key]

    return model_state_dict


def get_args():
    parser = argparse.ArgumentParser()

    def none_or_str(value: str):
        if isinstance(value, str) and value.lower() == "none":
            return None
        return value

    parser.add_argument(
        "--original_state_dict_repo_id",
        default="Lightricks/LTX-2",
        type=none_or_str,
        help="HF Hub repo id with LTX 2.0 checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Local checkpoint path for LTX 2.0. Will be used if `original_state_dict_repo_id` is not specified.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="2.0",
        choices=["test", "2.0"],
        help="Version of the LTX 2.0 model",
    )

    parser.add_argument(
        "--combined_filename",
        default="ltx-2-19b-dev.safetensors",
        type=none_or_str,
        help="Filename for combined checkpoint with all LTX 2.0 models (VAE, DiT, etc.)",
    )
    parser.add_argument("--vae_prefix", default="vae.", type=str)
    parser.add_argument("--audio_vae_prefix", default="audio_vae.", type=str)
    parser.add_argument("--dit_prefix", default="model.diffusion_model.", type=str)
    parser.add_argument("--vocoder_prefix", default="vocoder.", type=str)

    parser.add_argument("--vae_filename", default=None, type=str, help="VAE filename; overrides combined ckpt if set")
    parser.add_argument(
        "--audio_vae_filename", default=None, type=str, help="Audio VAE filename; overrides combined ckpt if set"
    )
    parser.add_argument("--dit_filename", default=None, type=str, help="DiT filename; overrides combined ckpt if set")
    parser.add_argument(
        "--vocoder_filename", default=None, type=str, help="Vocoder filename; overrides combined ckpt if set"
    )
    parser.add_argument(
        "--text_encoder_model_id",
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        type=none_or_str,
        help="HF Hub id for the LTX 2.0 base text encoder model",
    )
    parser.add_argument(
        "--tokenizer_id",
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        type=none_or_str,
        help="HF Hub id for the LTX 2.0 text tokenizer",
    )
    parser.add_argument(
        "--latent_upsampler_filename",
        default="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        type=none_or_str,
        help="Latent upsampler filename",
    )

    parser.add_argument(
        "--timestep_conditioning", action="store_true", help="Whether to add timestep condition to the video VAE model"
    )
    parser.add_argument("--vae", action="store_true", help="Whether to convert the video VAE model")
    parser.add_argument("--audio_vae", action="store_true", help="Whether to convert the audio VAE model")
    parser.add_argument("--dit", action="store_true", help="Whether to convert the DiT model")
    parser.add_argument("--connectors", action="store_true", help="Whether to convert the connector model")
    parser.add_argument("--vocoder", action="store_true", help="Whether to convert the vocoder model")
    parser.add_argument("--text_encoder", action="store_true", help="Whether to conver the text encoder")
    parser.add_argument("--latent_upsampler", action="store_true", help="Whether to convert the latent upsampler")
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Whether to save the pipeline. This will attempt to convert all models (e.g. vae, dit, etc.)",
    )
    parser.add_argument(
        "--upsample_pipeline",
        action="store_true",
        help="Whether to save a latent upsampling pipeline",
    )

    parser.add_argument("--vae_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--audio_vae_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--dit_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vocoder_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--text_encoder_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")

    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

VARIANT_MAPPING = {
    "fp32": None,
    "fp16": "fp16",
    "bf16": "bf16",
}


def main(args):
    vae_dtype = DTYPE_MAPPING[args.vae_dtype]
    audio_vae_dtype = DTYPE_MAPPING[args.audio_vae_dtype]
    dit_dtype = DTYPE_MAPPING[args.dit_dtype]
    vocoder_dtype = DTYPE_MAPPING[args.vocoder_dtype]
    text_encoder_dtype = DTYPE_MAPPING[args.text_encoder_dtype]

    combined_ckpt = None
    load_combined_models = any(
        [
            args.vae,
            args.audio_vae,
            args.dit,
            args.vocoder,
            args.text_encoder,
            args.full_pipeline,
            args.upsample_pipeline,
        ]
    )
    if args.combined_filename is not None and load_combined_models:
        combined_ckpt = load_original_checkpoint(args, filename=args.combined_filename)

    if args.vae or args.full_pipeline or args.upsample_pipeline:
        if args.vae_filename is not None:
            original_vae_ckpt = load_hub_or_local_checkpoint(filename=args.vae_filename)
        elif combined_ckpt is not None:
            original_vae_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.vae_prefix)
        vae = convert_ltx2_video_vae(
            original_vae_ckpt, version=args.version, timestep_conditioning=args.timestep_conditioning
        )
        if not args.full_pipeline and not args.upsample_pipeline:
            vae.to(vae_dtype).save_pretrained(os.path.join(args.output_path, "vae"))

    if args.audio_vae or args.full_pipeline:
        if args.audio_vae_filename is not None:
            original_audio_vae_ckpt = load_hub_or_local_checkpoint(filename=args.audio_vae_filename)
        elif combined_ckpt is not None:
            original_audio_vae_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.audio_vae_prefix)
        audio_vae = convert_ltx2_audio_vae(original_audio_vae_ckpt, version=args.version)
        if not args.full_pipeline:
            audio_vae.to(audio_vae_dtype).save_pretrained(os.path.join(args.output_path, "audio_vae"))

    if args.dit or args.full_pipeline:
        if args.dit_filename is not None:
            original_dit_ckpt = load_hub_or_local_checkpoint(filename=args.dit_filename)
        elif combined_ckpt is not None:
            original_dit_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.dit_prefix)
        transformer = convert_ltx2_transformer(original_dit_ckpt, version=args.version)
        if not args.full_pipeline:
            transformer.to(dit_dtype).save_pretrained(os.path.join(args.output_path, "transformer"))

    if args.connectors or args.full_pipeline:
        if args.dit_filename is not None:
            original_connectors_ckpt = load_hub_or_local_checkpoint(filename=args.dit_filename)
        elif combined_ckpt is not None:
            original_connectors_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.dit_prefix)
        connectors = convert_ltx2_connectors(original_connectors_ckpt, version=args.version)
        if not args.full_pipeline:
            connectors.to(dit_dtype).save_pretrained(os.path.join(args.output_path, "connectors"))

    if args.vocoder or args.full_pipeline:
        if args.vocoder_filename is not None:
            original_vocoder_ckpt = load_hub_or_local_checkpoint(filename=args.vocoder_filename)
        elif combined_ckpt is not None:
            original_vocoder_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.vocoder_prefix)
        vocoder = convert_ltx2_vocoder(original_vocoder_ckpt, version=args.version)
        if not args.full_pipeline:
            vocoder.to(vocoder_dtype).save_pretrained(os.path.join(args.output_path, "vocoder"))

    if args.text_encoder or args.full_pipeline:
        # text_encoder = AutoModel.from_pretrained(args.text_encoder_model_id)
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(args.text_encoder_model_id)
        if not args.full_pipeline:
            text_encoder.to(text_encoder_dtype).save_pretrained(os.path.join(args.output_path, "text_encoder"))

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
        if not args.full_pipeline:
            tokenizer.save_pretrained(os.path.join(args.output_path, "tokenizer"))

    if args.latent_upsampler or args.full_pipeline or args.upsample_pipeline:
        original_latent_upsampler_ckpt = load_hub_or_local_checkpoint(
            repo_id=args.original_state_dict_repo_id, filename=args.latent_upsampler_filename
        )
        latent_upsampler_config = get_ltx2_spatial_latent_upsampler_config(args.version)
        latent_upsampler = convert_ltx2_spatial_latent_upsampler(
            original_latent_upsampler_ckpt,
            latent_upsampler_config,
            dtype=vae_dtype,
        )
        if not args.full_pipeline and not args.upsample_pipeline:
            latent_upsampler.save_pretrained(os.path.join(args.output_path, "latent_upsampler"))

    if args.full_pipeline:
        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_shift=0.95,
            max_shift=2.05,
            base_image_seq_len=1024,
            max_image_seq_len=4096,
            shift_terminal=0.1,
        )

        pipe = LTX2Pipeline(
            scheduler=scheduler,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=transformer,
            vocoder=vocoder,
        )

        pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.upsample_pipeline:
        pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler)

        # Put latent upsampling pipeline in its own subdirectory so it doesn't mess with the full pipeline
        pipe.save_pretrained(
            os.path.join(args.output_path, "upsample_pipeline"), safe_serialization=True, max_shard_size="5GB"
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
