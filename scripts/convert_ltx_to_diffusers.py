import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXConditionPipeline,
    LTXLatentUpsamplePipeline,
    LTXPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel


def remove_keys_(key: str, state_dict: Dict[str, Any]):
    state_dict.pop(key)


TOKENIZER_MAX_LENGTH = 128

TRANSFORMER_KEYS_RENAME_DICT = {
    "patchify_proj": "proj_in",
    "adaln_single": "time_embed",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "vae": remove_keys_,
}

VAE_KEYS_RENAME_DICT = {
    # decoder
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0",
    "up_blocks.2": "up_blocks.1.upsamplers.0",
    "up_blocks.3": "up_blocks.1",
    "up_blocks.4": "up_blocks.2.conv_in",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "up_blocks.7": "up_blocks.3.conv_in",
    "up_blocks.8": "up_blocks.3.upsamplers.0",
    "up_blocks.9": "up_blocks.3",
    # encoder
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.0.conv_out",
    "down_blocks.3": "down_blocks.1",
    "down_blocks.4": "down_blocks.1.downsamplers.0",
    "down_blocks.5": "down_blocks.1.conv_out",
    "down_blocks.6": "down_blocks.2",
    "down_blocks.7": "down_blocks.2.downsamplers.0",
    "down_blocks.8": "down_blocks.3",
    "down_blocks.9": "mid_block",
    # common
    "conv_shortcut": "conv_shortcut.conv",
    "res_blocks": "resnets",
    "norm3.norm": "norm3",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

VAE_091_RENAME_DICT = {
    # decoder
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
    # common
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
}

VAE_095_RENAME_DICT = {
    # decoder
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
    # encoder
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    # common
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
}

VAE_SPECIAL_KEYS_REMAP = {
    "per_channel_statistics.channel": remove_keys_,
    "per_channel_statistics.mean-of-means": remove_keys_,
    "per_channel_statistics.mean-of-stds": remove_keys_,
    "model.diffusion_model": remove_keys_,
}


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def update_state_dict_inplace(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def convert_transformer(ckpt_path: str, config, dtype: torch.dtype):
    PREFIX_KEY = "model.diffusion_model."

    original_state_dict = get_state_dict(load_file(ckpt_path))
    with init_empty_weights():
        transformer = LTXVideoTransformer3DModel(**config)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        if new_key.startswith(PREFIX_KEY):
            new_key = key[len(PREFIX_KEY) :]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    transformer.load_state_dict(original_state_dict, strict=True, assign=True)
    return transformer


def convert_vae(ckpt_path: str, config, dtype: torch.dtype):
    PREFIX_KEY = "vae."

    original_state_dict = get_state_dict(load_file(ckpt_path))
    with init_empty_weights():
        vae = AutoencoderKLLTXVideo(**config)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        if new_key.startswith(PREFIX_KEY):
            new_key = key[len(PREFIX_KEY) :]
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def convert_spatial_latent_upsampler(ckpt_path: str, config, dtype: torch.dtype):
    original_state_dict = get_state_dict(load_file(ckpt_path))

    with init_empty_weights():
        latent_upsampler = LTXLatentUpsamplerModel(**config)

    latent_upsampler.load_state_dict(original_state_dict, strict=True, assign=True)
    latent_upsampler.to(dtype)
    return latent_upsampler


def get_transformer_config(version: str) -> Dict[str, Any]:
    if version == "0.9.7":
        config = {
            "in_channels": 128,
            "out_channels": 128,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "cross_attention_dim": 4096,
            "num_layers": 48,
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 4096,
            "attention_bias": True,
            "attention_out_bias": True,
        }
    else:
        config = {
            "in_channels": 128,
            "out_channels": 128,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "cross_attention_dim": 2048,
            "num_layers": 28,
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 4096,
            "attention_bias": True,
            "attention_out_bias": True,
        }
    return config


def get_vae_config(version: str) -> Dict[str, Any]:
    if version in ["0.9.0"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 512),
            "down_block_types": (
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
            ),
            "decoder_block_out_channels": (128, 256, 512, 512),
            "layers_per_block": (4, 3, 3, 3, 4),
            "decoder_layers_per_block": (4, 3, 3, 3, 4),
            "spatio_temporal_scaling": (True, True, True, False),
            "decoder_spatio_temporal_scaling": (True, True, True, False),
            "decoder_inject_noise": (False, False, False, False, False),
            "downsample_type": ("conv", "conv", "conv", "conv"),
            "upsample_residual": (False, False, False, False),
            "upsample_factor": (1, 1, 1, 1),
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
            "timestep_conditioning": False,
        }
    elif version in ["0.9.1"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 512),
            "down_block_types": (
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
                "LTXVideoDownBlock3D",
            ),
            "decoder_block_out_channels": (256, 512, 1024),
            "layers_per_block": (4, 3, 3, 3, 4),
            "decoder_layers_per_block": (5, 6, 7, 8),
            "spatio_temporal_scaling": (True, True, True, False),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (True, True, True, False),
            "downsample_type": ("conv", "conv", "conv", "conv"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": True,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
        }
        VAE_KEYS_RENAME_DICT.update(VAE_091_RENAME_DICT)
    elif version in ["0.9.5"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 1024, 2048),
            "down_block_types": (
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
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
            "timestep_conditioning": True,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        }
        VAE_KEYS_RENAME_DICT.update(VAE_095_RENAME_DICT)
    elif version in ["0.9.7"]:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (128, 256, 512, 1024, 2048),
            "down_block_types": (
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
                "LTXVideo095DownBlock3D",
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
            "timestep_conditioning": True,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0,
            "encoder_causal": True,
            "decoder_causal": False,
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        }
        VAE_KEYS_RENAME_DICT.update(VAE_095_RENAME_DICT)
    return config


def get_spatial_latent_upsampler_config(version: str) -> Dict[str, Any]:
    if version == "0.9.7":
        config = {
            "in_channels": 128,
            "mid_channels": 512,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
        }
    elif version == "0.9.8":
        config = {
            "in_channels": 128,
            "mid_channels": 512,
            "num_blocks_per_stage": 4,
            "dims": 3,
            "spatial_upsample": True,
            "temporal_upsample": False,
        }
    else:
        raise ValueError(f"Unsupported version: {version}")
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original vae checkpoint")
    parser.add_argument(
        "--spatial_latent_upsampler_path",
        type=str,
        default=None,
        help="Path to original spatial latent upsampler checkpoint",
    )
    parser.add_argument(
        "--text_encoder_cache_dir", type=str, default=None, help="Path to text encoder cache directory"
    )
    parser.add_argument(
        "--typecast_text_encoder",
        action="store_true",
        default=False,
        help="Whether or not to apply fp16/bf16 precision to text_encoder",
    )
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="fp32", help="Torch dtype to save the model in.")
    parser.add_argument(
        "--version",
        type=str,
        default="0.9.0",
        choices=["0.9.0", "0.9.1", "0.9.5", "0.9.7", "0.9.8"],
        help="Version of the LTX model",
    )
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


if __name__ == "__main__":
    args = get_args()

    transformer = None
    dtype = DTYPE_MAPPING[args.dtype]
    variant = VARIANT_MAPPING[args.dtype]
    output_path = Path(args.output_path)

    if args.transformer_ckpt_path is not None:
        config = get_transformer_config(args.version)
        transformer: LTXVideoTransformer3DModel = convert_transformer(args.transformer_ckpt_path, config, dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(
                output_path / "transformer", safe_serialization=True, max_shard_size="5GB", variant=variant
            )

    if args.vae_ckpt_path is not None:
        config = get_vae_config(args.version)
        vae: AutoencoderKLLTXVideo = convert_vae(args.vae_ckpt_path, config, dtype)
        if not args.save_pipeline:
            vae.save_pretrained(output_path / "vae", safe_serialization=True, max_shard_size="5GB", variant=variant)

    if args.spatial_latent_upsampler_path is not None:
        config = get_spatial_latent_upsampler_config(args.version)
        latent_upsampler: LTXLatentUpsamplerModel = convert_spatial_latent_upsampler(
            args.spatial_latent_upsampler_path, config, dtype
        )
        if not args.save_pipeline:
            latent_upsampler.save_pretrained(
                output_path / "latent_upsampler", safe_serialization=True, max_shard_size="5GB", variant=variant
            )

    if args.save_pipeline:
        text_encoder_id = "google/t5-v1_1-xxl"
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_id, model_max_length=TOKENIZER_MAX_LENGTH)
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, cache_dir=args.text_encoder_cache_dir)

        if args.typecast_text_encoder:
            text_encoder = text_encoder.to(dtype=dtype)

        # Apparently, the conversion does not work anymore without this :shrug:
        for param in text_encoder.parameters():
            param.data = param.data.contiguous()

        if args.version in ["0.9.5", "0.9.7"]:
            scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False)
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=True,
                base_shift=0.95,
                max_shift=2.05,
                base_image_seq_len=1024,
                max_image_seq_len=4096,
                shift_terminal=0.1,
            )

        if args.version in ["0.9.0", "0.9.1", "0.9.5"]:
            pipe = LTXPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
            )
            pipe.save_pretrained(
                output_path.as_posix(), safe_serialization=True, variant=variant, max_shard_size="5GB"
            )
        elif args.version in ["0.9.7"]:
            pipe = LTXConditionPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
            )
            pipe_upsample = LTXLatentUpsamplePipeline(
                vae=vae,
                latent_upsampler=latent_upsampler,
            )
            pipe.save_pretrained(
                (output_path / "ltx_pipeline").as_posix(),
                safe_serialization=True,
                variant=variant,
                max_shard_size="5GB",
            )
            pipe_upsample.save_pretrained(
                (output_path / "ltx_upsample_pipeline").as_posix(),
                safe_serialization=True,
                variant=variant,
                max_shard_size="5GB",
            )
        else:
            raise ValueError(f"Unsupported version: {args.version}")
