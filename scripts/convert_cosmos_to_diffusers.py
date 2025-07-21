import argparse
import pathlib
from typing import Any, Dict

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKLCosmos,
    AutoencoderKLWan,
    Cosmos2TextToImagePipeline,
    Cosmos2VideoToWorldPipeline,
    CosmosTextToWorldPipeline,
    CosmosTransformer3DModel,
    CosmosVideoToWorldPipeline,
    EDMEulerScheduler,
    FlowMatchEulerDiscreteScheduler,
)


def remove_keys_(key: str, state_dict: Dict[str, Any]):
    state_dict.pop(key)


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def rename_transformer_blocks_(key: str, state_dict: Dict[str, Any]):
    block_index = int(key.split(".")[1].removeprefix("block"))
    new_key = key

    old_prefix = f"blocks.block{block_index}"
    new_prefix = f"transformer_blocks.{block_index}"
    new_key = new_prefix + new_key.removeprefix(old_prefix)

    state_dict[new_key] = state_dict.pop(key)


TRANSFORMER_KEYS_RENAME_DICT_COSMOS_1_0 = {
    "t_embedder.1": "time_embed.t_embedder",
    "affline_norm": "time_embed.norm",
    ".blocks.0.block.attn": ".attn1",
    ".blocks.1.block.attn": ".attn2",
    ".blocks.2.block": ".ff",
    ".blocks.0.adaLN_modulation.1": ".norm1.linear_1",
    ".blocks.0.adaLN_modulation.2": ".norm1.linear_2",
    ".blocks.1.adaLN_modulation.1": ".norm2.linear_1",
    ".blocks.1.adaLN_modulation.2": ".norm2.linear_2",
    ".blocks.2.adaLN_modulation.1": ".norm3.linear_1",
    ".blocks.2.adaLN_modulation.2": ".norm3.linear_2",
    "to_q.0": "to_q",
    "to_q.1": "norm_q",
    "to_k.0": "to_k",
    "to_k.1": "norm_k",
    "to_v.0": "to_v",
    "layer1": "net.0.proj",
    "layer2": "net.2",
    "proj.1": "proj",
    "x_embedder": "patch_embed",
    "extra_pos_embedder": "learnable_pos_embed",
    "final_layer.adaLN_modulation.1": "norm_out.linear_1",
    "final_layer.adaLN_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}

TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_1_0 = {
    "blocks.block": rename_transformer_blocks_,
    "logvar.0.freqs": remove_keys_,
    "logvar.0.phases": remove_keys_,
    "logvar.1.weight": remove_keys_,
    "pos_embedder.seq": remove_keys_,
}

TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0 = {
    "t_embedder.1": "time_embed.t_embedder",
    "t_embedding_norm": "time_embed.norm",
    "blocks": "transformer_blocks",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
    "self_attn": "attn1",
    "cross_attn": "attn2",
    "q_proj": "to_q",
    "k_proj": "to_k",
    "v_proj": "to_v",
    "output_proj": "to_out.0",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "x_embedder.proj.1": "patch_embed.proj",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}

TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_2_0 = {
    "accum_video_sample_counter": remove_keys_,
    "accum_image_sample_counter": remove_keys_,
    "accum_iteration": remove_keys_,
    "accum_train_in_hours": remove_keys_,
    "pos_embedder.seq": remove_keys_,
    "pos_embedder.dim_spatial_range": remove_keys_,
    "pos_embedder.dim_temporal_range": remove_keys_,
    "_extra_state": remove_keys_,
}


TRANSFORMER_CONFIGS = {
    "Cosmos-1.0-Diffusion-7B-Text2World": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (2.0, 1.0, 1.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": "learnable",
    },
    "Cosmos-1.0-Diffusion-7B-Video2World": {
        "in_channels": 16 + 1,
        "out_channels": 16,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (2.0, 1.0, 1.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": "learnable",
    },
    "Cosmos-1.0-Diffusion-14B-Text2World": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "num_layers": 36,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (2.0, 2.0, 2.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": "learnable",
    },
    "Cosmos-1.0-Diffusion-14B-Video2World": {
        "in_channels": 16 + 1,
        "out_channels": 16,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "num_layers": 36,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (2.0, 2.0, 2.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": "learnable",
    },
    "Cosmos-2.0-Diffusion-2B-Text2Image": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (1.0, 4.0, 4.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    },
    "Cosmos-2.0-Diffusion-14B-Text2Image": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "num_layers": 36,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (1.0, 4.0, 4.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    },
    "Cosmos-2.0-Diffusion-2B-Video2World": {
        "in_channels": 16 + 1,
        "out_channels": 16,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (1.0, 3.0, 3.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    },
    "Cosmos-2.0-Diffusion-14B-Video2World": {
        "in_channels": 16 + 1,
        "out_channels": 16,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "num_layers": 36,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (20 / 24, 2.0, 2.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    },
}

VAE_KEYS_RENAME_DICT = {
    "down.0": "down_blocks.0",
    "down.1": "down_blocks.1",
    "down.2": "down_blocks.2",
    "up.0": "up_blocks.2",
    "up.1": "up_blocks.1",
    "up.2": "up_blocks.0",
    ".block.": ".resnets.",
    "downsample": "downsamplers.0",
    "upsample": "upsamplers.0",
    "mid.block_1": "mid_block.resnets.0",
    "mid.attn_1.0": "mid_block.attentions.0",
    "mid.attn_1.1": "mid_block.temp_attentions.0",
    "mid.block_2": "mid_block.resnets.1",
    ".q.conv3d": ".to_q",
    ".k.conv3d": ".to_k",
    ".v.conv3d": ".to_v",
    ".proj_out.conv3d": ".to_out.0",
    ".0.conv3d": ".conv_s",
    ".1.conv3d": ".conv_t",
    "conv1.conv3d": "conv1",
    "conv2.conv3d": "conv2",
    "conv3.conv3d": "conv3",
    "nin_shortcut.conv3d": "conv_shortcut",
    "quant_conv.conv3d": "quant_conv",
    "post_quant_conv.conv3d": "post_quant_conv",
}

VAE_SPECIAL_KEYS_REMAP = {
    "wavelets": remove_keys_,
    "_arange": remove_keys_,
    "patch_size_buffer": remove_keys_,
}

VAE_CONFIGS = {
    "CV8x8x8-0.1": {
        "name": "nvidia/Cosmos-0.1-Tokenizer-CV8x8x8",
        "diffusers_config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 16,
            "encoder_block_out_channels": (128, 256, 512, 512),
            "decode_block_out_channels": (256, 512, 512, 512),
            "attention_resolutions": (32,),
            "resolution": 1024,
            "num_layers": 2,
            "patch_size": 4,
            "patch_type": "haar",
            "scaling_factor": 1.0,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 8,
            "latents_mean": None,
            "latents_std": None,
        },
    },
    "CV8x8x8-1.0": {
        "name": "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
        "diffusers_config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 16,
            "encoder_block_out_channels": (128, 256, 512, 512),
            "decode_block_out_channels": (256, 512, 512, 512),
            "attention_resolutions": (32,),
            "resolution": 1024,
            "num_layers": 2,
            "patch_size": 4,
            "patch_type": "haar",
            "scaling_factor": 1.0,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 8,
            "latents_mean": None,
            "latents_std": None,
        },
    },
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


def convert_transformer(transformer_type: str, ckpt_path: str, weights_only: bool = True):
    PREFIX_KEY = "net."
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=weights_only))

    if "Cosmos-1.0" in transformer_type:
        TRANSFORMER_KEYS_RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT_COSMOS_1_0
        TRANSFORMER_SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_1_0
    elif "Cosmos-2.0" in transformer_type:
        TRANSFORMER_KEYS_RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0
        TRANSFORMER_SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP_COSMOS_2_0
    else:
        assert False

    with init_empty_weights():
        config = TRANSFORMER_CONFIGS[transformer_type]
        transformer = CosmosTransformer3DModel(**config)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        if new_key.startswith(PREFIX_KEY):
            new_key = new_key.removeprefix(PREFIX_KEY)
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    transformer.load_state_dict(original_state_dict, strict=True, assign=True)
    return transformer


def convert_vae(vae_type: str):
    model_name = VAE_CONFIGS[vae_type]["name"]
    snapshot_directory = snapshot_download(model_name, repo_type="model")
    directory = pathlib.Path(snapshot_directory)

    autoencoder_file = directory / "autoencoder.jit"
    mean_std_file = directory / "mean_std.pt"

    original_state_dict = torch.jit.load(autoencoder_file.as_posix()).state_dict()
    if mean_std_file.exists():
        mean_std = torch.load(mean_std_file, map_location="cpu", weights_only=True)
    else:
        mean_std = (None, None)

    config = VAE_CONFIGS[vae_type]["diffusers_config"]
    config.update(
        {
            "latents_mean": mean_std[0].detach().cpu().numpy().tolist(),
            "latents_std": mean_std[1].detach().cpu().numpy().tolist(),
        }
    )
    vae = AutoencoderKLCosmos(**config)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def save_pipeline_cosmos_1_0(args, transformer, vae):
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.bfloat16)
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_path)
    # The original code initializes EDM config with sigma_min=0.0002, but does not make use of it anywhere directly.
    # So, the sigma_min values that is used is the default value of 0.002.
    scheduler = EDMEulerScheduler(
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        sigma_schedule="karras",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        rho=7.0,
        final_sigmas_type="sigma_min",
    )

    pipe_cls = CosmosTextToWorldPipeline if "Text2World" in args.transformer_type else CosmosVideoToWorldPipeline
    pipe = pipe_cls(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=lambda *args, **kwargs: None,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")


def save_pipeline_cosmos_2_0(args, transformer, vae):
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.bfloat16)
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_path)

    scheduler = FlowMatchEulerDiscreteScheduler(use_karras_sigmas=True)

    pipe_cls = Cosmos2TextToImagePipeline if "Text2Image" in args.transformer_type else Cosmos2VideoToWorldPipeline
    pipe = pipe_cls(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=lambda *args, **kwargs: None,
    )
    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_type", type=str, default=None, choices=list(TRANSFORMER_CONFIGS.keys()))
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument(
        "--vae_type", type=str, default=None, choices=["none", *list(VAE_CONFIGS.keys())], help="Type of VAE"
    )
    parser.add_argument("--text_encoder_path", type=str, default="google-t5/t5-11b")
    parser.add_argument("--tokenizer_path", type=str, default="google-t5/t5-11b")
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the transformer in.")
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


if __name__ == "__main__":
    args = get_args()

    transformer = None
    dtype = DTYPE_MAPPING[args.dtype]

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None
        assert args.vae_type is not None
        assert args.text_encoder_path is not None
        assert args.tokenizer_path is not None

    if args.transformer_ckpt_path is not None:
        weights_only = "Cosmos-1.0" in args.transformer_type
        transformer = convert_transformer(args.transformer_type, args.transformer_ckpt_path, weights_only)
        transformer = transformer.to(dtype=dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.vae_type is not None:
        if "Cosmos-1.0" in args.transformer_type:
            vae = convert_vae(args.vae_type)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32
            )
        if not args.save_pipeline:
            vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.save_pipeline:
        if "Cosmos-1.0" in args.transformer_type:
            save_pipeline_cosmos_1_0(args, transformer, vae)
        elif "Cosmos-2.0" in args.transformer_type:
            save_pipeline_cosmos_2_0(args, transformer, vae)
        else:
            assert False
