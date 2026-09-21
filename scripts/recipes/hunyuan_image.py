import argparse
import logging

import torch
from safetensors import safe_open

from diffusers import AutoencoderKLHunyuanImage, AutoencoderKLHunyuanImageRefiner, HunyuanImageTransformer2DModel
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


"""
Usage examples
==============

python scripts/recipes/hunyuan_image.py \
    --model_type hunyuanimage2.1 \
    --transformer_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/dit/hunyuanimage2.1.safetensors" \
    --vae_checkpoint_path "HunyuanImage-2.1/ckpts/vae/vae_2_1/pytorch_model.ckpt" \
    --output_path "/raid/yiyi/test-hy21-diffusers" \
    --dtype fp32

python scripts/recipes/hunyuan_image.py \
    --model_type hunyuanimage2.1-distilled \
    --transformer_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/dit/hunyuanimage2.1-distilled.safetensors" \
    --vae_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/vae/vae_2_1/pytorch_model.ckpt" \
    --output_path "/raid/yiyi/test-hy21-distilled-diffusers" \
    --dtype fp32


python scripts/recipes/hunyuan_image.py \
  --model_type hunyuanimage-refiner \
  --transformer_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/dit/hunyuanimage-refiner.safetensors" \
  --vae_checkpoint_path "/raid/yiyi/HunyuanImage-2.1/ckpts/vae/vae_refiner/pytorch_model.pt" \
  --output_path "/raid/yiyi/test-hy2-refiner-diffusers" \
  --dtype fp32
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type", type=str, default=None
)  # hunyuanimage2.1, hunyuanimage2.1-distilled, hunyuanimage-refiner
parser.add_argument("--transformer_checkpoint_path", default=None, type=str)  # ckpts/dit/hunyuanimage2.1.safetensors
parser.add_argument("--vae_checkpoint_path", default=None, type=str)  # ckpts/vae/vae_2_1/pytorch_model.ckpt
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str, default="fp32")

args = parser.parse_args()
dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32


# copied from https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/hyimage/models/hunyuan/modules/hunyuanimage_dit.py#L21


def load_original_vae_checkpoint(args):
    # "ckpts/vae/vae_2_1/pytorch_model.ckpt"
    state_dict = torch.load(args.vae_checkpoint_path)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    vae_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_state_dict[k.replace("vae.", "")] = v

    return vae_state_dict


def load_original_refiner_vae_checkpoint(args):
    # "ckpts/vae/vae_refiner/pytorch_model.pt"
    state_dict = torch.load(args.vae_checkpoint_path)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    vae_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_state_dict[k.replace("vae.", "")] = v
    return vae_state_dict


def load_original_transformer_checkpoint(args):
    # ckpts/dit/hunyuanimage-refiner.safetensors"
    # ckpts/dit/hunyuanimage2.1.safetensors"
    state_dict = {}
    with safe_open(args.transformer_checkpoint_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def convert_hunyuan_image_transformer_checkpoint_to_diffusers(original_state_dict, config):
    return convert_component_checkpoint(original_state_dict, config, "HunyuanImageTransformer2DModel"), {}


def convert_hunyuan_image_vae_checkpoint_to_diffusers(
    original_state_dict, block_out_channels=[128, 256, 512, 512, 1024, 1024], layers_per_block=2
):
    config = {"block_out_channels": block_out_channels, "layers_per_block": layers_per_block}
    return convert_component_checkpoint(original_state_dict, config, "AutoencoderKLHunyuanImage"), {}


def convert_hunyuan_image_refiner_vae_checkpoint_to_diffusers(
    original_state_dict, block_out_channels=[128, 256, 512, 1024, 1024], layers_per_block=2
):
    config = {"block_out_channels": block_out_channels, "layers_per_block": layers_per_block}
    return convert_component_checkpoint(original_state_dict, config, "AutoencoderKLHunyuanImageRefiner"), {}


def main(args):
    if args.model_type == "hunyuanimage2.1":
        original_transformer_state_dict = load_original_transformer_checkpoint(args)
        original_vae_state_dict = load_original_vae_checkpoint(args)

        transformer_config = {
            "in_channels": 64,
            "out_channels": 64,
            "num_attention_heads": 28,
            "attention_head_dim": 128,
            "num_layers": 20,
            "num_single_layers": 40,
            "num_refiner_layers": 2,
            "patch_size": (1, 1),
            "qk_norm": "rms_norm",
            "guidance_embeds": False,
            "text_embed_dim": 3584,
            "text_embed_2_dim": 1472,
            "rope_theta": 256.0,
            "rope_axes_dim": (64, 64),
        }

        converted_transformer_state_dict, original_transformer_state_dict = (
            convert_hunyuan_image_transformer_checkpoint_to_diffusers(
                original_transformer_state_dict, config=transformer_config
            )
        )

        if original_transformer_state_dict:
            logger.warning(
                f"Unused {len(original_transformer_state_dict)} original keys for transformer: {list(original_transformer_state_dict.keys())}"
            )

        transformer = HunyuanImageTransformer2DModel(**transformer_config)
        missing_keys, unexpected_key = transformer.load_state_dict(converted_transformer_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for transformer: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for transformer: {unexpected_key}")

        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

        vae_config_diffusers = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 64,
            "block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "layers_per_block": 2,
            "spatial_compression_ratio": 32,
            "sample_size": 384,
            "scaling_factor": 0.75289,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
        }
        converted_vae_state_dict, original_vae_state_dict = convert_hunyuan_image_vae_checkpoint_to_diffusers(
            original_vae_state_dict, block_out_channels=[128, 256, 512, 512, 1024, 1024], layers_per_block=2
        )
        if original_vae_state_dict:
            logger.warning(
                f"Unused {len(original_vae_state_dict)} original keys for vae: {list(original_vae_state_dict.keys())}"
            )

        vae = AutoencoderKLHunyuanImage(**vae_config_diffusers)
        missing_keys, unexpected_key = vae.load_state_dict(converted_vae_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for vae: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for vae: {unexpected_key}")

        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")

    elif args.model_type == "hunyuanimage2.1-distilled":
        original_transformer_state_dict = load_original_transformer_checkpoint(args)
        original_vae_state_dict = load_original_vae_checkpoint(args)

        transformer_config = {
            "in_channels": 64,
            "out_channels": 64,
            "num_attention_heads": 28,
            "attention_head_dim": 128,
            "num_layers": 20,
            "num_single_layers": 40,
            "num_refiner_layers": 2,
            "patch_size": (1, 1),
            "qk_norm": "rms_norm",
            "guidance_embeds": True,
            "text_embed_dim": 3584,
            "text_embed_2_dim": 1472,
            "rope_theta": 256.0,
            "rope_axes_dim": (64, 64),
            "use_meanflow": True,
        }

        converted_transformer_state_dict, original_transformer_state_dict = (
            convert_hunyuan_image_transformer_checkpoint_to_diffusers(
                original_transformer_state_dict, config=transformer_config
            )
        )

        if original_transformer_state_dict:
            logger.warning(
                f"Unused {len(original_transformer_state_dict)} original keys for transformer: {list(original_transformer_state_dict.keys())}"
            )

        transformer = HunyuanImageTransformer2DModel(**transformer_config)
        missing_keys, unexpected_key = transformer.load_state_dict(converted_transformer_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for transformer: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for transformer: {unexpected_key}")

        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

        vae_config_diffusers = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 64,
            "block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "layers_per_block": 2,
            "spatial_compression_ratio": 32,
            "sample_size": 384,
            "scaling_factor": 0.75289,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
        }
        converted_vae_state_dict, original_vae_state_dict = convert_hunyuan_image_vae_checkpoint_to_diffusers(
            original_vae_state_dict, block_out_channels=[128, 256, 512, 512, 1024, 1024], layers_per_block=2
        )
        if original_vae_state_dict:
            logger.warning(
                f"Unused {len(original_vae_state_dict)} original keys for vae: {list(original_vae_state_dict.keys())}"
            )

        vae = AutoencoderKLHunyuanImage(**vae_config_diffusers)
        missing_keys, unexpected_key = vae.load_state_dict(converted_vae_state_dict, strict=True)

        if missing_keys:
            logger.warning(f"Missing keys for vae: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for vae: {unexpected_key}")

        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")

    elif args.model_type == "hunyuanimage-refiner":
        original_transformer_state_dict = load_original_transformer_checkpoint(args)
        original_vae_state_dict = load_original_refiner_vae_checkpoint(args)

        transformer_config = {
            "in_channels": 128,
            "out_channels": 64,
            "num_layers": 20,
            "num_single_layers": 40,
            "rope_axes_dim": [16, 56, 56],
            "num_attention_heads": 26,
            "attention_head_dim": 128,
            "mlp_ratio": 4,
            "patch_size": (1, 1, 1),
            "text_embed_dim": 3584,
            "guidance_embeds": True,
        }
        converted_transformer_state_dict, original_transformer_state_dict = (
            convert_hunyuan_image_transformer_checkpoint_to_diffusers(
                original_transformer_state_dict, config=transformer_config
            )
        )
        if original_transformer_state_dict:
            logger.warning(
                f"Unused {len(original_transformer_state_dict)} original keys for transformer: {list(original_transformer_state_dict.keys())}"
            )

        transformer = HunyuanImageTransformer2DModel(**transformer_config)
        missing_keys, unexpected_key = transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        if missing_keys:
            logger.warning(f"Missing keys for transformer: {missing_keys}")
        if unexpected_key:
            logger.warning(f"Unexpected keys for transformer: {unexpected_key}")

        transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

        vae = AutoencoderKLHunyuanImageRefiner()

        converted_vae_state_dict, original_vae_state_dict = convert_hunyuan_image_refiner_vae_checkpoint_to_diffusers(
            original_vae_state_dict
        )
        if original_vae_state_dict:
            logger.warning(
                f"Unused {len(original_vae_state_dict)} original keys for vae: {list(original_vae_state_dict.keys())}"
            )

        missing_keys, unexpected_key = vae.load_state_dict(converted_vae_state_dict, strict=True)
        logger.warning(f"Missing keys for vae: {missing_keys}")
        logger.warning(f"Unexpected keys for vae: {unexpected_key}")

        vae.to(dtype).save_pretrained(f"{args.output_path}/vae")


if __name__ == "__main__":
    main(args)
