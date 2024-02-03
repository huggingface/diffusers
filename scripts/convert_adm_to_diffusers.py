import argparse
import os

import torch
from convert_consistency_to_diffusers import con_pt_to_diffuser

from diffusers import (
    UNet2DModel,
)


SMALL_256_UNET_CONFIG = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 6,
    "layers_per_block": 1,
    "num_class_embeds": None,
    "block_out_channels": [128, 128, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "ResnetDownsampleBlock2D",
    ],
    "up_block_types": [
        "ResnetUpsampleBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
}


LARGE_256_UNET_CONFIG = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 6,
    "layers_per_block": 2,
    "num_class_embeds": None,
    "block_out_channels": [256, 256, 256 * 2, 256 * 2, 256 * 4, 256 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_path", default=None, type=str, required=True, help="Path to the unet.pt to convert.")
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output the converted UNet model."
    )

    args = parser.parse_args()

    ckpt_name = os.path.basename(args.unet_path)
    print(f"Checkpoint: {ckpt_name}")

    # Get U-Net config
    if "ffhq" in ckpt_name:
        unet_config = SMALL_256_UNET_CONFIG
    else:
        unet_config = LARGE_256_UNET_CONFIG

    unet_config["num_class_embeds"] = None

    converted_unet_ckpt = con_pt_to_diffuser(args.unet_path, unet_config)

    image_unet = UNet2DModel(**unet_config)
    image_unet.load_state_dict(converted_unet_ckpt)

    torch.save(converted_unet_ckpt, args.dump_path)
