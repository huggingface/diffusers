import argparse
import os

import torch

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.consistency import (
    CD_SCHEDULER_CONFIG,
    CT_IMAGENET_64_SCHEDULER_CONFIG,
    CT_LSUN_256_SCHEDULER_CONFIG,
    IMAGENET_64_UNET_CONFIG,
    LSUN_256_UNET_CONFIG,
    TEST_UNET_CONFIG,
)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def con_pt_to_diffuser(checkpoint_path, config):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return get_conversion("UNet2DModel", {**config, "original_format": "consistency"}).to_diffusers(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_path", default=None, type=str, required=True, help="Path to the unet.pt to convert.")
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output the converted UNet model."
    )
    parser.add_argument("--class_cond", default=True, type=str, help="Whether the model is class-conditional.")

    args = parser.parse_args()
    args.class_cond = str2bool(args.class_cond)

    ckpt_name = os.path.basename(args.unet_path)
    print(f"Checkpoint: {ckpt_name}")

    # Get U-Net config
    if "imagenet64" in ckpt_name:
        unet_config = IMAGENET_64_UNET_CONFIG
    elif "256" in ckpt_name and (("bedroom" in ckpt_name) or ("cat" in ckpt_name)):
        unet_config = LSUN_256_UNET_CONFIG
    elif "test" in ckpt_name:
        unet_config = TEST_UNET_CONFIG
    else:
        raise ValueError(f"Checkpoint type {ckpt_name} is not currently supported.")

    if not args.class_cond:
        unet_config["num_class_embeds"] = None

    converted_unet_ckpt = con_pt_to_diffuser(args.unet_path, unet_config)

    image_unet = UNet2DModel(**unet_config)
    image_unet.load_state_dict(converted_unet_ckpt)

    # Get scheduler config
    if "cd" in ckpt_name or "test" in ckpt_name:
        scheduler_config = CD_SCHEDULER_CONFIG
    elif "ct" in ckpt_name and "imagenet64" in ckpt_name:
        scheduler_config = CT_IMAGENET_64_SCHEDULER_CONFIG
    elif "ct" in ckpt_name and "256" in ckpt_name and (("bedroom" in ckpt_name) or ("cat" in ckpt_name)):
        scheduler_config = CT_LSUN_256_SCHEDULER_CONFIG
    else:
        raise ValueError(f"Checkpoint type {ckpt_name} is not currently supported.")

    cm_scheduler = CMStochasticIterativeScheduler(**scheduler_config)

    consistency_model = ConsistencyModelPipeline(unet=image_unet, scheduler=cm_scheduler)
    consistency_model.save_pretrained(args.dump_path)
