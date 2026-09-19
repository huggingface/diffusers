import argparse
import json

import torch

from diffusers import AutoencoderKL, DDPMPipeline, DDPMScheduler, UNet2DModel, VQModel
from diffusers.loaders.conversion import get_conversion


def convert_ddpm_checkpoint(checkpoint, config):
    return get_conversion("UNet2DModel", {**config, "original_format": "ddpm"}).to_diffusers(checkpoint)


def convert_vq_autoenc_checkpoint(checkpoint, config):
    return get_conversion(config["_class_name"], config).to_diffusers(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_path)

    with open(args.config_file) as f:
        config = json.loads(f.read())

    # unet case
    key_prefix_set = {key.split(".")[0] for key in checkpoint.keys()}
    if "encoder" in key_prefix_set and "decoder" in key_prefix_set:
        converted_checkpoint = convert_vq_autoenc_checkpoint(checkpoint, config)
    else:
        converted_checkpoint = convert_ddpm_checkpoint(checkpoint, config)

    if "ddpm" in config:
        del config["ddpm"]

    if config["_class_name"] == "VQModel":
        model = VQModel(**config)
        model.load_state_dict(converted_checkpoint)
        model.save_pretrained(args.dump_path)
    elif config["_class_name"] == "AutoencoderKL":
        model = AutoencoderKL(**config)
        model.load_state_dict(converted_checkpoint)
        model.save_pretrained(args.dump_path)
    else:
        model = UNet2DModel(**config)
        model.load_state_dict(converted_checkpoint)

        scheduler = DDPMScheduler.from_config("/".join(args.checkpoint_path.split("/")[:-1]))

        pipe = DDPMPipeline(unet=model, scheduler=scheduler)
        pipe.save_pretrained(args.dump_path)
