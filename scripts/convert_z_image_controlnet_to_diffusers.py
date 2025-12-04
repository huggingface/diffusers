import argparse
from contextlib import nullcontext

import torch
import safetensors.torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel
from diffusers.utils.import_utils import is_accelerate_available


"""
python scripts/convert_z_image_controlnet_to_diffusers.py  \
--original_controlnet_repo_id "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union" \
--filename "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
--output_path "z-image-controlnet-hf/"
"""


CTX = init_empty_weights if is_accelerate_available else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_controlnet_repo_id", default=None, type=str)
parser.add_argument("--filename", default="Z-Image-Turbo-Fun-Controlnet-Union.safetensors", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", type=str)

args = parser.parse_args()


def load_original_checkpoint(args):
    if args.original_controlnet_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_controlnet_repo_id, filename=args.filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError(" please provide either `original_controlnet_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


def convert_z_image_controlnet_checkpoint_to_diffusers(original_state_dict):
    converted_state_dict = {}

    converted_state_dict.update(original_state_dict)

    return converted_state_dict


def main(args):
    original_ckpt = load_original_checkpoint(args)

    control_in_dim = 16
    control_layers_places = [0, 5, 10, 15, 20, 25]

    converted_controlnet_state_dict = convert_z_image_controlnet_checkpoint_to_diffusers(original_ckpt)

    controlnet = ZImageControlNetModel(
        control_layers_places=control_layers_places,
        control_in_dim=control_in_dim,
    ).to(torch.bfloat16)
    controlnet.load_state_dict(converted_controlnet_state_dict)
    print("Saving Z-Image ControlNet in Diffusers format")
    controlnet.save_pretrained(args.output_path)


if __name__ == "__main__":
    main(args)
