import argparse
from contextlib import nullcontext

import torch
import safetensors.torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from diffusers.utils.import_utils import is_accelerate_available
from diffusers.models import ZImageTransformer2DModel
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel

"""
python scripts/convert_z_image_controlnet_to_diffusers.py  \
--original_z_image_repo_id "Tongyi-MAI/Z-Image-Turbo" \
--original_controlnet_repo_id "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union" \
--filename "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
--output_path "z-image-controlnet-hf/"
"""


CTX = init_empty_weights if is_accelerate_available else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_z_image_repo_id", default="Tongyi-MAI/Z-Image-Turbo", type=str)
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

def load_z_image(args):
    model = ZImageTransformer2DModel.from_pretrained(args.original_z_image_repo_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    return model.state_dict(), model.config

def convert_z_image_controlnet_checkpoint_to_diffusers(z_image, original_state_dict):
    converted_state_dict = {}

    converted_state_dict.update(original_state_dict)

    to_copy = {"all_x_embedder.", "noise_refiner.", "context_refiner.", "t_embedder.", "cap_embedder.", "x_pad_token", "cap_pad_token"}

    for key in z_image.keys():
        for copy_key in to_copy:
            if key.startswith(copy_key):
                converted_state_dict[key] = z_image[key]

    return converted_state_dict


def main(args):
    original_ckpt = load_original_checkpoint(args)
    z_image, config = load_z_image(args)

    control_in_dim = 16
    control_layers_places = [0, 5, 10, 15, 20, 25]

    converted_controlnet_state_dict = convert_z_image_controlnet_checkpoint_to_diffusers(z_image, original_ckpt)

    for key, tensor in converted_controlnet_state_dict.items():
        print(f"{key} - {tensor.dtype}")

    controlnet = ZImageControlNetModel(
        all_patch_size=config["all_patch_size"],
        all_f_patch_size=config["all_f_patch_size"],
        in_channels=config["in_channels"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_refiner_layers=config["n_refiner_layers"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        norm_eps=config["norm_eps"],
        qk_norm=config["qk_norm"],
        cap_feat_dim=config["cap_feat_dim"],
        rope_theta=config["rope_theta"],
        t_scale=config["t_scale"],
        axes_dims=config["axes_dims"],
        axes_lens=config["axes_lens"],
        control_layers_places=control_layers_places,
        control_in_dim=control_in_dim,
    )
    missing, unexpected = controlnet.load_state_dict(converted_controlnet_state_dict)
    print(f"{missing=}")
    print(f"{unexpected=}")
    print("Saving Z-Image ControlNet in Diffusers format")
    controlnet.save_pretrained(args.output_path, max_shard_size="5GB")


if __name__ == "__main__":
    main(args)
