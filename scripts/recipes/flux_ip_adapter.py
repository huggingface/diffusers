import argparse

import safetensors.torch
from huggingface_hub import hf_hub_download

from diffusers.loaders.conversion import get_conversion
from diffusers.utils.import_utils import is_transformers_available


if is_transformers_available():
    from transformers import CLIPVisionModelWithProjection

    vision = True
else:
    vision = False

"""
python scripts/recipes/flux_ip_adapter.py  \
--original_state_dict_repo_id "XLabs-AI/flux-ip-adapter" \
--filename "flux-ip-adapter.safetensors"
--output_path "flux-ip-adapter-hf/"
"""


parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--filename", default="flux.safetensors", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--vision_pretrained_or_path", default="openai/clip-vit-large-patch14", type=str)

args = parser.parse_args()


def load_original_checkpoint(args):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError(" please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


def convert_flux_ipadapter_checkpoint_to_diffusers(original_state_dict, num_layers):
    return get_conversion("FluxIPAdapter", {"num_layers": num_layers}).to_diffusers(original_state_dict)


def main(args):
    original_ckpt = load_original_checkpoint(args)

    num_layers = 19
    converted_ip_adapter_state_dict = convert_flux_ipadapter_checkpoint_to_diffusers(original_ckpt, num_layers)

    from pathlib import Path

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    print("Saving Flux IP-Adapter in Diffusers format.")
    safetensors.torch.save_file(converted_ip_adapter_state_dict, f"{args.output_path}/model.safetensors")

    if vision:
        model = CLIPVisionModelWithProjection.from_pretrained(args.vision_pretrained_or_path)
        model.save_pretrained(f"{args.output_path}/image_encoder")


if __name__ == "__main__":
    main(args)
