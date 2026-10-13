import argparse
import json

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from diffusers import Flux3Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers


"""
Converts the FLUX.3 image transformer from a BFL model package (`config.json` + `model.safetensors`) to diffusers.

# from the Hub package
python scripts/convert_flux3_to_diffusers.py \
  --original_state_dict_repo_id <release repo id> \
  --output_path flux3-image-diffusers

# from local files
python scripts/convert_flux3_to_diffusers.py \
  --checkpoint_path /path/to/model.safetensors --config_path /path/to/config.json \
  --output_path flux3-image-diffusers

The image VAE (`ae.safetensors`) is the FLUX.2 autoencoder; convert it with the FLUX.2 script:

python scripts/convert_flux2_to_diffusers.py --vae --checkpoint_path /path/to/ae.safetensors --output_path flux3-image-diffusers

The text encoder (`text_encoder/`, `tokenizer/`) is Qwen3-VL in transformers format and needs no conversion.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--dit_filename", default="model.safetensors", type=str)
parser.add_argument("--config_filename", default="config.json", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument(
    "--config_path",
    default=None,
    type=str,
    help="the package config.json; the default architecture is used when omitted",
)
parser.add_argument("--dit_dtype", type=str, default="bf16", choices=["bf16", "fp32"])
parser.add_argument("--output_path", type=str, required=True)


def load_original_checkpoint(args):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.dit_filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError("please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    return safetensors.torch.load_file(ckpt_path)


def load_package_config(args):
    if args.config_path is not None:
        config_path = args.config_path
    elif args.original_state_dict_repo_id is not None:
        config_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.config_filename)
    else:
        return None
    with open(config_path) as f:
        return json.load(f)


def transformer_config_from_package(package_config):
    """The diffusers config of the package's `dit` section."""
    dit = package_config["dit"]
    if dit.get("vec_in_dim") is not None:
        raise ValueError(
            "this package has a pooled-vector embedder (`vec_in_dim`), which Flux3Transformer2DModel does not have"
        )
    if list(dit["out_modes"]) != ["image"]:
        raise ValueError(f"expected an image-only package (`out_modes` == ['image']), got {dit['out_modes']}")
    return {
        "in_channels": dit["in_channels"]["image"],
        "cond_in_channels": dit["in_channels"]["image_cond"],
        "num_layers": dit["depth"],
        "num_single_layers": dit["depth_single"],
        "attention_head_dim": dit["hidden"] // dit["heads"],
        "num_attention_heads": dit["heads"],
        "joint_attention_dim": dit["in_channels"]["text"],
        "mlp_ratio": dit["mlp_ratio"],
        "axes_dims_rope": tuple(dit["axes_dim"]),
        "rope_theta": dit["theta"],
        "flow_map": dit.get("flow_map", False),
    }


def main(args):
    original_ckpt = load_original_checkpoint(args)
    converted_transformer_state_dict = convert_flux3_transformer_checkpoint_to_diffusers(original_ckpt)

    package_config = load_package_config(args)
    transformer_config = transformer_config_from_package(package_config) if package_config is not None else {}
    transformer = Flux3Transformer2DModel(**transformer_config)
    transformer.load_state_dict(converted_transformer_state_dict, strict=True)

    dtype = torch.bfloat16 if args.dit_dtype == "bf16" else torch.float32
    transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer")

    # confirm the checkpoint round-trips
    Flux3Transformer2DModel.from_pretrained(f"{args.output_path}/transformer")
    print(f"saved transformer to {args.output_path}/transformer")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
