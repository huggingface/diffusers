import argparse
import os

import torch
from huggingface_hub import create_repo, upload_folder
from safetensors.torch import load_file, save_file


def convert_motion_module(original_state_dict):
    converted_state_dict = {}
    for k, v in original_state_dict.items():
        if "pos_encoder" in k:
            continue

        else:
            converted_state_dict[
                k.replace(".norms.0", ".norm1")
                .replace(".norms.1", ".norm2")
                .replace(".ff_norm", ".norm3")
                .replace(".attention_blocks.0", ".attn1")
                .replace(".attention_blocks.1", ".attn2")
                .replace(".temporal_transformer", "")
            ] = v

    return converted_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Whether to push the converted model to the HF or not",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.ckpt_path.endswith(".safetensors"):
        state_dict = load_file(args.ckpt_path)
    else:
        state_dict = torch.load(args.ckpt_path, map_location="cpu")

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    conv_state_dict = convert_motion_module(state_dict)

    # convert to new format
    output_dict = {}
    for module_name, params in conv_state_dict.items():
        if type(params) is not torch.Tensor:
            continue
        output_dict.update({f"unet.{module_name}": params})

    os.makedirs(args.output_path, exist_ok=True)

    filepath = os.path.join(args.output_path, "diffusion_pytorch_model.safetensors")
    save_file(output_dict, filepath)

    if args.push_to_hub:
        repo_id = create_repo(args.output_path, exist_ok=True).repo_id
        upload_folder(repo_id=repo_id, folder_path=args.output_path, repo_type="model")
