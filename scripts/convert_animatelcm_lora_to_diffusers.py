import argparse
import re

import torch
from safetensors.torch import load_file, save_file


def convert_lora(original_state_dict):
    converted_state_dict = {}
    for k, v in original_state_dict.items():
        if "pos_encoder" in k:
            continue
        if "alpha" in k:
            continue
        else:
            diffusers_key = (
                k.replace(".norms.0", ".norm1")
                .replace(".norms.1", ".norm2")
                .replace(".ff_norm", ".norm3")
                .replace(".attention_blocks.0", ".attn1")
                .replace(".attention_blocks.1", ".attn2")
                .replace(".temporal_transformer", "")
                .replace("lora_unet_", "")
            )
            diffusers_key = diffusers_key.replace("to_out_0_", "to_out_")
            diffusers_key = diffusers_key.replace("mid_block_", "mid_block.")
            diffusers_key = diffusers_key.replace("attn1_", "attn1.processor.")
            diffusers_key = diffusers_key.replace("attn2_", "attn2.processor.")
            diffusers_key = diffusers_key.replace(".lora_", "_lora.")
            diffusers_key = re.sub(r'_(\d+)_', r'.\1.', diffusers_key)

            converted_state_dict[diffusers_key] = v

    return converted_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    state_dict = load_file(args.ckpt_path)

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    converted_state_dict = convert_lora(state_dict)

    # convert to new format
    output_dict = {}
    for module_name, params in converted_state_dict.items():
        if type(params) is not torch.Tensor:
            continue
        output_dict.update({f"unet.{module_name}": params})

    save_file(output_dict, f"{args.output_path}/diffusion_pytorch_model.safetensors")
