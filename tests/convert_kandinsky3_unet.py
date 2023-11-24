#!/usr/bin/env python3
import torch
import argparse
from diffusers import Kandinsky3UNet
from safetensors.torch import load_file

MAPPING = {
    "to_time_embed.1": "time_embedding.linear_1",
    "to_time_embed.3": "time_embedding.linear_2",
    "in_layer": "conv_in",
    "out_layer.0": "conv_norm_out",
    "out_layer.2": "conv_out",
    "down_samples": "down_blocks",
    "up_samples": "up_blocks",
    "projection_lin": "encoder_hid_proj.projection_linear",
    "projection_ln": "encoder_hid_proj.projection_norm",
    "feature_pooling": "add_time_condition",
    "to_query": "to_q",
    "to_key": "to_k",
    "to_value": "to_v",
    "output_layer": "to_out.0",
}
# MAPPING = {}


def convert_state_dict(unet_state_dict):
    """
    Convert the state dict of a U-Net model to match the key format expected by Kandinsky3UNet model.
    Args:
        unet_model (torch.nn.Module): The original U-Net model.
        unet_kandi3_model (torch.nn.Module): The Kandinsky3UNet model to match keys with.

    Returns:
        OrderedDict: The converted state dictionary.
    """
    # Example of renaming logic (this will vary based on your model's architecture)
    converted_state_dict = {}
    for key in unet_state_dict:
        new_key = key
        for pattern, new_pattern in MAPPING.items():
            new_key = new_key.replace(pattern, new_pattern)

        converted_state_dict[new_key] = unet_state_dict[key]

    return converted_state_dict

def main(model_path, output_path):
    # Load your original U-Net model
    unet_state_dict = load_file(model_path)

    # Initialize your Kandinsky3UNet model
    config = {

    }

    # Convert the state dict
    converted_state_dict = convert_state_dict(unet_state_dict)

    unet = Kandinsky3UNet(config)
    unet.load_state_dict(converted_state_dict)

    unet.save_pretrained(output_path)
    print(f"Converted model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert U-Net PyTorch model to Kandinsky3UNet format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original U-Net PyTorch model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the converted model")

    args = parser.parse_args()
    main(args.model_path, args.output_path)
