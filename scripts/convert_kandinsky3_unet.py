#!/usr/bin/env python3
import argparse

from safetensors.torch import load_file

from diffusers import Kandinsky3UNet


MAPPING = {"to_time_embed.1": "time_embedding.linear_1", "to_time_embed.3": "time_embedding.linear_3"}
MAPPING = {}


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
        for pattern, new_pattern in MAPPING:
            new_key = key.replace(pattern, new_pattern)
        converted_state_dict[new_key] = unet_state_dict[key]

    return converted_state_dict


def main(model_path, output_path):
    # Load your original U-Net model
    unet_state_dict = load_file(model_path)

    # Initialize your Kandinsky3UNet model
    config = {}

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
