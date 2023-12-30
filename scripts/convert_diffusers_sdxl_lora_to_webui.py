# Script for converting a HF Diffusers trained SDXL LoRAs (be it in the old, new or PEFT format)
# To the Kohya format used by some WebUIs such as AUTOMATIC1111, ComfyUI, SD.Next and others.

import argparse
import os

import torch
from safetensors.torch import load_file, save_file

from diffusers.utils import convert_state_dict_to_peft, convert_unet_state_dict_to_peft


def convert_all_diffusers_to_peft(diffusers_dict):
    try:
        peft_dict = convert_state_dict_to_peft(diffusers_dict)
    except Exception as e:
        if str(e) == "Could not automatically infer state dict type":
            peft_dict = convert_unet_state_dict_to_peft(diffusers_dict)
        else:
            raise

    if not any("lora_A" in key or "lora_B" in key for key in peft_dict.keys()):
        raise ValueError(
            "Your LoRA could not be converted to PEFT. Hence, it could not be converted to Kohya/AUTOMATIC1111 format"
        )

    return peft_dict


def convert_peft_to_kohya(state_dict):
    kohya_ss_state_dict = {}
    for peft_key, weight in state_dict.items():
        if "text_encoder_2." in peft_key:
            kohya_key = peft_key.replace("text_encoder_2.", "lora_te2.")
        elif "text_encoder." in peft_key:
            kohya_key = peft_key.replace("text_encoder.", "lora_te1.")
        elif "unet" in peft_key:
            kohya_key = peft_key.replace("unet", "lora_unet")
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(len(weight))

    return kohya_ss_state_dict


def convert_and_save(input_lora, output_lora=None):
    if output_lora is None:
        base_name = os.path.splitext(input_lora)[0]
        output_lora = f"{base_name}_webui.safetensors"

    diffusers_state_dict = load_file(input_lora)
    peft_state_dict = convert_all_diffusers_to_peft(diffusers_state_dict)
    kohya_state_dict = convert_peft_to_kohya(peft_state_dict)
    save_file(kohya_state_dict, output_lora)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA model to PEFT and then to Kohya format.")
    parser.add_argument(
        "input_lora",
        type=str,
        help="Path to the input LoRA model file in the diffusers format.",
    )
    parser.add_argument(
        "output_lora",
        type=str,
        nargs="?",
        help="Path for the converted LoRA (safetensors format for AUTOMATIC1111, ComfyUI, etc.). Optional, defaults to input name with a _webui suffix.",
    )

    args = parser.parse_args()

    convert_and_save(args.input_lora, args.output_lora)
