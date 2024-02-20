# Script for converting a Hugging Face Diffusers trained SDXL LoRAs to Kohya format
# This means that you can input your diffusers-trained LoRAs and
# Get the output to work with WebUIs such as AUTOMATIC1111, ComfyUI, SD.Next and others.

# To get started you can find some cool `diffusers` trained LoRAs such as this cute Corgy
# https://huggingface.co/ignasbud/corgy_dog_LoRA/, download its `pytorch_lora_weights.safetensors` file
# and run the script:
# python convert_diffusers_sdxl_lora_to_webui.py --input_lora pytorch_lora_weights.safetensors --output_lora corgy.safetensors
# now you can use corgy.safetensors in your WebUI of choice!

# To train your own, here are some diffusers training scripts and utils that you can use and then convert:
# LoRA Ease - no code SDXL Dreambooth LoRA trainer: https://huggingface.co/spaces/multimodalart/lora-ease
# Dreambooth Advanced Training Script - state of the art techniques such as pivotal tuning and prodigy optimizer:
# - Script: https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py
# - Colab (only on Pro): https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_Dreambooth_LoRA_advanced_example.ipynb
# Canonical diffusers training scripts:
# - Script: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
# - Colab (runs on free tier): https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb

import argparse
import os

from safetensors.torch import load_file, save_file

from diffusers.utils import convert_all_state_dict_to_peft, convert_state_dict_to_kohya


def convert_and_save(input_lora, output_lora=None):
    if output_lora is None:
        base_name = os.path.splitext(input_lora)[0]
        output_lora = f"{base_name}_webui.safetensors"

    diffusers_state_dict = load_file(input_lora)
    peft_state_dict = convert_all_state_dict_to_peft(diffusers_state_dict)
    kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
    save_file(kohya_state_dict, output_lora)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA model to PEFT and then to Kohya format.")
    parser.add_argument(
        "--input_lora",
        type=str,
        required=True,
        help="Path to the input LoRA model file in the diffusers format.",
    )
    parser.add_argument(
        "--output_lora",
        type=str,
        required=False,
        help="Path for the converted LoRA (safetensors format for AUTOMATIC1111, ComfyUI, etc.). Optional, defaults to input name with a _webui suffix.",
    )

    args = parser.parse_args()

    convert_and_save(args.input_lora, args.output_lora)
