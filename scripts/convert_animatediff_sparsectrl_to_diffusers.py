import argparse
from typing import Dict

import torch
import torch.nn as nn

from diffusers import SparseControlNetModel


KEYS_RENAME_MAPPING = {
    ".attention_blocks.0": ".attn1",
    ".attention_blocks.1": ".attn2",
    ".attn1.pos_encoder": ".pos_embed",
    ".ff_norm": ".norm3",
    ".norms.0": ".norm1",
    ".norms.1": ".norm2",
    ".temporal_transformer": "",
}


def convert(original_state_dict: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
    converted_state_dict = {}

    for key in list(original_state_dict.keys()):
        renamed_key = key
        for new_name, old_name in KEYS_RENAME_MAPPING.items():
            renamed_key = renamed_key.replace(new_name, old_name)
        converted_state_dict[renamed_key] = original_state_dict.pop(key)

    return converted_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    parser.add_argument(
        "--max_motion_seq_length",
        type=int,
        default=32,
        help="Max motion sequence length supported by the motion adapter",
    )
    parser.add_argument(
        "--conditioning_channels", type=int, default=4, help="Number of channels in conditioning input to controlnet"
    )
    parser.add_argument(
        "--use_simplified_condition_embedding",
        action="store_true",
        default=False,
        help="Whether or not to use simplified condition embedding. When `conditioning_channels==4` i.e. latent inputs, set this to `True`. When `conditioning_channels==3` i.e. image inputs, set this to `False`",
    )
    parser.add_argument(
        "--save_fp16",
        action="store_true",
        default=False,
        help="Whether or not to save model in fp16 precision along with fp32",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether or not to push saved model to the HF hub"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    if "state_dict" in state_dict.keys():
        state_dict: dict = state_dict["state_dict"]

    controlnet = SparseControlNetModel(
        conditioning_channels=args.conditioning_channels,
        motion_max_seq_length=args.max_motion_seq_length,
        use_simplified_condition_embedding=args.use_simplified_condition_embedding,
    )

    state_dict = convert(state_dict)
    controlnet.load_state_dict(state_dict, strict=True)

    controlnet.save_pretrained(args.output_path, push_to_hub=args.push_to_hub)
    if args.save_fp16:
        controlnet = controlnet.to(dtype=torch.float16)
        controlnet.save_pretrained(args.output_path, variant="fp16", push_to_hub=args.push_to_hub)
