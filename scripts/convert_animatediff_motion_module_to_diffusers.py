import argparse

import torch
from safetensors.torch import load_file

from diffusers import MotionAdapter


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
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--use_motion_mid_block", action="store_true")
    parser.add_argument("--motion_max_seq_length", type=int, default=32)
    parser.add_argument("--block_out_channels", nargs="+", default=[320, 640, 1280, 1280], type=int)
    parser.add_argument("--save_fp16", action="store_true")

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
    adapter = MotionAdapter(
        block_out_channels=args.block_out_channels,
        use_motion_mid_block=args.use_motion_mid_block,
        motion_max_seq_length=args.motion_max_seq_length,
    )
    # skip loading position embeddings
    adapter.load_state_dict(conv_state_dict, strict=False)
    adapter.save_pretrained(args.output_path)

    if args.save_fp16:
        adapter.to(dtype=torch.float16).save_pretrained(args.output_path, variant="fp16")
