# Convert the ABot-World checkpoint (https://huggingface.co/acvlab/ABot-World-0-5B-LF) to diffusers format.
#
#   python scripts/convert_abot_world_to_diffusers.py \
#       --checkpoint_path <repo>/diffusion_pytorch_model.safetensors --output_path <out_dir> [--dtype bf16]
import argparse

import torch
from safetensors.torch import load_file

from diffusers import ABotWorldTransformer3DModel


def convert_abot_world_transformer(state_dict):
    """Map the reference CausalWanModel state dict to ABotWorldTransformer3DModel naming."""
    converted = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace("text_embedding.0.", "condition_embedder.text_embedder.0.")
        new_key = new_key.replace("text_embedding.2.", "condition_embedder.text_embedder.2.")
        new_key = new_key.replace("time_embedding.0.", "condition_embedder.time_embedder.0.")
        new_key = new_key.replace("time_embedding.2.", "condition_embedder.time_embedder.2.")
        new_key = new_key.replace("time_projection.1.", "condition_embedder.time_proj.1.")
        if ".self_attn." in new_key or ".cross_attn." in new_key:
            new_key = new_key.replace(".self_attn.", ".attn1.").replace(".cross_attn.", ".attn2.")
            new_key = new_key.replace(".q.", ".to_q.").replace(".k.", ".to_k.").replace(".v.", ".to_v.")
            new_key = new_key.replace(".o.", ".to_out.0.")
        new_key = new_key.replace(".norm3.", ".norm2.")  # the cross-attn LayerNorm
        if new_key.endswith(".modulation"):
            new_key = new_key.replace("head.modulation", "scale_shift_table")
            new_key = new_key.replace(".modulation", ".scale_shift_table")
        new_key = new_key.replace("head.head.", "proj_out.")
        converted[new_key] = value
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    state_dict = convert_abot_world_transformer(load_file(args.checkpoint_path))

    transformer = ABotWorldTransformer3DModel()
    transformer.load_state_dict(state_dict, strict=True)
    if args.dtype == "bf16":
        transformer = transformer.to(torch.bfloat16)
    transformer.save_pretrained(args.output_path)

    # round-trip check
    ABotWorldTransformer3DModel.from_pretrained(args.output_path)
    print(f"saved and round-trip loaded: {args.output_path}")


if __name__ == "__main__":
    main()
