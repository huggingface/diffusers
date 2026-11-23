"""
A script to convert DiffSynth-Studio Blockwise ControlNet checkpoints to the Diffusers format.

The DiffSynth checkpoints only contain the ControlNet-specific weights (controlnet_blocks + img_in).
The transformer backbone weights are loaded from the base Qwen-Image model.

Example:
    Convert using HuggingFace repo:
    ```bash
    python scripts/convert_diffsynth_blockwise_controlnet_to_diffusers.py \
        --original_state_dict_repo_id "DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny" \
        --filename "model.safetensors" \
        --transformer_repo_id "Qwen/Qwen-Image" \
        --output_path "output/qwenimage-blockwise-controlnet-canny" \
        --dtype "bf16"
    ```

    Or convert from a local file:
    ```bash
    python scripts/convert_diffsynth_blockwise_controlnet_to_diffusers.py \
        --checkpoint_path "path/to/model.safetensors" \
        --transformer_repo_id "Qwen/Qwen-Image" \
        --output_path "output/qwenimage-blockwise-controlnet-canny" \
        --dtype "bf16"
    ```

Note:
    Available DiffSynth blockwise ControlNet checkpoints:
    - DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny
    - DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth
    - DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint
"""

import argparse

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from diffusers import QwenImageControlNetModel, QwenImageTransformer2DModel


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to local checkpoint file")
parser.add_argument(
    "--original_state_dict_repo_id", type=str, default=None, help="HuggingFace repo ID for the blockwise checkpoint"
)
parser.add_argument("--filename", type=str, default="model.safetensors", help="Filename in the HF repo")
parser.add_argument(
    "--transformer_repo_id",
    type=str,
    default="Qwen/Qwen-Image",
    help="HuggingFace repo ID for the base transformer model",
)
parser.add_argument("--output_path", type=str, required=True, help="Path to save the converted model")
parser.add_argument(
    "--dtype", type=str, default="bf16", help="Data type for the converted model (fp16, bf16, or fp32)"
)

args = parser.parse_args()


def load_original_checkpoint(args):
    if args.original_state_dict_repo_id is not None:
        print(f"Downloading checkpoint from {args.original_state_dict_repo_id}/{args.filename}")
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.filename)
    elif args.checkpoint_path is not None:
        print(f"Loading checkpoint from local path: {args.checkpoint_path}")
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


def main(args):
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be one of: fp16, bf16, fp32")

    # Load base transformer
    print(f"Loading base transformer from {args.transformer_repo_id}...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.transformer_repo_id, subfolder="transformer", torch_dtype=dtype
    )

    # Create controlnet from transformer (copies backbone weights)
    print("Creating blockwise ControlNet from transformer...")
    controlnet = QwenImageControlNetModel.from_transformer(
        transformer,
        num_layers=transformer.config.num_layers,
        attention_head_dim=transformer.config.attention_head_dim,
        num_attention_heads=transformer.config.num_attention_heads,
        controlnet_block_type="blockwise",
    )

    # Load DiffSynth blockwise weights (controlnet_blocks + img_in only)
    original_ckpt = load_original_checkpoint(args)
    missing, unexpected = controlnet.load_state_dict(original_ckpt, strict=False)

    # Verify: only transformer backbone keys should be missing, no unexpected keys
    print(f"Missing keys (expected - backbone from transformer): {len(missing)}")
    print(f"Unexpected keys (should be 0): {len(unexpected)}")
    if unexpected:
        print(f"WARNING: Unexpected keys found: {unexpected}")

    # Free transformer memory
    del transformer

    print(f"Saving blockwise ControlNet in Diffusers format to {args.output_path}")
    controlnet.to(dtype).save_pretrained(args.output_path)
    print("Done!")


if __name__ == "__main__":
    main(args)
