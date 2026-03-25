"""
A script to convert Stable Diffusion 3.5 ControlNet checkpoints to the Diffusers format.

Example:
    Convert a SD3.5 ControlNet checkpoint to Diffusers format using local file:
    ```bash
    python scripts/convert_sd3_controlnet_to_diffusers.py \
        --checkpoint_path "path/to/local/sd3.5_large_controlnet_canny.safetensors" \
        --output_path "output/sd35-controlnet-canny" \
        --dtype "fp16"  # optional, defaults to fp32
    ```

    Or download and convert from HuggingFace repository:
    ```bash
    python scripts/convert_sd3_controlnet_to_diffusers.py \
        --original_state_dict_repo_id "stabilityai/stable-diffusion-3.5-controlnets" \
        --filename "sd3.5_large_controlnet_canny.safetensors" \
        --output_path "/raid/yiyi/sd35-controlnet-canny-diffusers" \
        --dtype "fp32"  # optional, defaults to fp32
    ```

Note:
    The script supports the following ControlNet types from SD3.5:
    - Canny edge detection
    - Depth estimation
    - Blur detection

    The checkpoint files can be downloaded from:
    https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets
"""

import argparse

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from diffusers import SD3ControlNetModel


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to local checkpoint file")
parser.add_argument(
    "--original_state_dict_repo_id", type=str, default=None, help="HuggingFace repo ID containing the checkpoint"
)
parser.add_argument("--filename", type=str, default=None, help="Filename of the checkpoint in the HF repo")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the converted model")
parser.add_argument(
    "--dtype", type=str, default="fp32", help="Data type for the converted model (fp16, bf16, or fp32)"
)

args = parser.parse_args()


def load_original_checkpoint(args):
    if args.original_state_dict_repo_id is not None:
        if args.filename is None:
            raise ValueError("When using `original_state_dict_repo_id`, `filename` must also be specified")
        print(f"Downloading checkpoint from {args.original_state_dict_repo_id}/{args.filename}")
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=args.filename)
    elif args.checkpoint_path is not None:
        print(f"Loading checkpoint from local path: {args.checkpoint_path}")
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


def convert_sd3_controlnet_checkpoint_to_diffusers(original_state_dict):
    converted_state_dict = {}

    # Direct mappings for controlnet blocks
    for i in range(19):  # 19 controlnet blocks
        converted_state_dict[f"controlnet_blocks.{i}.weight"] = original_state_dict[f"controlnet_blocks.{i}.weight"]
        converted_state_dict[f"controlnet_blocks.{i}.bias"] = original_state_dict[f"controlnet_blocks.{i}.bias"]

    # Positional embeddings
    converted_state_dict["pos_embed_input.proj.weight"] = original_state_dict["pos_embed_input.proj.weight"]
    converted_state_dict["pos_embed_input.proj.bias"] = original_state_dict["pos_embed_input.proj.bias"]

    # Time and text embeddings
    time_text_mappings = {
        "time_text_embed.timestep_embedder.linear_1.weight": "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_1.bias": "time_text_embed.timestep_embedder.linear_1.bias",
        "time_text_embed.timestep_embedder.linear_2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
        "time_text_embed.timestep_embedder.linear_2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
        "time_text_embed.text_embedder.linear_1.weight": "time_text_embed.text_embedder.linear_1.weight",
        "time_text_embed.text_embedder.linear_1.bias": "time_text_embed.text_embedder.linear_1.bias",
        "time_text_embed.text_embedder.linear_2.weight": "time_text_embed.text_embedder.linear_2.weight",
        "time_text_embed.text_embedder.linear_2.bias": "time_text_embed.text_embedder.linear_2.bias",
    }

    for new_key, old_key in time_text_mappings.items():
        if old_key in original_state_dict:
            converted_state_dict[new_key] = original_state_dict[old_key]

    # Transformer blocks
    for i in range(19):
        # Split QKV into separate Q, K, V
        qkv_weight = original_state_dict[f"transformer_blocks.{i}.attn.qkv.weight"]
        qkv_bias = original_state_dict[f"transformer_blocks.{i}.attn.qkv.bias"]
        q, k, v = torch.chunk(qkv_weight, 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)

        block_mappings = {
            f"transformer_blocks.{i}.attn.to_q.weight": q,
            f"transformer_blocks.{i}.attn.to_q.bias": q_bias,
            f"transformer_blocks.{i}.attn.to_k.weight": k,
            f"transformer_blocks.{i}.attn.to_k.bias": k_bias,
            f"transformer_blocks.{i}.attn.to_v.weight": v,
            f"transformer_blocks.{i}.attn.to_v.bias": v_bias,
            # Output projections
            f"transformer_blocks.{i}.attn.to_out.0.weight": original_state_dict[
                f"transformer_blocks.{i}.attn.proj.weight"
            ],
            f"transformer_blocks.{i}.attn.to_out.0.bias": original_state_dict[
                f"transformer_blocks.{i}.attn.proj.bias"
            ],
            # Feed forward
            f"transformer_blocks.{i}.ff.net.0.proj.weight": original_state_dict[
                f"transformer_blocks.{i}.mlp.fc1.weight"
            ],
            f"transformer_blocks.{i}.ff.net.0.proj.bias": original_state_dict[f"transformer_blocks.{i}.mlp.fc1.bias"],
            f"transformer_blocks.{i}.ff.net.2.weight": original_state_dict[f"transformer_blocks.{i}.mlp.fc2.weight"],
            f"transformer_blocks.{i}.ff.net.2.bias": original_state_dict[f"transformer_blocks.{i}.mlp.fc2.bias"],
            # Norms
            f"transformer_blocks.{i}.norm1.linear.weight": original_state_dict[
                f"transformer_blocks.{i}.adaLN_modulation.1.weight"
            ],
            f"transformer_blocks.{i}.norm1.linear.bias": original_state_dict[
                f"transformer_blocks.{i}.adaLN_modulation.1.bias"
            ],
        }
        converted_state_dict.update(block_mappings)

    return converted_state_dict


def main(args):
    original_ckpt = load_original_checkpoint(args)
    original_dtype = next(iter(original_ckpt.values())).dtype

    # Initialize dtype with fp32 as default
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be one of: fp16, bf16, fp32")

    if dtype != original_dtype:
        print(
            f"Converting checkpoint from {original_dtype} to {dtype}. This can lead to unexpected results, proceed with caution."
        )

    converted_controlnet_state_dict = convert_sd3_controlnet_checkpoint_to_diffusers(original_ckpt)

    controlnet = SD3ControlNetModel(
        patch_size=2,
        in_channels=16,
        num_layers=19,
        attention_head_dim=64,
        num_attention_heads=38,
        joint_attention_dim=None,
        caption_projection_dim=2048,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=None,
        pos_embed_type=None,
        use_pos_embed=False,
        force_zeros_for_pooled_projection=False,
    )

    controlnet.load_state_dict(converted_controlnet_state_dict, strict=True)

    print(f"Saving SD3 ControlNet in Diffusers format in {args.output_path}.")
    controlnet.to(dtype).save_pretrained(args.output_path)


if __name__ == "__main__":
    main(args)
