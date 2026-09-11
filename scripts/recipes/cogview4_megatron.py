"""
Convert a CogView4 checkpoint from Megatron to the Diffusers format.

Example usage:
    python scripts/build_pipeline.py cogview4_megatron \
        --transformer_checkpoint_path 'your path/cogview4_6b/mp_rank_00/model_optim_rng.pt' \
        --vae_checkpoint_path 'your path/cogview4_6b/imagekl_ch16.pt' \
        --output_path "THUDM/CogView4-6B" \
        --dtype "bf16"

Arguments:
    --transformer_checkpoint_path: Path to Transformer state dict.
    --vae_checkpoint_path: Path to VAE state dict.
    --output_path: The path to save the converted model.
    --push_to_hub: Whether to push the converted checkpoint to the HF Hub or not. Defaults to `False`.
    --text_encoder_cache_dir: Cache directory where text encoder is located. Defaults to None, which means HF_HOME will be used.
    --dtype: The dtype to save the model in (default: "bf16", options: "fp16", "bf16", "fp32"). If None, the dtype of the state dict is considered.

    Default is "bf16" because CogView4 uses bfloat16 for training.

Note: You must provide either --transformer_checkpoint_path or --vae_checkpoint_path.
"""

import argparse

import torch
from transformers import GlmModel, PreTrainedTokenizerFast

from diffusers import (
    AutoencoderKL,
    CogView4ControlPipeline,
    CogView4Pipeline,
    CogView4Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--transformer_checkpoint_path",
    default=None,
    type=str,
    help="Path to Megatron (not SAT) Transformer checkpoint, e.g., 'model_optim_rng.pt'.",
)
parser.add_argument(
    "--vae_checkpoint_path",
    default=None,
    type=str,
    help="(Optional) Path to VAE checkpoint, e.g., 'imagekl_ch16.pt'.",
)
parser.add_argument(
    "--output_path",
    required=True,
    type=str,
    help="Directory to save the final Diffusers format pipeline.",
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    default=False,
    help="Whether to push the converted model to the HuggingFace Hub.",
)
parser.add_argument(
    "--text_encoder_cache_dir",
    type=str,
    default=None,
    help="Specify the cache directory for the text encoder.",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="bf16",
    choices=["fp16", "bf16", "fp32"],
    help="Data type to save the model in.",
)

parser.add_argument(
    "--num_layers",
    type=int,
    default=28,
    help="Number of Transformer layers (e.g., 28, 48...).",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=32,
    help="Number of attention heads.",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=4096,
    help="Transformer hidden dimension size.",
)
parser.add_argument(
    "--attention_head_dim",
    type=int,
    default=128,
    help="Dimension of each attention head.",
)
parser.add_argument(
    "--time_embed_dim",
    type=int,
    default=512,
    help="Dimension of time embeddings.",
)
parser.add_argument(
    "--condition_dim",
    type=int,
    default=256,
    help="Dimension of condition embeddings.",
)
parser.add_argument(
    "--pos_embed_max_size",
    type=int,
    default=128,
    help="Maximum size for positional embeddings.",
)
parser.add_argument(
    "--control",
    action="store_true",
    default=False,
    help="Whether to use control model.",
)

args = parser.parse_args()


def convert_megatron_transformer_checkpoint_to_diffusers(ckpt_path, num_layers, num_heads, hidden_size):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    config = {
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "attention_head_dim": hidden_size // num_heads,
        "original_format": "megatron",
    }
    return convert_component_checkpoint(state, config, "CogView4Transformer2DModel")


def convert_cogview4_vae_checkpoint_to_diffusers(ckpt_path, vae_config):
    """
    Convert a CogView4 VAE checkpoint to Diffusers format.

    Args:
        ckpt_path (str): Path to the VAE checkpoint.
        vae_config (dict): Configuration dictionary for the VAE.

    Returns:
        dict: The converted VAE state dictionary compatible with Diffusers.
    """
    original_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    return convert_component_checkpoint(original_state_dict, vae_config, "AutoencoderKL")


def main(args):
    """
    Main function to convert CogView4 checkpoints to Diffusers format.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Determine the desired data type
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    transformer = None
    vae = None

    # Convert Transformer checkpoint if provided
    if args.transformer_checkpoint_path is not None:
        converted_transformer_state_dict = convert_megatron_transformer_checkpoint_to_diffusers(
            ckpt_path=args.transformer_checkpoint_path,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_size=args.hidden_size,
        )
        transformer = CogView4Transformer2DModel(
            patch_size=2,
            in_channels=32 if args.control else 16,
            num_layers=args.num_layers,
            attention_head_dim=args.attention_head_dim,
            num_attention_heads=args.num_heads,
            out_channels=16,
            text_embed_dim=args.hidden_size,
            time_embed_dim=args.time_embed_dim,
            condition_dim=args.condition_dim,
            pos_embed_max_size=args.pos_embed_max_size,
        )

        transformer.load_state_dict(converted_transformer_state_dict, strict=True)

        # Convert to the specified dtype
        if dtype is not None:
            transformer = transformer.to(dtype=dtype)

    # Convert VAE checkpoint if provided
    if args.vae_checkpoint_path is not None:
        vae_config = {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ("DownEncoderBlock2D",) * 4,
            "up_block_types": ("UpDecoderBlock2D",) * 4,
            "block_out_channels": (128, 512, 1024, 1024),
            "layers_per_block": 3,
            "act_fn": "silu",
            "latent_channels": 16,
            "norm_num_groups": 32,
            "sample_size": 1024,
            "scaling_factor": 1.0,
            "shift_factor": 0.0,
            "force_upcast": True,
            "use_quant_conv": False,
            "use_post_quant_conv": False,
            "mid_block_add_attention": False,
        }
        converted_vae_state_dict = convert_cogview4_vae_checkpoint_to_diffusers(args.vae_checkpoint_path, vae_config)
        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_state_dict, strict=True)
        if dtype is not None:
            vae = vae.to(dtype=dtype)

    # Load the text encoder and tokenizer
    text_encoder_id = "THUDM/glm-4-9b-hf"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(text_encoder_id)
    text_encoder = GlmModel.from_pretrained(
        text_encoder_id,
        cache_dir=args.text_encoder_cache_dir,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
    )
    for param in text_encoder.parameters():
        param.data = param.data.contiguous()

    # Initialize the scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_shift=0.25, max_shift=0.75, base_image_seq_len=256, use_dynamic_shifting=True, time_shift_type="linear"
    )

    # Create the pipeline
    if args.control:
        pipe = CogView4ControlPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
    else:
        pipe = CogView4Pipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    # Save the converted pipeline
    pipe.save_pretrained(
        args.output_path,
        safe_serialization=True,
        max_shard_size="5GB",
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main(args)
