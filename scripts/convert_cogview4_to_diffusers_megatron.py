"""
Convert a CogView4 checkpoint from Megatron to the Diffusers format.

Example usage:
    python scripts/convert_cogview4_to_diffusers.py \
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
from tqdm import tqdm
from transformers import GlmModel, PreTrainedTokenizerFast

from diffusers import (
    AutoencoderKL,
    CogView4ControlPipeline,
    CogView4Pipeline,
    CogView4Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint


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


def swap_scale_shift(weight, dim):
    """
    Swap the scale and shift components in the weight tensor.

    Args:
        weight (torch.Tensor): The original weight tensor.
        dim (int): The dimension along which to split.

    Returns:
        torch.Tensor: The modified weight tensor with scale and shift swapped.
    """
    shift, scale = weight.chunk(2, dim=dim)
    new_weight = torch.cat([scale, shift], dim=dim)
    return new_weight


def convert_megatron_transformer_checkpoint_to_diffusers(
    ckpt_path: str,
    num_layers: int,
    num_heads: int,
    hidden_size: int,
):
    """
    Convert a Megatron Transformer checkpoint to Diffusers format.

    Args:
        ckpt_path (str): Path to the Megatron Transformer checkpoint.
        num_layers (int): Number of Transformer layers.
        num_heads (int): Number of attention heads.
        hidden_size (int): Hidden size of the Transformer.

    Returns:
        dict: The converted state dictionary compatible with Diffusers.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mega = ckpt["model"]

    new_state_dict = {}

    # Patch Embedding
    new_state_dict["patch_embed.proj.weight"] = mega["encoder_expand_linear.weight"].reshape(
        hidden_size, 128 if args.control else 64
    )
    new_state_dict["patch_embed.proj.bias"] = mega["encoder_expand_linear.bias"]
    new_state_dict["patch_embed.text_proj.weight"] = mega["text_projector.weight"]
    new_state_dict["patch_embed.text_proj.bias"] = mega["text_projector.bias"]

    # Time Condition Embedding
    new_state_dict["time_condition_embed.timestep_embedder.linear_1.weight"] = mega[
        "time_embedding.time_embed.0.weight"
    ]
    new_state_dict["time_condition_embed.timestep_embedder.linear_1.bias"] = mega["time_embedding.time_embed.0.bias"]
    new_state_dict["time_condition_embed.timestep_embedder.linear_2.weight"] = mega[
        "time_embedding.time_embed.2.weight"
    ]
    new_state_dict["time_condition_embed.timestep_embedder.linear_2.bias"] = mega["time_embedding.time_embed.2.bias"]

    new_state_dict["time_condition_embed.condition_embedder.linear_1.weight"] = mega[
        "label_embedding.label_embed.0.weight"
    ]
    new_state_dict["time_condition_embed.condition_embedder.linear_1.bias"] = mega[
        "label_embedding.label_embed.0.bias"
    ]
    new_state_dict["time_condition_embed.condition_embedder.linear_2.weight"] = mega[
        "label_embedding.label_embed.2.weight"
    ]
    new_state_dict["time_condition_embed.condition_embedder.linear_2.bias"] = mega[
        "label_embedding.label_embed.2.bias"
    ]

    # Convert each Transformer layer
    for i in tqdm(range(num_layers), desc="Converting layers (Megatron->Diffusers)"):
        block_prefix = f"transformer_blocks.{i}."

        # AdaLayerNorm
        new_state_dict[block_prefix + "norm1.linear.weight"] = mega[f"decoder.layers.{i}.adaln.weight"]
        new_state_dict[block_prefix + "norm1.linear.bias"] = mega[f"decoder.layers.{i}.adaln.bias"]
        qkv_weight = mega[f"decoder.layers.{i}.self_attention.linear_qkv.weight"]
        qkv_bias = mega[f"decoder.layers.{i}.self_attention.linear_qkv.bias"]

        # Reshape to match SAT logic
        qkv_weight = qkv_weight.view(num_heads, 3, hidden_size // num_heads, hidden_size)
        qkv_weight = qkv_weight.permute(1, 0, 2, 3).reshape(3 * hidden_size, hidden_size)

        qkv_bias = qkv_bias.view(num_heads, 3, hidden_size // num_heads)
        qkv_bias = qkv_bias.permute(1, 0, 2).reshape(3 * hidden_size)

        # Assign to Diffusers keys
        q, k, v = torch.chunk(qkv_weight, 3, dim=0)
        qb, kb, vb = torch.chunk(qkv_bias, 3, dim=0)

        new_state_dict[block_prefix + "attn1.to_q.weight"] = q
        new_state_dict[block_prefix + "attn1.to_q.bias"] = qb
        new_state_dict[block_prefix + "attn1.to_k.weight"] = k
        new_state_dict[block_prefix + "attn1.to_k.bias"] = kb
        new_state_dict[block_prefix + "attn1.to_v.weight"] = v
        new_state_dict[block_prefix + "attn1.to_v.bias"] = vb

        # Attention Output
        new_state_dict[block_prefix + "attn1.to_out.0.weight"] = mega[
            f"decoder.layers.{i}.self_attention.linear_proj.weight"
        ]
        new_state_dict[block_prefix + "attn1.to_out.0.bias"] = mega[
            f"decoder.layers.{i}.self_attention.linear_proj.bias"
        ]

        # MLP
        new_state_dict[block_prefix + "ff.net.0.proj.weight"] = mega[f"decoder.layers.{i}.mlp.linear_fc1.weight"]
        new_state_dict[block_prefix + "ff.net.0.proj.bias"] = mega[f"decoder.layers.{i}.mlp.linear_fc1.bias"]
        new_state_dict[block_prefix + "ff.net.2.weight"] = mega[f"decoder.layers.{i}.mlp.linear_fc2.weight"]
        new_state_dict[block_prefix + "ff.net.2.bias"] = mega[f"decoder.layers.{i}.mlp.linear_fc2.bias"]

    # Final Layers
    new_state_dict["norm_out.linear.weight"] = swap_scale_shift(mega["adaln_final.weight"], dim=0)
    new_state_dict["norm_out.linear.bias"] = swap_scale_shift(mega["adaln_final.bias"], dim=0)
    new_state_dict["proj_out.weight"] = mega["output_projector.weight"]
    new_state_dict["proj_out.bias"] = mega["output_projector.bias"]

    return new_state_dict


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
    return convert_ldm_vae_checkpoint(original_state_dict, vae_config)


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
