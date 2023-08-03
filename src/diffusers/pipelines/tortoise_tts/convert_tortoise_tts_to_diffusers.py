"""Conversion script for Tortoise TTS model checkpoints."""

import argparse
import os

import numpy as np
import torch

from diffusers.pipelines.tortoise_tts.modeling_autoregressive import TortoiseTTSAutoregressiveModel
from diffusers.pipelines.tortoise_tts.modeling_common import ConditioningEncoder, RandomLatentConverter
from diffusers.pipelines.tortoise_tts.modeling_diffusion import TortoiseTTSDenoisingModel


# Hardcode Tortoise TTS default model configs.
CONDITIONING_ENC_AUTO_CONFIG = {
    "in_channels": 80,
    "out_channels": 1024,
    "num_layers": 6,
    "attention_head_dim": 64,  # TODO: 1024 / 16 (=num_heads) = 64? confirm
    "relative_pos_embeddings": False,
    "input_transform": "conv",
    "input_conv_kernel_size": 1,
    "input_conv_stride": 1,
    "input_conv_padding": 0,
    "output_transform": None,
    "output_type": "first",
}

# tortoise.models.diffusion_decoder.DiffusionTts.contextual_embedder
CONDITIONING_ENC_DIFF_CONFIG = {
    "in_channels": 100,
    "out_channels": 2048,
    "num_layers": 5,
    "attention_head_dim": 128,  # TODO: 2048 / 16 (=num_heads) = 128? confirm
    "relative_pos_embeddings": True,
    "input_transform": "conv2",
    "input_conv_kernel_size": 3,
    "input_conv_stride": 2,
    "input_conv_padding": 1,
    "input_conv2_hidden_dim": 1024,
    "output_transform": "groupnorm",
    "output_type": "none",  # TODO: confirm
}

RLG_AUTO_CONFIG = {
    "channels": 1024,
    "num_equallinear_layers": 5,
    "lr_mul": 0.1,
}

RLG_DIFFUSION_CONFIG = {
    "channels": 2048,
    "num_equallinear_layers": 5,
    "lr_mul": 0.1,
}


def prepare_prefixes(key_prefix, new_key_prefix):
    if key_prefix:
        prefix = key_prefix + "."
    
    if new_key_prefix:
        new_prefix = new_key_prefix + "."
    
    return prefix, new_prefix


def convert_random_latent_converter(checkpoint, config):
    # Convert the EqualLinear layers
    new_checkpoint = {}
    for i in range(config["num_equallinear_layers"]):
        new_checkpoint[f"equallinear.{i}.weight"] = checkpoint[f"layers.{i}.weight"]
        new_checkpoint[f"equallinear.{i}.bias"] = checkpoint[f"layers.{i}.bias"]
    # Convert the final Linear layer
    new_checkpoint[f"linear.weight"] = checkpoint[f"layers.{i + 1}.weight"]
    new_checkpoint[f"linear.bias"] = checkpoint[f"layers.{i + 1}.bias"]

    return new_checkpoint


def convert_attention_block(
    checkpoint,
    config,
    new_checkpoint={},
    key_prefix="",
    new_key_prefix="",
):
    # checkpoint: torch state dict
    # config: dict of argument, value pairs when initializing the attention block
    # key_prefix: prefix in original checkpoint (if submodule of another checkpoint)
    # new_key_prefix: prefix in diffusers checkpoint (if submodule of )
    prefix, new_prefix = prepare_prefixes(key_prefix, new_key_prefix)

    # TODO: convert Tortoise TTS attention block to diffuser attention block


def convert_conditioning_encoder(
    checkpoint,
    config,
    new_checkpoint={},
    key_prefix="",
    new_key_prefix="",
    output_transform_key=None,
):
    prefix, new_prefix = prepare_prefixes(key_prefix, new_key_prefix)

    # Handle input transform
    start_idx = 0
    attn_name = ""
    input_transform = config["input_transform"]
    if input_transform == "conv":
        if prefix + "init" in checkpoint:
            new_checkpoint[f"{new_prefix}input_transform.weight"] = checkpoint[f"{prefix}init.weight"]
            new_checkpoint[f"{new_prefix}input_transform.bias"] = checkpoint[f"{prefix}init.bias"]
            attn_name = "attn."
        else:
            # part of nn.Sequential
            new_checkpoint[f"{new_prefix}input_transform.weight"] = checkpoint[f"{prefix}.0.weight"]
            new_checkpoint[f"{new_prefix}input_transform.bias"] = checkpoint[f"{prefix}.0.bias"]
            start_idx = 1
    elif input_transform == "conv2":
        # part of nn.Sequential
        new_checkpoint[f"{new_prefix}input_transform.0.weight"] = checkpoint[f"{prefix}.0.weight"]
        new_checkpoint[f"{new_prefix}input_transform.0.bias"] = checkpoint[f"{prefix}.0.bias"]
        new_checkpoint[f"{new_prefix}input_transform.1.weight"] = checkpoint[f"{prefix}.1.weight"]
        new_checkpoint[f"{new_prefix}input_transform.1.bias"] = checkpoint[f"{prefix}.1.bias"]
        start_idx = 2
    
    # Handle attention layers
    # These are always within a nn.Sequential container
    for i in range(config["num_layers"]):
        attn_prefix = prefix + attn_name + str(start_idx + i)
        new_attn_prefix = new_prefix + "attention." + str(i)
        convert_attention_block(checkpoint, config, new_checkpoint, attn_prefix, new_attn_prefix)
    
    # Handle output transform
    if config["output_transform"] == "groupnorm":
        # requires full key for output transform
        new_checkpoint[f"{new_prefix}output_transform.weight"] = checkpoint[f"{output_transform_key}.weight"]
        new_checkpoint[f"{new_prefix}output_transform.bias"] = checkpoint[f"{output_transform_key}.bias"]
    
    return new_checkpoint


# Temporary test functions for loading Tortoise TTS modeling components
@torch.no_grad()
def test_rlg(rlg_model, expected_output):
    generator = torch.manual_seed(0)

    noise = torch.randn((1, rlg_model.config.channels), generator=generator)
    random_latents = rlg_model(noise).latents

    assert np.abs(random_latents[0, -9:] - expected_output).max() < 1e-3, "RLG outputs are different"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Use separate arguments for each modeling component for now
    parser.add_argument(
        "--autoregressive_checkpoint_path",
        default=None,
        type=str,
        help="Path to the original Tortoise TTS autoregressive model checkpoint.",
    )
    parser.add_argument("--clvp_checkpoint_path", default=None, type=str, help="Path to the CLVP model checkpoint.")
    parser.add_argument(
        "--diffusion_checkpoint_path",
        default=None,
        type=str,
        help="Path to the original Tortoise TTS diffusion denoising model checkpoint.",
    )
    parser.add_argument(
        "--rlg_auto_checkpoint_path",
        default=None,
        type=str,
        help="Path to the random latent generator for the autoregressive model.",
    )
    parser.add_argument(
        "--rlg_diffuser_checkpoint_path",
        default=None,
        type=str,
        help="Path to the random latent generator for the diffusion model.",
    )
    parser.add_argument("--vocoder_checkpoint_path", default=None, type=str, help="Path to the UnivNet vocoder model.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to test loaded components against original Torotoise TTS checkpoint.",
    )

    args = parser.parse_args()

    if args.rlg_auto_checkpoint_path is not None:
        rlg_config = RLG_AUTO_CONFIG
        rlg_ckpt = torch.load(args.rlg_auto_checkpoint_path)
        converted_rlg_auto_ckpt = convert_random_latent_converter(rlg_ckpt, rlg_config)

        autoregressive_random_latent_converter = RandomLatentConverter(**rlg_config)
        autoregressive_random_latent_converter.load_state_dict(converted_rlg_auto_ckpt)

        if args.test:
            # Temporary testing code (assumes we are converting the original Tortoise-TTS v2 checkpoint)
            # https://huggingface.co/jbetker/tortoise-tts-v2/tree/main/.models
            expected_slice = np.array([0.4754, 2.1399, -0.4318, 0.7416, 1.0822, -0.6554, 2.4975, 1.4024, 0.4224])
            test_rlg(autoregressive_random_latent_converter, expected_slice)

        rlg_auto_output_path = os.path.join(args.dump_path, "autoregressive_random_latent_converter")
        autoregressive_random_latent_converter.save_pretrained(rlg_auto_output_path)
    
    if args.rlg_diffuser_checkpoint_path is not None:
        rlg_config = RLG_DIFFUSION_CONFIG
        rlg_ckpt = torch.load(args.rlg_diffuser_checkpoint_path)
        converted_rlg_diffuser_ckpt = convert_random_latent_converter(rlg_ckpt, rlg_config)

        diffusion_random_latent_converter = RandomLatentConverter(**rlg_config)
        diffusion_random_latent_converter.load_state_dict(converted_rlg_diffuser_ckpt)

        if args.test:
            # Temporary testing code (assumes we are converting the original Tortoise-TTS v2 checkpoint)
            # https://huggingface.co/jbetker/tortoise-tts-v2/tree/main/.models
            expected_slice = np.array([-0.2311, 0.1064, 0.1808, 0.0041, 0.1143, -0.1110, -0.2520, -0.0834, 0.2059])
            test_rlg(diffusion_random_latent_converter, expected_slice)

        rlg_diffusion_output_path = os.path.join(args.dump_path, "diffusion_random_latent_converter")
        diffusion_random_latent_converter.save_pretrained(rlg_diffusion_output_path)
