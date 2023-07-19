import argparse
import time
from pathlib import Path
from typing import Any, Dict, Literal

import torch

from diffusers import AsymmetricAutoencoderKL


ASYMMETRIC_AUTOENCODER_KL_x_1_5_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "down_block_out_channels": [128, 256, 512, 512],
    "layers_per_down_block": 2,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "up_block_out_channels": [192, 384, 768, 768],
    "layers_per_up_block": 3,
    "act_fn": "silu",
    "latent_channels": 4,
    "norm_num_groups": 32,
    "sample_size": 256,
    "scaling_factor": 0.18215,
}

ASYMMETRIC_AUTOENCODER_KL_x_2_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "down_block_out_channels": [128, 256, 512, 512],
    "layers_per_down_block": 2,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "up_block_out_channels": [256, 512, 1024, 1024],
    "layers_per_up_block": 5,
    "act_fn": "silu",
    "latent_channels": 4,
    "norm_num_groups": 32,
    "sample_size": 256,
    "scaling_factor": 0.18215,
}


def convert_asymmetric_autoencoder_kl_state_dict(original_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    converted_state_dict = {}
    for k, v in original_state_dict.items():
        if k.startswith("encoder."):
            converted_state_dict[
                k.replace("encoder.down.", "encoder.down_blocks.")
                .replace("encoder.mid.", "encoder.mid_block.")
                .replace("encoder.norm_out.", "encoder.conv_norm_out.")
                .replace(".downsample.", ".downsamplers.0.")
                .replace(".nin_shortcut.", ".conv_shortcut.")
                .replace(".block.", ".resnets.")
                .replace(".block_1.", ".resnets.0.")
                .replace(".block_2.", ".resnets.1.")
                .replace(".attn_1.k.", ".attentions.0.to_k.")
                .replace(".attn_1.q.", ".attentions.0.to_q.")
                .replace(".attn_1.v.", ".attentions.0.to_v.")
                .replace(".attn_1.proj_out.", ".attentions.0.to_out.0.")
                .replace(".attn_1.norm.", ".attentions.0.group_norm.")
            ] = v
        elif k.startswith("decoder.") and "up_layers" not in k:
            converted_state_dict[
                k.replace("decoder.encoder.", "decoder.condition_encoder.")
                .replace(".norm_out.", ".conv_norm_out.")
                .replace(".up.0.", ".up_blocks.3.")
                .replace(".up.1.", ".up_blocks.2.")
                .replace(".up.2.", ".up_blocks.1.")
                .replace(".up.3.", ".up_blocks.0.")
                .replace(".block.", ".resnets.")
                .replace("mid", "mid_block")
                .replace(".0.upsample.", ".0.upsamplers.0.")
                .replace(".1.upsample.", ".1.upsamplers.0.")
                .replace(".2.upsample.", ".2.upsamplers.0.")
                .replace(".nin_shortcut.", ".conv_shortcut.")
                .replace(".block_1.", ".resnets.0.")
                .replace(".block_2.", ".resnets.1.")
                .replace(".attn_1.k.", ".attentions.0.to_k.")
                .replace(".attn_1.q.", ".attentions.0.to_q.")
                .replace(".attn_1.v.", ".attentions.0.to_v.")
                .replace(".attn_1.proj_out.", ".attentions.0.to_out.0.")
                .replace(".attn_1.norm.", ".attentions.0.group_norm.")
            ] = v
        elif k.startswith("quant_conv."):
            converted_state_dict[k] = v
        elif k.startswith("post_quant_conv."):
            converted_state_dict[k] = v
        else:
            print(f"  skipping key `{k}`")
    # fix weights shape
    for k, v in converted_state_dict.items():
        if (
            (k.startswith("encoder.mid_block.attentions.0") or k.startswith("decoder.mid_block.attentions.0"))
            and k.endswith("weight")
            and ("to_q" in k or "to_k" in k or "to_v" in k or "to_out" in k)
        ):
            converted_state_dict[k] = converted_state_dict[k][:, :, 0, 0]

    return converted_state_dict


def get_asymmetric_autoencoder_kl_from_original_checkpoint(
    scale: Literal["1.5", "2"], original_checkpoint_path: str, map_location: torch.device
) -> AsymmetricAutoencoderKL:
    print("Loading original state_dict")
    original_state_dict = torch.load(original_checkpoint_path, map_location=map_location)
    original_state_dict = original_state_dict["state_dict"]
    print("Converting state_dict")
    converted_state_dict = convert_asymmetric_autoencoder_kl_state_dict(original_state_dict)
    kwargs = ASYMMETRIC_AUTOENCODER_KL_x_1_5_CONFIG if scale == "1.5" else ASYMMETRIC_AUTOENCODER_KL_x_2_CONFIG
    print("Initializing AsymmetricAutoencoderKL model")
    asymmetric_autoencoder_kl = AsymmetricAutoencoderKL(**kwargs)
    print("Loading weight from converted state_dict")
    asymmetric_autoencoder_kl.load_state_dict(converted_state_dict)
    asymmetric_autoencoder_kl.eval()
    print("AsymmetricAutoencoderKL successfully initialized")
    return asymmetric_autoencoder_kl


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        default=None,
        type=str,
        required=True,
        help="Asymmetric VQGAN scale: `1.5` or `2`",
    )
    parser.add_argument(
        "--original_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the original Asymmetric VQGAN checkpoint",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="Path to save pretrained AsymmetricAutoencoderKL model",
    )
    parser.add_argument(
        "--map_location",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading the checkpoint",
    )
    args = parser.parse_args()

    assert args.scale in ["1.5", "2"], f"{args.scale} should be `1.5` of `2`"
    assert Path(args.original_checkpoint_path).is_file()

    asymmetric_autoencoder_kl = get_asymmetric_autoencoder_kl_from_original_checkpoint(
        scale=args.scale,
        original_checkpoint_path=args.original_checkpoint_path,
        map_location=torch.device(args.map_location),
    )
    print("Saving pretrained AsymmetricAutoencoderKL")
    asymmetric_autoencoder_kl.save_pretrained(args.output_path)
    print(f"Done in {time.time() - start:.2f} seconds")
