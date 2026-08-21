"""
Convert a TAEHV checkpoint (https://github.com/madebyollin/taehv, e.g. `taew2_2.pth` for the Wan 2.2 VAE) to an
`AutoencoderTinyVideo`:

    python scripts/convert_taehv_to_diffusers.py --checkpoint_path taew2_2.pth --variant taew2_2 --output_path ./taew2_2
"""

import argparse

import torch

from diffusers import AutoencoderTinyVideo


# TAEHV model configs, keyed by checkpoint name
VARIANTS = {
    "taehv": {"latent_channels": 16, "patch_size": 1},  # Hunyuan Video
    "taew2_1": {"latent_channels": 16, "patch_size": 1},  # Wan 2.1
    "taew2_2": {"latent_channels": 48, "patch_size": 2},  # Wan 2.2
    "taehv1_5": {"latent_channels": 32, "patch_size": 2},  # Hunyuan Video 1.5
    "taeltx": {  # LTX-2 / LTX-2.3
        "latent_channels": 128,
        "patch_size": 4,
        "encoder_time_downscale": (True, True, True),
        "decoder_time_upscale": (True, True, True),
    },
}

# the reference builds the encoder/decoder as `nn.Sequential`; these are the module names at each index
ENCODER_LAYERS = {
    0: "conv_in",
    2: "blocks.0.time_pool",
    3: "blocks.0.conv_down",
    4: "blocks.0.mem_blocks.0",
    5: "blocks.0.mem_blocks.1",
    6: "blocks.0.mem_blocks.2",
    7: "blocks.1.time_pool",
    8: "blocks.1.conv_down",
    9: "blocks.1.mem_blocks.0",
    10: "blocks.1.mem_blocks.1",
    11: "blocks.1.mem_blocks.2",
    12: "blocks.2.time_pool",
    13: "blocks.2.conv_down",
    14: "blocks.2.mem_blocks.0",
    15: "blocks.2.mem_blocks.1",
    16: "blocks.2.mem_blocks.2",
    17: "conv_out",
}
DECODER_LAYERS = {
    1: "conv_in",
    3: "blocks.0.mem_blocks.0",
    4: "blocks.0.mem_blocks.1",
    5: "blocks.0.mem_blocks.2",
    7: "blocks.0.time_grow",
    8: "blocks.0.conv_out",
    9: "blocks.1.mem_blocks.0",
    10: "blocks.1.mem_blocks.1",
    11: "blocks.1.mem_blocks.2",
    13: "blocks.1.time_grow",
    14: "blocks.1.conv_out",
    15: "blocks.2.mem_blocks.0",
    16: "blocks.2.mem_blocks.1",
    17: "blocks.2.mem_blocks.2",
    19: "blocks.2.time_grow",
    20: "blocks.2.conv_out",
    22: "conv_out",
}
# inside a MemBlock / TPool / TGrow
PARAM_RENAMES = {".conv.0.": ".conv1.", ".conv.2.": ".conv2.", ".conv.4.": ".conv3.", ".conv.": "."}


def convert_taehv_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        part, index, rest = key.split(".", 2)
        layers = ENCODER_LAYERS if part == "encoder" else DECODER_LAYERS
        new_key = f"{part}.{layers[int(index)]}.{rest}"
        for old, new in PARAM_RENAMES.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
                break
        converted[new_key] = value
    return converted


def convert_taehv(checkpoint_path: str, variant: str) -> AutoencoderTinyVideo:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = AutoencoderTinyVideo(**VARIANTS[variant])
    model.load_state_dict(convert_taehv_state_dict(state_dict), strict=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--variant", type=str, choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    model = convert_taehv(args.checkpoint_path, args.variant)
    model.save_pretrained(args.output_path)
    AutoencoderTinyVideo.from_pretrained(args.output_path)
    print(f"saved to {args.output_path}")
