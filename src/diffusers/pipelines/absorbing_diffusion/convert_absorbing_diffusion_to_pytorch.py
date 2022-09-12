# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Absorbing Diffusion checkpoints from the original repository.

URL: https://github.com/samb-t/unleashing-transformers"""


import argparse
from pathlib import Path

import torch

import requests
from diffusers.models.vae import Decoder, Encoder
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    # encoder blocks
    if "conv_out" in name:
        name = name.replace("conv_out", "conv_shortcut")
    if "ae.encoder.blocks.0" in name:
        name = name.replace("ae.encoder.blocks.0", "conv_in")
    elif "ae.encoder.blocks.1." in name:
        name = name.replace("ae.encoder.blocks.1", "down_blocks.0.resnets.0")
    elif "ae.encoder.blocks.2." in name:
        name = name.replace("ae.encoder.blocks.2", "down_blocks.0.resnets.1")
    elif "ae.encoder.blocks.3." in name:
        name = name.replace("ae.encoder.blocks.3", "down_blocks.0.downsamplers.0")
    elif "ae.encoder.blocks.4." in name:
        name = name.replace("ae.encoder.blocks.4", "down_blocks.1.resnets.0")
    elif "ae.encoder.blocks.5." in name:
        name = name.replace("ae.encoder.blocks.5", "down_blocks.1.resnets.1")
    elif "ae.encoder.blocks.6." in name:
        name = name.replace("ae.encoder.blocks.6", "down_blocks.1.downsamplers.0")
    elif "ae.encoder.blocks.7." in name:
        name = name.replace("ae.encoder.blocks.7", "down_blocks.2.resnets.0")
    elif "ae.encoder.blocks.8." in name:
        name = name.replace("ae.encoder.blocks.8", "down_blocks.2.resnets.1")
    elif "ae.encoder.blocks.9." in name:
        name = name.replace("ae.encoder.blocks.9", "down_blocks.2.downsamplers.0")
    elif "ae.encoder.blocks.10" in name:
        name = name.replace("ae.encoder.blocks.10", "down_blocks.3.resnets.0")
    elif "ae.encoder.blocks.11" in name:
        name = name.replace("ae.encoder.blocks.11", "down_blocks.3.resnets.1")
    elif "ae.encoder.blocks.12" in name:
        name = name.replace("ae.encoder.blocks.12", "down_blocks.3.downsamplers.0")
    elif "ae.encoder.blocks.13" in name:
        name = name.replace("ae.encoder.blocks.13", "down_blocks.4.resnets.0")
    elif "ae.encoder.blocks.14" in name:
        name = name.replace("ae.encoder.blocks.14", "down_blocks.4.attentions.0")
    elif "ae.encoder.blocks.15" in name:
        name = name.replace("ae.encoder.blocks.15", "down_blocks.4.resnets.1")
    elif "ae.encoder.blocks.16" in name:
        name = name.replace("ae.encoder.blocks.16", "down_blocks.4.attentions.1")
    elif "ae.encoder.blocks.17" in name:
        name = name.replace("ae.encoder.blocks.17", "mid_block.resnets.0")
    elif "ae.encoder.blocks.18" in name:
        name = name.replace("ae.encoder.blocks.18", "mid_block.attentions.0")
    elif "ae.encoder.blocks.19" in name:
        name = name.replace("ae.encoder.blocks.19", "mid_block.resnets.1")
    elif "ae.encoder.blocks.20" in name:
        name = name.replace("ae.encoder.blocks.20", "conv_norm_out")
    elif "ae.encoder.blocks.21" in name:
        name = name.replace("ae.encoder.blocks.21", "conv_out")
    # attentions of encoder blocks
    if "attentions" in name:
        if "norm" in name:
            name = name.replace("norm", "group_norm")
        if "q.weight" in name:
            name = name.replace("q.weight", "query.weight")
        if "q.bias" in name:
            name = name.replace("q.bias", "query.bias")
        if "k.weight" in name:
            name = name.replace("k.weight", "key.weight")
        if "k.bias" in name:
            name = name.replace("k.bias", "key.bias")
        if "v.weight" in name:
            name = name.replace("v.weight", "value.weight")
        if "v.bias" in name:
            name = name.replace("v.bias", "value.bias")
        if "proj_out" in name:
            name = name.replace("proj_out", "proj_attn")

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key.startswith("ae.encoder"):
            new_key = rename_key(key)
            if (
                "query.weight" in new_key
                or "key.weight" in new_key
                or "value.weight" in new_key
                or "proj_attn.weight" in new_key
            ):
                val = val.squeeze()
            if "conv_shortcut" in new_key and (
                "down_blocks.2.resnets.0" not in new_key and "down_blocks.4.resnets.0" not in new_key
            ):
                pass
            else:
                orig_state_dict[new_key] = val
        else:
            # TODO decoder, quantizer, transformer
            pass

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_yolos_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Absorbing Diffusion structure.
    """
    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # load 🤗 encoder
    encoder = Encoder(
        in_channels=3,
        out_channels=256,
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
        ),
        block_out_channels=(128, 128, 256, 256, 512),
        layers_per_block=2,
        act_fn="swish",
        double_z=False,
        final_activation=False,
    )
    encoder.eval()
    new_state_dict = convert_state_dict(state_dict, encoder)

    encoder.load_state_dict(new_state_dict)

    # verify outputs on an image
    image_transformations = Compose(
        [Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    pixel_values = image_transformations(prepare_img()).unsqueeze(0)

    encoder_out = encoder(pixel_values)

    assert encoder_out.shape == (1, 256, 16, 16)
    assert torch.allclose(
        encoder_out[0, 0, :3, :3],
        torch.tensor([[-1.2561, -1.1712, -1.0690], [-1.3602, -1.3631, -1.3604], [-1.3849, 0.5701, 1.2044]]),
        atol=1e-3,
    )
    print("Looks ok!")

    # load 🤗 decoder
    decoder = Decoder(
        in_channels=256,
        out_channels=3,
        up_block_types=(
            "AttnUpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels=(512, 256, 256, 128, 128),
        layers_per_block=2,
        act_fn="swish",
    )

    decoder.eval()
    decoder_out = decoder(encoder_out)
    print("Shape of decoder output:", decoder_out.shape)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving encoder {model_name} to {pytorch_dump_folder_path}")
        encoder.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model_name = "nielsr/test"
        encoder.push_to_hub(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vqgan_churches",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/AbsorbingDiffusion/vqgan_ema_2200000.th",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_yolos_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
