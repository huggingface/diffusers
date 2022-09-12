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
from diffusers import VQModel
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
    if "conv_out" in name:
        name = name.replace("conv_out", "conv_shortcut")
    # encoder blocks
    if "ae.encoder.blocks.0" in name:
        name = name.replace("ae.encoder.blocks.0", "encoder.conv_in")
    elif "ae.encoder.blocks.1." in name:
        name = name.replace("ae.encoder.blocks.1", "encoder.down_blocks.0.resnets.0")
    elif "ae.encoder.blocks.2." in name:
        name = name.replace("ae.encoder.blocks.2", "encoder.down_blocks.0.resnets.1")
    elif "ae.encoder.blocks.3." in name:
        name = name.replace("ae.encoder.blocks.3", "encoder.down_blocks.0.downsamplers.0")
    elif "ae.encoder.blocks.4." in name:
        name = name.replace("ae.encoder.blocks.4", "encoder.down_blocks.1.resnets.0")
    elif "ae.encoder.blocks.5." in name:
        name = name.replace("ae.encoder.blocks.5", "encoder.down_blocks.1.resnets.1")
    elif "ae.encoder.blocks.6." in name:
        name = name.replace("ae.encoder.blocks.6", "encoder.down_blocks.1.downsamplers.0")
    elif "ae.encoder.blocks.7." in name:
        name = name.replace("ae.encoder.blocks.7", "encoder.down_blocks.2.resnets.0")
    elif "ae.encoder.blocks.8." in name:
        name = name.replace("ae.encoder.blocks.8", "encoder.down_blocks.2.resnets.1")
    elif "ae.encoder.blocks.9." in name:
        name = name.replace("ae.encoder.blocks.9", "encoder.down_blocks.2.downsamplers.0")
    elif "ae.encoder.blocks.10" in name:
        name = name.replace("ae.encoder.blocks.10", "encoder.down_blocks.3.resnets.0")
    elif "ae.encoder.blocks.11" in name:
        name = name.replace("ae.encoder.blocks.11", "encoder.down_blocks.3.resnets.1")
    elif "ae.encoder.blocks.12" in name:
        name = name.replace("ae.encoder.blocks.12", "encoder.down_blocks.3.downsamplers.0")
    elif "ae.encoder.blocks.13" in name:
        name = name.replace("ae.encoder.blocks.13", "encoder.down_blocks.4.resnets.0")
    elif "ae.encoder.blocks.14" in name:
        name = name.replace("ae.encoder.blocks.14", "encoder.down_blocks.4.attentions.0")
    elif "ae.encoder.blocks.15" in name:
        name = name.replace("ae.encoder.blocks.15", "encoder.down_blocks.4.resnets.1")
    elif "ae.encoder.blocks.16" in name:
        name = name.replace("ae.encoder.blocks.16", "encoder.down_blocks.4.attentions.1")
    elif "ae.encoder.blocks.17" in name:
        name = name.replace("ae.encoder.blocks.17", "encoder.mid_block.resnets.0")
    elif "ae.encoder.blocks.18" in name:
        name = name.replace("ae.encoder.blocks.18", "encoder.mid_block.attentions.0")
    elif "ae.encoder.blocks.19" in name:
        name = name.replace("ae.encoder.blocks.19", "encoder.mid_block.resnets.1")
    elif "ae.encoder.blocks.20" in name:
        name = name.replace("ae.encoder.blocks.20", "encoder.conv_norm_out")
    elif "ae.encoder.blocks.21" in name:
        name = name.replace("ae.encoder.blocks.21", "encoder.conv_out")
    # decoder blocks
    elif "ae.generator.blocks.0" in name:
        name = name.replace("ae.generator.blocks.0", "decoder.conv_in")
    elif "ae.generator.blocks.1." in name:
        name = name.replace("ae.generator.blocks.1", "decoder.mid_block.resnets.0")
    elif "ae.generator.blocks.2." in name:
        name = name.replace("ae.generator.blocks.2", "decoder.mid_block.attentions.0")
    elif "ae.generator.blocks.3." in name:
        name = name.replace("ae.generator.blocks.3", "decoder.mid_block.resnets.1")
    elif "ae.generator.blocks.4." in name:
        name = name.replace("ae.generator.blocks.4", "decoder.up_blocks.0.resnets.0")
    elif "ae.generator.blocks.5." in name:
        name = name.replace("ae.generator.blocks.5", "decoder.up_blocks.0.attentions.0")
    elif "ae.generator.blocks.6." in name:
        name = name.replace("ae.generator.blocks.6", "decoder.up_blocks.0.resnets.1")
    elif "ae.generator.blocks.7." in name:
        name = name.replace("ae.generator.blocks.7", "decoder.up_blocks.0.attentions.1")
    elif "ae.generator.blocks.8." in name:
        name = name.replace("ae.generator.blocks.8", "decoder.up_blocks.0.upsamplers.0")
    elif "ae.generator.blocks.9." in name:
        name = name.replace("ae.generator.blocks.9", "decoder.up_blocks.1.resnets.0")
    elif "ae.generator.blocks.10." in name:
        name = name.replace("ae.generator.blocks.10", "decoder.up_blocks.1.resnets.1")
    elif "ae.generator.blocks.11." in name:
        name = name.replace("ae.generator.blocks.11", "decoder.up_blocks.1.upsamplers.0")
    elif "ae.generator.blocks.12." in name:
        name = name.replace("ae.generator.blocks.12", "decoder.up_blocks.2.resnets.0")
    elif "ae.generator.blocks.13." in name:
        name = name.replace("ae.generator.blocks.13", "decoder.up_blocks.2.resnets.1")
    elif "ae.generator.blocks.14." in name:
        name = name.replace("ae.generator.blocks.14", "decoder.up_blocks.2.upsamplers.0")
    elif "ae.generator.blocks.15." in name:
        name = name.replace("ae.generator.blocks.15", "decoder.up_blocks.3.resnets.0")
    elif "ae.generator.blocks.16." in name:
        name = name.replace("ae.generator.blocks.16", "decoder.up_blocks.3.resnets.1")
    elif "ae.generator.blocks.17." in name:
        name = name.replace("ae.generator.blocks.17", "decoder.up_blocks.3.upsamplers.0")
    elif "ae.generator.blocks.18." in name:
        name = name.replace("ae.generator.blocks.18", "decoder.up_blocks.4.resnets.0")
    elif "ae.generator.blocks.19." in name:
        name = name.replace("ae.generator.blocks.19", "decoder.up_blocks.4.resnets.1")
    elif "ae.generator.blocks.20." in name:
        name = name.replace("ae.generator.blocks.20", "decoder.conv_norm_out")
    elif "ae.generator.blocks.21." in name:
        name = name.replace("ae.generator.blocks.21", "decoder.conv_out")
    # quantizer
    elif "ae.quantize" in name:
        name = name.replace("ae.quantize", "quantize")

    # attentions of encoder + decoder blocks
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

        if key.startswith("ae"):
            new_key = rename_key(key)
            if (
                "query.weight" in new_key
                or "key.weight" in new_key
                or "value.weight" in new_key
                or "proj_attn.weight" in new_key
            ):
                val = val.squeeze()
            if "conv_shortcut" in new_key:
                if (
                    "down_blocks.2.resnets.0" not in new_key
                    and "down_blocks.4.resnets.0" not in new_key
                    and "decoder.up_blocks.1.resnets.0" not in new_key
                    and "decoder.up_blocks.3.resnets.0" not in new_key
                ):
                    continue
            orig_state_dict[new_key] = val
        else:
            pass

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vqgan_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our VQGAN structure.
    """
    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # load 🤗 model
    model = VQModel(
        in_channels=3,
        out_channels=3,
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
        ),
        up_block_types=(
            "AttnUpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels=(128, 128, 256, 256, 512),
        layers_per_block=2,
        act_fn="swish",
        final_encoder_activation=False,
        final_decoder_activation=False,
        latent_channels=256,
        num_vq_embeddings=1024,
    )

    model.eval()
    new_state_dict = convert_state_dict(state_dict, model)

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert missing_keys == ["quant_conv.weight", "quant_conv.bias", "post_quant_conv.weight", "post_quant_conv.bias"]
    assert len(unexpected_keys) == 0

    # verify outputs on an image
    image_transformations = Compose(
        [Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    pixel_values = image_transformations(prepare_img()).unsqueeze(0)

    # we forward through the encoder, quantizer and decoder separately
    # since Absorbing Diffusion's VQ-GAN doesn't have quant_conv and post_quant_conv
    encoder_output = model.encoder(pixel_values)
    quant, emb_loss, info = model.quantize(encoder_output)
    decoder_output = model.decoder(quant)

    # verify encoder output
    assert encoder_output.shape == (1, 256, 16, 16)
    assert torch.allclose(
        encoder_output[0, 0, :3, :3],
        torch.tensor([[-1.2561, -1.1712, -1.0690], [-1.3602, -1.3631, -1.3604], [-1.3849, 0.5701, 1.2044]]),
        atol=1e-4,
    )
    # verify decoder output
    assert decoder_output.shape == (1, 3, 256, 256)
    assert torch.allclose(
        decoder_output[0, 0, :3, :3],
        torch.tensor([[0.0985, 0.0838, 0.0922], [0.0892, 0.0787, 0.0870], [0.0840, 0.0913, 0.0964]]),
        atol=1e-4,
    )

    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model_name = "nielsr/test"
        model.push_to_hub(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vqgan_churches",
        type=str,
        help="Name of the VQGAN model you'd like to convert.",
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
    convert_vqgan_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
