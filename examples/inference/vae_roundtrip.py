#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import torch
import typing

from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from PIL import Image
from torchvision import transforms  # type: ignore
from typing import Optional


def load_vae_model(
    *,
    device: torch.device,
    model_name_or_path: str,
    revision: Optional[str],
    variant: Optional[str],
    # NOTE: use subfolder="vae" if the pointed model is for stable diffusion as a whole instead of just the VAE
    subfolder: Optional[str],
) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(  # type: ignore
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        variant=variant,
    )
    assert isinstance(vae, AutoencoderKL)
    vae = vae.to(device)
    vae.eval()  # Set the model to inference mode
    return vae


def preprocess_image(
    *,
    device: torch.device,
    image_path: str,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    nhwc = transform(image).unsqueeze(0).to(device)  # type: ignore
    assert isinstance(nhwc, torch.Tensor)
    return nhwc


def postprocess_image(
    *,
    nhwc: torch.Tensor,
) -> Image.Image:
    assert nhwc.shape[0] == 1
    hwc = nhwc.squeeze(0).cpu()
    return transforms.ToPILImage()(hwc)  # type: ignore


def concatenate_images(
    *,
    left: Image.Image,
    right: Image.Image,
) -> Image.Image:
    width1, height1 = left.size
    width2, height2 = right.size
    total_width = width1 + width2
    max_height = max(height1, height2)

    new_image = Image.new("RGB", (total_width, max_height))
    new_image.paste(left, (0, 0))
    new_image.paste(right, (width1, 0))
    return new_image


def infer_and_show_images(
    *,
    device: torch.device,
    input_image_path: str,
    pretrained_model_name_or_path: str,
    revision: Optional[str],
    variant: Optional[str],
    subfolder: Optional[str],
) -> None:
    vae = load_vae_model(
        device=device,
        model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
    )
    original_image = preprocess_image(
        device=device,
        image_path=input_image_path,
    )
    with torch.no_grad():
        encoding = vae.encode(typing.cast(torch.FloatTensor, original_image))
        assert isinstance(encoding, AutoencoderKLOutput)
        latent = encoding.latent_dist.sample()  # type: ignore
        assert isinstance(latent, torch.Tensor)
        decoding = vae.decode(latent)  # type: ignore
        assert isinstance(decoding, DecoderOutput)
        reconstructed_image = decoding.sample

    original_pil = postprocess_image(nhwc=original_image)
    reconstructed_pil = postprocess_image(nhwc=reconstructed_image)

    combined_image = concatenate_images(
        left=original_pil,
        right=reconstructed_pil,
    )
    combined_image.show("Original | Reconstruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with VAE")
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image for inference.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained VAE model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model version.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model file variant, e.g., 'fp16'.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder in the model file.",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Use CUDA if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_image_path = args.input_image
    assert isinstance(input_image_path, str)

    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    assert isinstance(pretrained_model_name_or_path, str)

    revision = args.revision
    assert revision is None or isinstance(revision, str)

    variant = args.variant
    assert variant is None or isinstance(variant, str)

    subfolder = args.subfolder
    assert subfolder is None or isinstance(subfolder, str)

    use_cuda = args.use_cuda
    assert isinstance(use_cuda, bool)

    device = torch.device("cuda" if use_cuda else "cpu")

    infer_and_show_images(
        device=device,
        input_image_path=input_image_path,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
    )


if __name__ == "__main__":
    main()
