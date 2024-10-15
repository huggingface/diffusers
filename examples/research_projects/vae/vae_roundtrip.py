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
import typing
from typing import Optional, Union

import torch
from PIL import Image
from torchvision import transforms  # type: ignore

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import (
    AutoencoderKL,
    AutoencoderKLOutput,
)
from diffusers.models.autoencoders.autoencoder_tiny import (
    AutoencoderTiny,
    AutoencoderTinyOutput,
)
from diffusers.models.autoencoders.vae import DecoderOutput


SupportedAutoencoder = Union[AutoencoderKL, AutoencoderTiny]


def load_vae_model(
    *,
    device: torch.device,
    model_name_or_path: str,
    revision: Optional[str],
    variant: Optional[str],
    # NOTE: use subfolder="vae" if the pointed model is for stable diffusion as a whole instead of just the VAE
    subfolder: Optional[str],
    use_tiny_nn: bool,
) -> SupportedAutoencoder:
    if use_tiny_nn:
        # NOTE: These scaling factors don't have to be the same as each other.
        down_scale = 2
        up_scale = 2
        vae = AutoencoderTiny.from_pretrained(  # type: ignore
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
            downscaling_scaling_factor=down_scale,
            upsampling_scaling_factor=up_scale,
        )
        assert isinstance(vae, AutoencoderTiny)
    else:
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


def pil_to_nhwc(
    *,
    device: torch.device,
    image: Image.Image,
) -> torch.Tensor:
    assert image.mode == "RGB"
    transform = transforms.ToTensor()
    nhwc = transform(image).unsqueeze(0).to(device)  # type: ignore
    assert isinstance(nhwc, torch.Tensor)
    return nhwc


def nhwc_to_pil(
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
    vertical: bool = False,
) -> Image.Image:
    width1, height1 = left.size
    width2, height2 = right.size
    if vertical:
        total_height = height1 + height2
        max_width = max(width1, width2)
        new_image = Image.new("RGB", (max_width, total_height))
        new_image.paste(left, (0, 0))
        new_image.paste(right, (0, height1))
    else:
        total_width = width1 + width2
        max_height = max(height1, height2)
        new_image = Image.new("RGB", (total_width, max_height))
        new_image.paste(left, (0, 0))
        new_image.paste(right, (width1, 0))
    return new_image


def to_latent(
    *,
    rgb_nchw: torch.Tensor,
    vae: SupportedAutoencoder,
) -> torch.Tensor:
    rgb_nchw = VaeImageProcessor.normalize(rgb_nchw)  # type: ignore
    encoding_nchw = vae.encode(typing.cast(torch.FloatTensor, rgb_nchw))
    if isinstance(encoding_nchw, AutoencoderKLOutput):
        latent = encoding_nchw.latent_dist.sample()  # type: ignore
        assert isinstance(latent, torch.Tensor)
    elif isinstance(encoding_nchw, AutoencoderTinyOutput):
        latent = encoding_nchw.latents
        do_internal_vae_scaling = False  # Is this needed?
        if do_internal_vae_scaling:
            latent = vae.scale_latents(latent).mul(255).round().byte()  # type: ignore
            latent = vae.unscale_latents(latent / 255.0)  # type: ignore
            assert isinstance(latent, torch.Tensor)
    else:
        assert False, f"Unknown encoding type: {type(encoding_nchw)}"
    return latent


def from_latent(
    *,
    latent_nchw: torch.Tensor,
    vae: SupportedAutoencoder,
) -> torch.Tensor:
    decoding_nchw = vae.decode(latent_nchw)  # type: ignore
    assert isinstance(decoding_nchw, DecoderOutput)
    rgb_nchw = VaeImageProcessor.denormalize(decoding_nchw.sample)  # type: ignore
    assert isinstance(rgb_nchw, torch.Tensor)
    return rgb_nchw


def main_kwargs(
    *,
    device: torch.device,
    input_image_path: str,
    pretrained_model_name_or_path: str,
    revision: Optional[str],
    variant: Optional[str],
    subfolder: Optional[str],
    use_tiny_nn: bool,
) -> None:
    vae = load_vae_model(
        device=device,
        model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        use_tiny_nn=use_tiny_nn,
    )
    original_pil = Image.open(input_image_path).convert("RGB")
    original_image = pil_to_nhwc(
        device=device,
        image=original_pil,
    )
    print(f"Original image shape: {original_image.shape}")
    reconstructed_image: Optional[torch.Tensor] = None

    with torch.no_grad():
        latent_image = to_latent(rgb_nchw=original_image, vae=vae)
        print(f"Latent shape: {latent_image.shape}")
        reconstructed_image = from_latent(latent_nchw=latent_image, vae=vae)
        reconstructed_pil = nhwc_to_pil(nhwc=reconstructed_image)
    combined_image = concatenate_images(
        left=original_pil,
        right=reconstructed_pil,
        vertical=False,
    )
    combined_image.show("Original | Reconstruction")
    print(f"Reconstructed image shape: {reconstructed_image.shape}")


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
    parser.add_argument(
        "--use_tiny_nn",
        action="store_true",
        help="Use tiny neural network.",
    )
    return parser.parse_args()


# EXAMPLE USAGE:
#
# python vae_roundtrip.py --use_cuda --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --subfolder "vae" --input_image "foo.png"
#
# python vae_roundtrip.py --use_cuda --pretrained_model_name_or_path "madebyollin/taesd" --use_tiny_nn --input_image "foo.png"
#
def main_cli() -> None:
    args = parse_args()

    input_image_path = args.input_image
    assert isinstance(input_image_path, str)

    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    assert isinstance(pretrained_model_name_or_path, str)

    revision = args.revision
    assert isinstance(revision, (str, type(None)))

    variant = args.variant
    assert isinstance(variant, (str, type(None)))

    subfolder = args.subfolder
    assert isinstance(subfolder, (str, type(None)))

    use_cuda = args.use_cuda
    assert isinstance(use_cuda, bool)

    use_tiny_nn = args.use_tiny_nn
    assert isinstance(use_tiny_nn, bool)

    device = torch.device("cuda" if use_cuda else "cpu")

    main_kwargs(
        device=device,
        input_image_path=input_image_path,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        use_tiny_nn=use_tiny_nn,
    )


if __name__ == "__main__":
    main_cli()
