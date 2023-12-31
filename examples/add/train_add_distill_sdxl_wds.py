#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import copy
import functools
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import types
from pathlib import Path
from typing import Callable, List, Optional, Union

import accelerate
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
import webdataset as wds
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from braceexpand import braceexpand
from huggingface_hub import create_repo
from packaging import version
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import BaseOutput, check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


MAX_SEQ_LENGTH = 77

# Adjust for your dataset
WDS_JSON_WIDTH = "width"  # original_width for LAION
WDS_JSON_HEIGHT = "height"  # original_height for LAION
MIN_SIZE = 700  # ~960 for LAION, ideal: 1024 if the dataset contains large images

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__)


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def resolve_interpolation_mode(interpolation_type):
    if interpolation_type == "bilinear":
        interpolation_mode = TF.InterpolationMode.BILINEAR
    elif interpolation_type == "bicubic":
        interpolation_mode = TF.InterpolationMode.BICUBIC
    elif interpolation_type == "nearest":
        interpolation_mode = TF.InterpolationMode.NEAREST
    elif interpolation_type == "lanczos":
        interpolation_mode = TF.InterpolationMode.LANCZOS
    else:
        raise ValueError(
            f"The given interpolation mode {interpolation_type} is not supported. Currently supported interpolation"
            f" modes are `bilinear`, `bicubic`, `lanczos`, and `nearest`."
        )

    return interpolation_mode


class WebdatasetFilter:
    def __init__(self, min_size=1024, max_pwatermark=0.5):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark

    def __call__(self, x):
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                filter_size = (x_json.get(WDS_JSON_WIDTH, 0.0) or 0.0) >= self.min_size and x_json.get(
                    WDS_JSON_HEIGHT, 0
                ) >= self.min_size
                filter_watermark = (x_json.get("pwatermark", 0.0) or 0.0) <= self.max_pwatermark
                return filter_size and filter_watermark
            else:
                return False
        except Exception:
            return False


class SDXLText2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 1024,
        interpolation_type: str = "bilinear",
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        use_fix_crop_and_size: bool = False,
        use_image_conditioning: bool = False,
        cond_resolution: Optional[int] = None,
        cond_interpolation_type: Optional[str] = None,
    ):
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        def get_orig_size(json):
            if use_fix_crop_and_size:
                return (resolution, resolution)
            else:
                return (int(json.get(WDS_JSON_WIDTH, 0.0)), int(json.get(WDS_JSON_HEIGHT, 0.0)))

        interpolation_mode = resolve_interpolation_mode(interpolation_type)
        if use_image_conditioning:
            cond_interpolation_mode = resolve_interpolation_mode(cond_interpolation_type)

        def transform(example):
            # resize image
            image = example["image"]
            if use_image_conditioning:
                cond_image = copy.deepcopy(image)

            image = TF.resize(image, resolution, interpolation=interpolation_mode)

            # get crop coordinates and crop image
            c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
            image = TF.crop(image, c_top, c_left, resolution, resolution)
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            example["image"] = image
            example["crop_coords"] = (c_top, c_left) if not use_fix_crop_and_size else (0, 0)

            if use_image_conditioning:
                # Prepare a separate image for image conditioning since the preprocessing pipelines are different.
                cond_image = TF.resize(cond_image, cond_resolution, interpolation=cond_interpolation_mode)
                cond_image = TF.center_crop(cond_image, output_size=(cond_resolution, cond_resolution))
                cond_image = TF.normalize(cond_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                example["cond_image"] = cond_image

            return example

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp", text="text;txt;caption", orig_size="json", handler=wds.warn_and_continue
            ),
            wds.map(filter_keys({"image", "text", "orig_size"})),
            wds.map_dict(orig_size=get_orig_size),
            wds.map(transform),
        ]

        if use_image_conditioning:
            processing_pipeline.append(wds.to_tuple("image", "text", "orig_size", "crop_coords", "cond_image"))
        else:
            processing_pipeline.append(wds.to_tuple("image", "text", "orig_size", "crop_coords"))

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.select(WebdatasetFilter(min_size=MIN_SIZE)),
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


class Denoiser:
    def __init__(self, alphas, sigmas, prediction_type="epsilon"):
        self.alphas = alphas
        self.sigmas = sigmas
        self.prediction_type = prediction_type

    def to(self, device):
        self.alphas = self.alphas.to(device)
        self.sigmas = self.sigmas.to(device)
        return self

    def __call__(self, model_output, timesteps, sample):
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)
        if self.prediction_type == "epsilon":
            pred_x_0 = (sample - sigmas * model_output) / alphas
        elif self.prediction_type == "sample":
            pred_x_0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x_0 = alphas * sample - sigmas * model_output
        else:
            raise ValueError(
                f"Prediction type {self.prediction_type} is not supported; currently, `epsilon`, `sample`, and"
                f" `v_prediction` are supported."
            )

        return pred_x_0


# Based on SpectralConv1d from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/discriminator.py#L29
class SpectralConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.utils.parametrizations.spectral_norm(self, name="weight", n_power_iterations=1, eps=1e-12, dim=0)


# Based on ResidualBlock from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/shared.py#L20
class ResidualBlock(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


# Based on make_block from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/discriminator.py#L64
class DiscHeadBlock(torch.nn.Module):
    """
    StyleGAN-T block: SpectralConv1d => GroupNorm => LeakyReLU
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        num_groups: int = 8,
        leaky_relu_neg_slope: float = 0.2,
    ):
        super().__init__()
        self.channels = channels

        self.conv = SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        )
        self.norm = torch.nn.GroupNorm(num_groups, channels)
        self.act_fn = torch.nn.LeakyReLU(leaky_relu_neg_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act_fn(x)
        return x


# Based on DiscHead in the official StyleGAN-T implementation
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/discriminator.py#L78
class DiscriminatorHead(torch.nn.Module):
    """
    Implements a StyleGAN-T-style discriminator head. The discriminator head takes in a (possibly intermediate) 1D
    sequence of tokens from the feature network, processes it, and combines it with conditioning information to output
    per-token logits.
    """

    def __init__(
        self,
        channels: int,
        c_text_embedding_dim: int,
        c_img_embedding_dim: Optional[int] = None,
        cond_map_dim: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.c_text_embedding_dim = c_text_embedding_dim
        self.c_img_embedding_dim = c_img_embedding_dim
        self.cond_map_dim = cond_map_dim

        self.input_block = DiscHeadBlock(channels, kernel_size=1)
        self.resblock = ResidualBlock(DiscHeadBlock(channels, kernel_size=9))
        # Project each token embedding from channels dimensions to cond_map_dim dimensions.
        self.cls = SpectralConv1d(channels, cond_map_dim, kernel_size=1, padding=0)

        # Also project the concatenated conditioning embeddings to dimension cond_map_dim.
        c_map_input_dim = self.c_text_embedding_dim
        if self.c_img_embedding_dim is not None:
            c_map_input_dim += self.c_img_embedding_dim
        self.c_map = torch.nn.Linear(c_map_input_dim, cond_map_dim)

    def forward(self, x: torch.Tensor, c_text: torch.Tensor, c_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Maps a 1D sequence of tokens from a feature network (e.g. ViT trained with DINO) and a conditioning embedding
        to per-token logits.

        Args:
            x (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                A sequence of 1D tokens (possibly intermediate) from a ViT feature neetwork. Note that the channels dim
                should be the same as the feature network's embedding dim.
            c_text (`torch.Tensor` of shape `(batch_size, c_text_embedding_dim)`):
                A conditioning embedding representing text conditioning information.
            c_img (`torch.Tensor` of shape `(batch_size, c_img_embedding_dim)`):
                A conditioning embedding representing image conditioning information.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length)`: batched 1D sequence of per-token logits.
        """
        hidden_states = self.input_block(x)
        hidden_states = self.resblock(hidden_states)
        out = self.cls(hidden_states)

        if self.c_img_embedding_dim is not None:
            c = torch.cat([c_text, c_img], dim=1)
        else:
            c = c_text
        # Project conditioning embedding to cond_map_dim and unsqueeze in the sequence length dimension.
        c = self.c_map(c).unsqueeze(-1)

        # Combine image features with projected conditioning embedding via a product.
        out = (out * c).sum(1, keepdim=True) * (1 / np.sqrt(self.cond_map_dim))

        return out


activations = {}


# Based on get_activation from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/vit_utils.py#L111
def get_activation(name: str) -> Callable:
    def hook(model, input, output):
        activations[name] = output

    return hook


# Based on _resize_pos_embed from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/vit_utils.py#L66
def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int) -> torch.Tensor:
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


# Based on forward_flex from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/vit_utils.py#L83
def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
    # patch proj and dynamically resize
    B, C, H, W = x.size()
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    pos_embed = self._resize_pos_embed(self.pos_embed, H // self.patch_size[1], W // self.patch_size[0])

    # add cls token
    cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # forward pass
    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)
    return x


# Based on forward_vit from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/vit_utils.py#L60
def forward_vit(pretrained: torch.nn.Module, x: torch.Tensor) -> dict:
    _ = pretrained.model.forward_flex(x)
    return {k: pretrained.rearrange(v) for k, v in activations.items()}


# Based on AddReadout from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/vit_utils.py#L36
class AddReadout(torch.nn.Module):
    def __init__(self, start_index: int = 1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


# Based on Transpose from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/vit_utils.py#L49
class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()


# Based on DINO from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/discriminator.py#L107
class FeatureNetwork(torch.nn.Module):
    """
    DINO ViT model to act as feature extractor for the discriminator.
    """

    def __init__(
        self,
        pretrained_feature_network: str = "vit_small_patch14_dinov2.lvd142m",
        patch_size: List[int] = [14, 14],
        hooks: List[int] = [2, 5, 8, 11],
        start_index: int = 1,
    ):
        super().__init__()
        self.num_hooks = len(hooks) + 1

        pretrained_model = timm.create_model(pretrained_feature_network, pretrained=True)

        # Based on make_vit_backbone from the official StyleGAN-T code
        # https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/vit_utils.py#L117
        # which I believe is itself based on https://github.com/isl-org/DPT
        model_with_hooks = torch.nn.Module()
        model_with_hooks.model = pretrained_model

        # Add hooks
        model_with_hooks.model.blocks[hooks[0]].register_forward_hook(get_activation("0"))
        model_with_hooks.model.blocks[hooks[1]].register_forward_hook(get_activation("1"))
        model_with_hooks.model.blocks[hooks[2]].register_forward_hook(get_activation("2"))
        model_with_hooks.model.blocks[hooks[3]].register_forward_hook(get_activation("3"))
        model_with_hooks.model.pos_drop.register_forward_hook(get_activation("4"))

        # Configure readout
        model_with_hooks.rearrange = torch.nn.Sequential(AddReadout(start_index), Transpose(1, 2))
        model_with_hooks.model.start_index = start_index
        model_with_hooks.model.patch_size = patch_size

        # We inject this function into the VisionTransformer instances so that
        # we can use it with interpolated position embeddings without modifying the library source.
        model_with_hooks.model.forward_flex = types.MethodType(forward_flex, model_with_hooks.model)
        model_with_hooks.model._resize_pos_embed = types.MethodType(_resize_pos_embed, model_with_hooks.model)

        self.model = model_with_hooks
        # Freeze pretrained model with hooks
        self.model = self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor):
        """
        Forward pass consisting of interpolation, ImageNet normalization, and a forward pass of self.model.

        Args:
            x (`torch.Tensor`):
                Image with pixel values in [0, 1].

        Returns:
            `Dict[Any]`: dict of activations which are intermediate features from the feature network. The dict values
            (feature embeddings) have shape `(batch_size, embed_dim, sequence_length)`.
        """
        x = F.interpolate(x, self.img_resolution, mode="area")
        x = self.norm(x)

        activation_dict = forward_vit(self.model, x)
        return activation_dict


class DiscriminatorOutput(BaseOutput):
    """
    Output class for the Discriminator module.
    """

    logits: torch.FloatTensor


# Based on ProjectedDiscriminator from the official StyleGAN-T code
# https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/networks/discriminator.py#L130
class Discriminator(torch.nn.Module):
    """
    StyleGAN-T-style discriminator for adversarial diffusion distillation (ADD).
    """

    def __init__(
        self,
        pretrained_feature_network: str = "vit_small_patch14_dinov2.lvd142m",
        c_text_embedding_dim: int = 768,
        c_img_embedding_dim: Optional[int] = None,
        cond_map_dim: int = 64,
        patch_size: List[int] = [14, 14],
        hooks: List[int] = [2, 5, 8, 11],
        start_index: int = 1,
    ):
        super().__init__()
        self.c_text_embedding_dim = c_text_embedding_dim
        self.c_img_embedding_dim = c_img_embedding_dim
        self.cond_map_dim = cond_map_dim

        # Frozen feature network, e.g. DINO
        self.feature_network = FeatureNetwork(
            pretrained_feature_network=pretrained_feature_network,
            patch_size=patch_size,
            hooks=hooks,
            start_index=start_index,
        )

        # Trainable discriminator heads
        heads = []
        for i in range(self.feature_network.num_hooks):
            heads.append(
                [
                    str(i),
                    DiscriminatorHead(
                        self.feature_network.embed_dim, c_text_embedding_dim, c_img_embedding_dim, cond_map_dim
                    ),
                ]
            )
        self.heads = torch.nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        self.feature_network = self.feature_network.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(
        self,
        x: torch.Tensor,
        c_text: torch.Tensor,
        c_img: Optional[torch.Tensor] = None,
        transform_positive: bool = True,
        return_dict: bool = True,
    ):
        # TODO: do we need the augmentations from the original StyleGAN-T code?
        if transform_positive:
            # Transform to [0, 1].
            x = x.add(1).div(2)

        # Forward pass through feature network.
        features = self.feature_network(x)

        # Apply discriminator heads.
        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[k], c_text, c_img).view(x.size(0), -1))
        logits = torch.cat(logits, dim=1)

        if not return_dict:
            return (logits,)

        return DiscriminatorOutput(logits=logits)


def log_validation(vae, unet, args, accelerator, weight_dtype, step, name="student"):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_teacher_model,
        vae=vae,
        unet=unet,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        with torch.autocast("cuda"):
            images = pipeline(
                prompt=prompt,
                num_inference_steps=1,
                num_images_per_prompt=4,
                generator=generator,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({f"validation/{name}": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        default="bilinear",
        help=(
            "The interpolation function used when resizing images to the desired resolution. Choose between `bilinear`,"
            " `bicubic`, `lanczos`, and `nearest`."
        ),
    )
    parser.add_argument(
        "--use_fix_crop_and_size",
        action="store_true",
        help="Whether or not to use the fixed crop and size for the teacher model.",
        default=False,
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--discriminator_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--discriminator_adam_beta1", type=float, default=0.0, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--discriminator_adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument("--discriminator_adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--discriminator_adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer"
    )
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # ----Adversarial Diffusion Distillation (ADD) Specific Arguments----
    parser.add_argument(
        "--pretrained_feature_network",
        type=str,
        default="vit_small_patch14_dinov2.lvd142m",
        help=(
            "The pretrained feature network used in the discriminator, typically a vision transformer (ViT) trained"
            " the DINO objective. The given identifier should be compatible with `timm.create_model`."
        ),
    )
    parser.add_argument(
        "--feature_network_patch_size",
        type=int,
        default=14,
        help="The patch size of the `pretrained_feature_network`.",
    )
    parser.add_argument(
        "--cond_map_dim",
        type=int,
        default=64,
        help=(
            "The common dimension to which the discriminator feature network features and conditioning embeddings will"
            " be projected to in the discriminator heads."
        ),
    )
    parser.add_argument(
        "--use_image_conditioning",
        action="store_true",
        help=(
            "Whether to also use an image encoder to calculate image conditioning embeddings for the discriminator. If"
            " set, the model at the timm model id given in `image_encoder_with_proj` will be used."
        ),
    )
    parser.add_argument(
        "--pretrained_image_encoder",
        type=str,
        default="vit_large_patch14_dinov2.lvd142m",
        help=(
            "An optional image encoder to add image conditioning information to the discriminator. Is used if"
            " `use_image_conditioning` is set. The model id should be loadable by `timm.create_model`. Note that ADD"
            " uses a DINOv2 ViT-L encoder (see section 4 of the paper)."
        ),
    )
    parser.add_argument(
        "--cond_resolution",
        type=int,
        default=518,
        help="Resolution to resize the original images to for image conditioning.",
    )
    parser.add_argument(
        "--cond_interpolation_type",
        type=str,
        default="bicubic",
        help=(
            "The interpolation function used when resizing the image for conditioning. Choose between `bilinear`,"
            " `bicubic`, `lanczos`, and `nearest`."
        ),
    )
    parser.add_argument(
        "--weight_schedule",
        type=str,
        default="exponential",
        help=(
            "The time-dependent weighting function gamma used for scaling the distillation loss Choose between"
            " `uniform`, `exponential`, `sds`, and `nfsd`."
        ),
    )
    parser.add_argument(
        "--student_distillation_steps",
        type=int,
        default=4,
        help="The number of student timesteps N used during distillation.",
    )
    parser.add_argument(
        "--student_timestep_schedule",
        type=str,
        default="uniform",
        help="The method by which the student timestep schedule is determined. Currently, only `uniform` is implemented.",
    )
    parser.add_argument(
        "--student_custom_timesteps",
        type=str,
        default=None,
        help=(
            "A comma-separated list of timesteps which will override the timestep schedule set in"
            " `student_timestep_schedule` if set."
        ),
    )
    parser.add_argument(
        "--discriminator_r1_strength",
        type=float,
        default=1e-05,
        help="The discriminator R1 gradient penalty strength gamma.",
    )
    parser.add_argument(
        "--distillation_weight_factor",
        type=float,
        default=2.5,
        help="Multiplicative weight factor lambda for the distillation loss on the student generator U-Net.",
    )
    parser.add_argument(
        "--w_min",
        type=float,
        default=1.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=8.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to also maintain an EMA version of the student U-Net weights."
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    parser.add_argument(
        "--ema_min_decay",
        type=float,
        default=None,
        help=(
            "The minimum EMA decay rate, which the effective EMA decay rate (e.g. if warmup is used) will never go"
            " below. If not set, the value set for `ema_decay` will be used, which results in a fixed EMA decay rate"
            " equal to that value."
        ),
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_images(image_batch, image_encoder):
    # image_encoder pre-processing is done in SDText2ImageDataset
    image_embeds = image_encoder(image_batch)
    return image_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    # 1. Create the noise scheduler and the desired noise schedule.
    # Enforce zero terminal SNR (see section 3.1 of ADD paper)
    teacher_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler", revision=args.teacher_revision
    )
    if not teacher_scheduler.config.rescale_betas_zero_snr:
        teacher_scheduler.config["rescale_betas_zero_snr"] = True
    noise_scheduler = DDPMScheduler(**teacher_scheduler.config)

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    # Note that the ADD paper parameterizes alpha and sigma as x_t = alpha_t * x_0 + sigma_t * eps
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # denoiser gets predicted original sample x_0 from prediction_type using alpha and sigma noise schedules
    denoiser = Denoiser(alpha_schedule, sigma_schedule)

    # Create time-dependent weighting schedule c(t) for scaling the GAN generator reconstruction loss term.
    if args.weight_schedule == "uniform":
        train_weight_schedule = torch.ones_like(noise_scheduler.alphas_cumprod)
    elif args.weight_schedule == "exponential":
        # Set weight schedule equal to alpha_schedule. Higher timesteps have less weight.
        train_weight_schedule = alpha_schedule
    elif args.weight_schedule == "sds":
        # Score distillation sampling weighting: alpha_t / (2 * sigma_t) * w(t)
        # NOTE: choose w(t) = 1
        # Introduced in the DreamFusion paper: https://arxiv.org/pdf/2209.14988.pdf.
        train_weight_schedule = alpha_schedule / (2 * sigma_schedule)
    elif args.weight_schedule == "nfsd":
        # Noise-free score distillation weighting
        # Introduced in "Noise-Free Score Distillation": https://arxiv.org/pdf/2310.17590.pdf.
        raise NotImplementedError("NFSD distillation weighting is not yet implemented.")
    else:
        raise ValueError(
            f"Weight schedule {args.weight_schedule} is not currently supported. Supported schedules are `uniform`,"
            f" `exponential`, `sds`, and `nfsd`."
        )

    # Create student timestep schedule tau_1, ..., tau_N.
    if args.student_custom_timesteps is not None:
        student_timestep_schedule = np.asarray(
            sorted([int(timestep.strip()) for timestep in args.student_custom_timesteps.split(",")]), dtype=np.int64
        )
    elif args.student_timestep_schedule == "uniform":
        student_timestep_schedule = (
            np.linspace(0, noise_scheduler.config.num_train_timesteps - 1, args.student_distillation_steps)
            .round()
            .astype(np.int64)
        )
    else:
        raise ValueError(
            f"Student timestep schedule {args.student_timestep_schedule} was not recognized and custom student"
            f" timesteps have not been provided. Either use one of `uniform` for `student_timestep_schedule` or"
            f" provide custom timesteps via `student_custom_timesteps`."
        )
    student_distillation_steps = student_timestep_schedule.shape[0]

    # 2. Load tokenizers from SD-XL checkpoint.
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer", revision=args.teacher_revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer_2", revision=args.teacher_revision, use_fast=False
    )

    # 3. Load text encoders from SD-XL checkpoint.
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.teacher_revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.teacher_revision, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_teacher_model, subfolder="text_encoder", revision=args.teacher_revision
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_teacher_model, subfolder="text_encoder_2", revision=args.teacher_revision
    )

    # Optionally load a image encoder model for image conditioning of the discriminator.
    if args.use_image_conditioning:
        # Set num_classes=0 so that we get image embeddings from image_encoder forward pass
        image_encoder = timm.create_model(args.pretrained_image_encoder, pretrained=True, num_classes=0)

    # 4. Load VAE from SD-XL checkpoint (or more stable VAE)
    vae_path = (
        args.pretrained_teacher_model
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD-XL checkpoint
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
    )

    # 6. Initialize GAN generator U-Net from SD-XL checkpoint with the teacher U-Net's pretrained weights
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
    )

    # Make exponential moving average (EMA) version of the student unet weights, if using.
    if args.use_ema:
        if args.ema_min_decay is None:
            # Default to `args.ema_decay`, which results in a fixed EMA decay rate throughout distillation.
            args.ema_min_decay = args.ema_decay
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            decay=args.ema_decay,
            min_decay=args.ema_min_decay,
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    # 7. Initialize GAN discriminator.
    # Use text_encoder_two here since it already projects the CLIP embedding to a fixed length vector (e.g. it's
    # already a ClipTextModelWithProjection)
    # TODO: what if there's no text_encoder_two? I think we already assume text_encoder_two exists in Step 3 above so
    # it might be fine?
    text_conditioning_dim = text_encoder_two.config.projection_dim
    img_conditioning_dim = image_encoder.num_features if args.use_image_conditioning else None
    discriminator = Discriminator(
        pretrained_feature_network=args.pretrained_feature_network,
        c_text_embedding_dim=text_conditioning_dim,
        c_img_embedding_dim=img_conditioning_dim,
        patch_size=[args.feature_network_patch_size, args.feature_network_patch_size],
    )

    # 8. Freeze teacher vae, text_encoders, and teacher_unet
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    if args.use_image_conditioning:
        image_encoder.eval()
        image_encoder.requires_grad_(False)

    unet.train()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, text_encoders, and teacher_unet to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    teacher_unet.to(accelerator.device, dtype=weight_dtype)
    if args.use_image_conditioning:
        image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Move target (EMA) unet to device but keep in full precision
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Also move the denoiser and schedules to accelerator.device
    denoiser.to(accelerator.device)
    train_weight_schedule = train_weight_schedule.to(accelerator.device)
    student_timestep_schedule = torch.from_numpy(student_timestep_schedule).to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation for generator and discriminator
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    discriminator_optimizer = optimizer_class(
        discriminator.parameters(),
        lr=args.discriminator_learning_rate,
        betas=(args.discriminator_adam_beta1, args.discriminator_adam_beta2),
        weight_decay=args.discriminator_adam_weight_decay,
        eps=args.discriminator_adam_epsilon,
    )

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, original_sizes, crop_coords, proportion_empty_prompts, text_encoders, tokenizers, is_train=True
    ):
        target_size = (args.resolution, args.resolution)
        original_sizes = list(map(list, zip(*original_sizes)))
        crops_coords_top_left = list(map(list, zip(*crop_coords)))

        original_sizes = torch.tensor(original_sizes, dtype=torch.long)
        crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    def compute_image_embeddings(image_batch, image_encoder):
        image_embeds = encode_images(image_batch, image_encoder)
        return {"image_embeds": image_embeds}

    dataset = SDXLText2ImageDataset(
        train_shards_path_or_url=args.train_shards_path_or_url,
        num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,
        resolution=args.resolution,
        shuffle_buffer_size=1000,
        pin_memory=True,
        persistent_workers=True,
        use_fix_crop_and_size=args.use_fix_crop_and_size,
        use_image_conditioning=args.use_image_conditioning,
        cond_resolution=args.cond_resolution,
        cond_interpolation_type=args.cond_interpolation_type,
    )
    train_dataloader = dataset.train_dataloader

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
    )

    if args.use_image_conditioning:
        compute_image_embeddings_fn = functools.partial(
            compute_image_embeddings,
            image_encoder=image_encoder,
        )

    # 14. Create learning rate scheduler for generator and discriminator
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    discriminator_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=discriminator_optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    (
        unet,
        discriminator,
        optimizer,
        discriminator_optimizer,
        lr_scheduler,
        discriminator_lr_scheduler,
    ) = accelerator.prepare(
        unet,
        discriminator,
        optimizer,
        discriminator_optimizer,
        lr_scheduler,
        discriminator_lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # 16. Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Load and process the image, text, and micro-conditioning (original image size, crop coordinates)
                if args.use_image_conditioning:
                    image, text, orig_size, crop_coords, cond_image = batch
                else:
                    image, text, orig_size, crop_coords = batch

                image = image.to(accelerator.device, non_blocking=True)
                encoded_text = compute_embeddings_fn(text, orig_size, crop_coords)
                if args.use_image_conditioning:
                    encoded_image = compute_image_embeddings_fn(cond_image)

                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = image.to(dtype=weight_dtype)
                    if vae.dtype != weight_dtype:
                        vae.to(dtype=weight_dtype)
                else:
                    pixel_values = image

                # encode pixel values with batch size of at most 8
                latents = []
                for i in range(0, pixel_values.shape[0], 8):
                    latents.append(vae.encode(pixel_values[i : i + 8]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)

                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)
                bsz = latents.shape[0]

                # 2. Sample random student timesteps s uniformly in `student_timestep_schedule` and sample random
                # teacher timesteps t uniformly in [0, ..., noise_scheduler.config.num_train_timesteps - 1].
                student_index = torch.randint(0, student_distillation_steps, (bsz,), device=latents.device).long()
                student_timesteps = student_timestep_schedule[student_index]
                teacher_timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # 3. Sample noise and add it to the latents according to the noise magnitude at each student timestep
                # (that is, run the forward process on the student model)
                student_noise = torch.randn_like(latents)
                noisy_student_input = noise_scheduler.add_noise(latents, student_noise, student_timesteps)

                # 4. Prepare prompt embeds (for teacher/student U-Net) and text embedding (for discriminator).
                prompt_embeds = encoded_text.pop("prompt_embeds")
                text_embedding = encoded_text["text_embeds"]
                image_embedding = None
                if args.use_image_conditioning:
                    image_embedding = encoded_image.pop("image_embeds")
                    # Only supply image conditioning when student timestep is not last training timestep T.
                    image_embedding = torch.where(
                        student_timesteps.unsqueeze(1) < noise_scheduler.config.num_train_timesteps - 1,
                        image_embedding,
                        torch.zeros_like(image_embedding),
                    )

                # 5. Get the student model predicted original sample `student_x_0`.
                student_noise_pred = unet(
                    noisy_student_input,
                    student_timesteps,
                    encoder_hidden_states=prompt_embeds.float(),
                    added_cond_kwargs=encoded_text,
                ).sample
                student_x_0 = denoiser(student_noise_pred, student_timesteps, noisy_student_input)

                # 6. Sample noise and add it to the student's predicted original sample according to the noise
                # magnitude at each teacher timestep (that is, run the forward process on the teacher model, but
                # using `student_x_0` instead of latents sampled from the prior).
                teacher_noise = torch.randn_like(student_x_0)
                noisy_teacher_input = noise_scheduler.add_noise(student_x_0, teacher_noise, teacher_timesteps)

                # 7. Sample random guidance scales w ~ U[w_min, w_max] for CFG.
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w = w.reshape(bsz, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)

                # 8. Get teacher model predicted original sample `teacher_x_0`.
                with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
                    teacher_cond_noise_pred = teacher_unet(
                        noisy_teacher_input.detach(),
                        teacher_timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=encoded_text,
                    ).sample

                    uncond_prompt_embeds = torch.zeros_like(prompt_embeds)
                    uncond_pooled_prompt_embeds = torch.zeros_like(encoded_text["text_embeds"])
                    uncond_added_conditions = copy.deepcopy(encoded_text)
                    uncond_added_conditions["text_embeds"] = uncond_pooled_prompt_embeds
                    teacher_uncond_noise_pred = teacher_unet(
                        noisy_teacher_input.detach(),
                        teacher_timesteps,
                        encoder_hidden_states=uncond_prompt_embeds,
                        added_cond_kwargs=uncond_added_conditions,
                    ).sample

                    # Get the teacher's CFG estimate of x_0.
                    teacher_cfg_noise_pred = w * teacher_cond_noise_pred + (1 - w) * teacher_uncond_noise_pred
                    teacher_x_0 = denoiser(teacher_cfg_noise_pred, teacher_timesteps, noisy_teacher_input)

                ############################
                # 9. Discriminator Loss
                ############################
                discriminator_optimizer.zero_grad(set_to_none=True)

                # 1. Decode real and fake (generated) latents back to pixel space.
                # NOTE: the paper doesn't mention this explicitly AFAIK but I think this makes sense since the
                # pretrained feature network for the discriminator operates in pixel space rather than latent space.
                unscaled_student_x_0 = (1 / vae.config.scaling_factor) * student_x_0
                if args.pretrained_vae_model_name_or_path is not None:
                    student_gen_image = vae.decode(unscaled_student_x_0.to(dtype=weight_dtype)).sample
                else:
                    # VAE is in full precision due to possible NaN issues
                    student_gen_image = vae.decode(unscaled_student_x_0).sample

                # 2. Get discriminator real/fake outputs on the real and fake (generated) images respectively.
                disc_output_real = discriminator(pixel_values.float(), text_embedding, image_embedding)
                disc_output_fake = discriminator(student_gen_image.detach().float(), text_embedding, image_embedding)

                # 3. Calculate the discriminator real adversarial loss terms.
                d_logits_real = disc_output_real.logits
                # Use hinge loss (see section 3.2, Equation 3 of paper)
                d_adv_loss_real = torch.mean(F.relu(torch.ones_like(d_logits_real) - d_logits_real))

                # 4. Calculate the discriminator R1 gradient penalty term with respect to the gradients from the real
                # data.
                d_r1_regularizer = 0
                for k, head in discriminator.heads.items():
                    head_grad_params = torch.autograd.grad(
                        outputs=d_adv_loss_real, inputs=head.parameters(), create_graph=True
                    )
                    head_grad_norm = 0
                    for grad in head_grad_params:
                        head_grad_norm += grad.abs().sum()
                    d_r1_regularizer += head_grad_norm

                d_loss_real = d_adv_loss_real + args.discriminator_r1_strength * d_r1_regularizer
                accelerator.backward(d_loss_real, retain_graph=True)

                # 5. Calculate the discriminator fake adversarial loss terms.
                d_logits_fake = disc_output_fake.logits
                # Use hinge loss (see section 3.2, Equation 3 of paper)
                d_adv_loss_fake = torch.mean(F.relu(torch.ones_like(d_logits_fake) + d_logits_fake))
                accelerator.backward(d_adv_loss_fake)

                d_total_loss = d_loss_real + d_adv_loss_fake

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()

                ############################
                # 10. Generator Loss
                ############################
                optimizer.zero_grad(set_to_none=True)

                # 1. Rerun the disc on generated image, but this time allow gradients to flow through the generator
                disc_output_fake = discriminator(student_gen_image, text_embedding, image_embedding)

                # 2. Calculate generator adversarial loss term
                g_logits_fake = disc_output_fake.logits
                g_adv_loss = torch.mean(-g_logits_fake)

                ############################
                # 11. Distillation Loss
                ############################
                # Calculate distillation loss in pixel space rather than latent space (see section 3.1)
                unscaled_teacher_x_0 = (1 / vae.config.scaling_factor) * teacher_x_0
                if args.pretrained_vae_model_name_or_path is not None:
                    teacher_gen_image = vae.decode(unscaled_teacher_x_0.to(dtype=weight_dtype)).sample
                else:
                    # VAE is in full precision due to possible NaN issues
                    teacher_gen_image = vae.decode(unscaled_teacher_x_0).sample

                per_instance_distillation_loss = F.mse_loss(
                    student_gen_image.float(), teacher_gen_image.float(), reduction="none"
                )
                # Note that we use the teacher timesteps t when getting the loss weights.
                c_t = extract_into_tensor(
                    train_weight_schedule, teacher_timesteps, per_instance_distillation_loss.shape
                )
                g_distillation_loss = torch.mean(c_t * per_instance_distillation_loss)

                g_total_loss = g_adv_loss + args.distillation_weight_factor * g_distillation_loss

                # Backprop on the generator total loss
                accelerator.backward(g_total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 12. Perform an EMA update on the EMA version of the student U-Net weights.
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        if args.use_ema:
                            # Store the student unet weights and load the EMA weights.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                            log_validation(vae, unet, args, accelerator, weight_dtype, global_step, "ema_student")

                            # Restore student unet weights
                            ema_unet.restore(unet.parameters())

                        log_validation(vae, unet, args, accelerator, weight_dtype, global_step, "student")

            logs = {
                "d_total_loss": d_total_loss.detach().item(),
                "g_total_loss": g_total_loss.detach().item(),
                "g_adv_loss": g_adv_loss.detach().item(),
                "g_distill_loss": g_distillation_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            # Write out additional values for accelerator to report.
            logs["d_adv_loss_fake"] = d_adv_loss_fake.detach().item()
            logs["d_adv_loss_real"] = d_adv_loss_real.detach().item()
            logs["d_r1_regularizer"] = d_r1_regularizer.detach().item()
            logs["d_loss_real"] = d_loss_real.detach().item()
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))

        # If using EMA, save EMA weights as well.
        if args.use_ema:
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())

            unet.save_pretrained(os.path.join(args.output_dir, "ema_unet"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
