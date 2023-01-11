# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under
# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import collections.abc
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from .attention import BasicTransformerBlock
from .embeddings import LabelEmbedding, TimestepEmbedding, Timesteps, get_2d_sincos_pos_embed
from .modeling_utils import ModelMixin


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, latent, cls):
        shift, scale = self.adaLN_modulation(cls).chunk(2, dim=1)
        latent = self.norm_final(latent) * (1 + scale[:, None]) + shift[:, None]
        latent = self.linear(latent)
        return latent


class DiT(ModelMixin, ConfigMixin):
    """
    Diffusion model with a Transformer backbone.
    """

    @register_to_config
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        # self.sample_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    hidden_size,
                    num_heads,
                    attention_head_dim=hidden_size // num_heads,
                    activation_fn="gelu-approximate",
                    num_embeds_ada_norm=num_classes,
                    attention_bias=True,
                    use_ada_layer_norm_zero=True,
                    norm_elementwise_affine=False,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.sample_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.sample_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # def unpatchify(self, latent):
    #     """
    # Args: # latent: (N, T, patch_size**2 * C)

    # Returns: # imgs: (N, C, H, W) #"""
    #     chan = self.out_channels
    #     patch = self.sample_embedder.patch_size[0]
    #     height = width = int(latent.shape[1] ** 0.5)
    #     if height * width != latent.shape[1]:
    #         ValueError("Latent size does not match the number of patches")

    #     latent = latent.reshape(shape=(latent.shape[0], height, width, patch, patch, chan))
    #     latent = torch.einsum("nhwpqc->nchpwq", latent)
    #     imgs = latent.reshape(shape=(latent.shape[0], chan, height * patch, width * patch))
    #     return imgs

    def forward(self, sample, timestep, class_labels):
        """
        Forward pass of DiT.

        Args:
            sample: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
            timestep: (N,) tensor of diffusion timesteps
            class_labels: (N,) tensor of class labels
        """

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        sample = self.sample_embedder(sample) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        for block in self.blocks:
            sample = block(sample, timestep=timesteps, class_labels=class_labels)  # (N, T, D)
        conditioning = block.norm1.emb(timesteps, class_labels)
        sample = self.final_layer(sample, conditioning)  # (N, T, patch_size ** 2 * out_channels)
        # Can be done in transformer 2d gated by patched flag
        sample = self.unpatchify(sample)  # (N, out_channels, H, W)
        return sample
