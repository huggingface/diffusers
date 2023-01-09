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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6) if layer_norm else nn.Identity()

    def forward(self, latent):
        _, _, height, width = latent.shape
        if height != self.img_size[0] or width != self.img_size[1]:
            ValueError(
                f"Input image size ({height}x{width}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
            )

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        latent = self.norm(latent)
        return latent


# class DiTBlock(nn.Module):
#     """
#     A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """

#     def __init__(self, hidden_size, num_heads, mlp_ratio=4):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.attn = CrossAttention(
#             query_dim=hidden_size, heads=num_heads, dim_head=hidden_size // num_heads, bias=True
#         )
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.mlp = FeedForward(
#             dim=hidden_size,
#             mult=mlp_ratio,
#             activation_fn="gelu-approximate",
#             final_dropout=True,
#         )
#         self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

#     def forward(self, latent, cls):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cls).chunk(6, dim=1)
#         latent = latent + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(latent), shift_msa, scale_msa))
#         latent = latent + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latent), shift_mlp, scale_mlp))
#         return latent


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, latent, cls):
        shift, scale = self.adaLN_modulation(cls).chunk(2, dim=1)
        latent = modulate(self.norm_final(latent), shift, scale)
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

        self.sample_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        num_patches = self.sample_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                hidden_size, 
                num_heads,
                attention_head_dim=hidden_size // num_heads,
                activation_fn="gelu-approximate",
                num_embeds_ada_norm=num_classes,
                attention_bias=True,
                use_ada_layer_norm_zero=True,
                norm_elementwise_affine=False,
                final_dropout=True,
            ) for _ in range(depth)
        ])
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

        # # Initialize label embedding table:
        # nn.init.normal_(self.class_embedder.embedding_table.weight, std=0.02)

        # # Initialize timestep embedding MLP:
        # nn.init.normal_(self.timestep_embedder.linear_1.weight, std=0.02)
        # nn.init.normal_(self.timestep_embedder.linear_2.weight, std=0.02)

        # # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, latent):
        """
        Args:
            latent: (N, T, patch_size**2 * C)

        Returns:
            imgs: (N, C, H, W)
        """
        chan = self.out_channels
        patch = self.sample_embedder.patch_size[0]
        height = width = int(latent.shape[1] ** 0.5)
        if height * width != latent.shape[1]:
            ValueError("Latent size does not match the number of patches")

        latent = latent.reshape(shape=(latent.shape[0], height, width, patch, patch, chan))
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        imgs = latent.reshape(shape=(latent.shape[0], chan, height * patch, width * patch))
        return imgs

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
        # timesteps_proj = self.time_proj(timesteps)
        # timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

        # class_labels = self.class_embedder(class_labels, self.training)  # (N, D)
        # conditioning = timesteps_emb + class_labels  # (N, D)

        sample = self.sample_embedder(sample) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        for block in self.blocks:
            sample = block(sample, timestep=timesteps, class_labels=class_labels)  # (N, T, D)
        conditioning = block.norm1.emb(timestep, class_labels)
        sample = self.final_layer(sample, conditioning)  # (N, T, patch_size ** 2 * out_channels)
        sample = self.unpatchify(sample)  # (N, out_channels, H, W)
        return sample
