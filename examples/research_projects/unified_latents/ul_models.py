#!/usr/bin/env python
# coding=utf-8

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel


class ULTwoStageBaseModel(ModelMixin, ConfigMixin):
    """
    Approximation of UL Sec. 5.1 stage-2 base model:
    - Stage A: ViT-like latent denoiser, width ~512, 6 layers
    - Stage B: ViT-like latent denoiser, width ~1024, 16 layers

    Both stages are DiT blocks and predict residual denoised latents.
    """

    @register_to_config
    def __init__(
        self,
        latent_channels: int,
        latent_size: int,
        num_train_timesteps: int,
        stage_a_layers: int = 6,
        stage_b_layers: int = 16,
        stage_a_heads: int = 8,
        stage_a_head_dim: int = 64,  # 512 width
        stage_b_heads: int = 16,
        stage_b_head_dim: int = 64,  # 1024 width
        patch_size: int = 1,
    ):
        super().__init__()
        self.stage_a = DiTTransformer2DModel(
            num_attention_heads=stage_a_heads,
            attention_head_dim=stage_a_head_dim,
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_layers=stage_a_layers,
            sample_size=latent_size,
            patch_size=patch_size,
            num_embeds_ada_norm=num_train_timesteps,
            dropout=0.1,
        )
        self.stage_b = DiTTransformer2DModel(
            num_attention_heads=stage_b_heads,
            attention_head_dim=stage_b_head_dim,
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_layers=stage_b_layers,
            sample_size=latent_size,
            patch_size=patch_size,
            num_embeds_ada_norm=num_train_timesteps,
            dropout=0.1,
        )

    def forward(self, z_t: torch.Tensor, timesteps: torch.LongTensor, class_labels: torch.LongTensor) -> torch.Tensor:
        h = self.stage_a(z_t, timestep=timesteps, class_labels=class_labels).sample
        out = self.stage_b(h, timestep=timesteps, class_labels=class_labels).sample
        return out
