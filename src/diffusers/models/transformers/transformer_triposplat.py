# Copyright 2026 The HuggingFace Team. All rights reserved.
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


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import BaseOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


@dataclass
class TripoSplatTransformer3DModelOutput(BaseOutput):
    """Joint Gaussian and camera velocity predictions.

    Args:
        sample (`torch.Tensor`):
            Gaussian velocity shaped `(batch, query_tokens, in_channels)`.
        camera_sample (`torch.Tensor`, *optional*):
            Camera velocity shaped `(batch, 1, cam_channels)`.
    """

    sample: torch.Tensor
    camera_sample: torch.Tensor | None = None


class TripoSplatRMSNorm(nn.Module):
    def __init__(self, head_dim: int, num_heads: int) -> None:
        super().__init__()
        self.scale = head_dim**0.5
        self.gamma = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (F.normalize(hidden_states.float(), dim=-1) * self.gamma.float() * self.scale).to(hidden_states.dtype)


class TripoSplatAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self, attn: "TripoSplatAttention", hidden_states: torch.Tensor, rotary_emb: torch.Tensor
    ) -> torch.Tensor:
        query, key, value = attn.to_qkv(hidden_states).unflatten(-1, (3, attn.num_heads, attn.head_dim)).unbind(2)
        # TripoSplat predicts separate complex rotation phases for each token and attention head.
        query = torch.view_as_complex(query.float().unflatten(-1, (-1, 2))) * rotary_emb
        key = torch.view_as_complex(key.float().unflatten(-1, (-1, 2))) * rotary_emb
        query = torch.view_as_real(query).flatten(-2).to(hidden_states.dtype)
        key = torch.view_as_real(key).flatten(-2).to(hidden_states.dtype)
        query = attn.q_norm(query)
        key = attn.k_norm(key)
        hidden_states = dispatch_attention_fn(
            query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        return attn.to_out(hidden_states.flatten(-2))


class TripoSplatAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = TripoSplatAttnProcessor
    _available_processors = [TripoSplatAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.to_qkv = nn.Linear(channels, 3 * channels)
        self.q_norm = TripoSplatRMSNorm(self.head_dim, num_heads)
        self.k_norm = TripoSplatRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)
        self.set_processor(self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb)


class TripoSplatRotaryEmbedding(nn.Module):
    def __init__(self, model_channels: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        hidden_size = int(model_channels * 0.125)
        self.norm = FP32LayerNorm(model_channels)
        self.gate_map = nn.Linear(model_channels, hidden_size, bias=False)
        self.content_map = nn.Linear(model_channels, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.final_map = nn.Linear(hidden_size, 3 * num_heads, bias=False)
        dims = [2 * (head_dim // 6), 2 * (head_dim // 6), head_dim - 4 * (head_dim // 6)]
        self.freqs_0 = nn.Parameter(torch.linspace(1.0, 16.0, dims[0] // 2))
        self.freqs_1 = nn.Parameter(torch.linspace(1.0, 16.0, dims[1] // 2))
        self.freqs_2 = nn.Parameter(torch.linspace(1.0, 16.0, dims[2] // 2))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        features = self.act(self.gate_map(hidden_states)) * self.content_map(hidden_states)
        positions = self.final_map(features).unflatten(-1, (self.num_heads, 3))
        angles = []
        for axis, frequencies in enumerate((self.freqs_0, self.freqs_1, self.freqs_2)):
            position = positions[..., axis, None]
            frequency_tanh = frequencies.tanh()
            angle = position * frequency_tanh + position.detach() * (frequencies - frequency_tanh)
            angles.append(angle * torch.pi)
        angles = torch.cat(angles, dim=-1).float()
        return torch.polar(torch.ones_like(angles), angles)


class TripoSplatPositionEmbedder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.freq_dim = channels // 3 // 2

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        frequencies = torch.arange(16, dtype=positions.dtype, device=positions.device)
        residual_dim = max(0, self.freq_dim - 16)
        residual = (
            torch.arange(residual_dim, dtype=positions.dtype, device=positions.device) / max(residual_dim, 1) * 16
        )
        frequencies = torch.pow(2.0, torch.cat([frequencies, residual])[: self.freq_dim])
        angles = torch.outer(positions.float().reshape(-1), frequencies) * 2 * torch.pi
        embeddings = torch.cat([angles.sin(), angles.cos()], dim=-1).reshape(*positions.shape[:-1], -1)
        return F.pad(embeddings, (0, self.channels - embeddings.shape[-1])).to(positions.dtype)


class TripoSplatTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, mlp_ratio: float, modulation: bool) -> None:
        super().__init__()
        self.modulation = modulation
        self.norm1 = FP32LayerNorm(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = FP32LayerNorm(channels, elementwise_affine=not modulation, eps=1e-6)
        self.attn = TripoSplatAttention(channels, num_heads)
        self.mlp = FeedForward(channels, mult=mlp_ratio, activation_fn="gelu-approximate")
        if modulation:
            self.shift_table = nn.Parameter(torch.randn(1, 6 * channels) / channels**0.5)

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: torch.Tensor, modulation: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.modulation:
            modulation = modulation + self.shift_table.to(modulation.dtype)
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
            residual = self.norm1(hidden_states) * (1 + scale_attn[:, None]) + shift_attn[:, None]
            hidden_states = hidden_states + self.attn(residual, rotary_emb) * gate_attn[:, None]
            residual = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            hidden_states = hidden_states + self.mlp(residual) * gate_mlp[:, None]
        else:
            hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb)
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class TripoSplatTimestepEmbedder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device="cpu") / 128)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        frequencies = self.frequencies.to(timestep.device)
        angles = timestep[:, None].float() * frequencies[None]
        embeddings = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.mlp(embeddings.to(dtype))


class TripoSplatTransformer3DModel(ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin):
    """Image-conditioned flow transformer for Gaussian and camera latents.

    Parameters:
        q_token_length (`int`, defaults to `8192`):
            Number of Gaussian latent tokens.
        in_channels (`int`, defaults to `16`):
            Number of channels in each Gaussian latent token.
        cam_channels (`int`, defaults to `5`):
            Number of camera latent channels.
        out_channels (`int`, defaults to `16`):
            Number of predicted Gaussian velocity channels.
        model_channels (`int`, defaults to `1024`):
            Transformer hidden width.
        cond_channels (`int`, defaults to `1280`):
            DINOv3 feature width.
        cond2_channels (`int`, defaults to `128`):
            Packed image VAE feature width.
        num_refiner_blocks (`int`, defaults to `2`):
            Number of Gaussian and image conditioning refinement blocks.
        num_blocks (`int`, defaults to `24`):
            Number of joint transformer blocks.
        num_heads (`int`, defaults to `16`):
            Number of attention heads.
        mlp_ratio (`float`, defaults to `4.0`):
            Feed-forward hidden width relative to `model_channels`.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TripoSplatTransformerBlock", "TripoSplatRotaryEmbedding"]
    _repeated_blocks = ["TripoSplatTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm", "embedder", "input_layer", "shift_table"]

    @register_to_config
    def __init__(
        self,
        q_token_length: int = 8192,
        in_channels: int = 16,
        cam_channels: int = 5,
        out_channels: int = 16,
        model_channels: int = 1024,
        cond_channels: int = 1280,
        cond2_channels: int = 128,
        num_refiner_blocks: int = 2,
        num_blocks: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.t_embedder = TripoSplatTimestepEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels)
        # The checkpoint uses a fixed scrambled Sobol lattice for its query positions.
        sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123)
        positions = sobol.draw(q_token_length, dtype=torch.float32)
        # Sobol's first point uses the default dtype, which can be float16 during checkpoint loading.
        positions[0] = sobol.shift.to(torch.float32) / 2**sobol.MAXBIT
        self.pos_pe = positions.unsqueeze(0)
        self.pos_embedder = TripoSplatPositionEmbedder(model_channels)
        self.noise_repo_layers = nn.ModuleList(
            [
                TripoSplatRotaryEmbedding(model_channels, num_heads, model_channels // num_heads)
                for _ in range(num_refiner_blocks)
            ]
        )
        self.context_repo_layers = nn.ModuleList(
            [
                TripoSplatRotaryEmbedding(model_channels, num_heads, model_channels // num_heads)
                for _ in range(num_refiner_blocks)
            ]
        )
        self.repo_layers = nn.ModuleList(
            [
                TripoSplatRotaryEmbedding(model_channels, num_heads, model_channels // num_heads)
                for _ in range(num_blocks)
            ]
        )
        self.noise_refiner = nn.ModuleList(
            [TripoSplatTransformerBlock(model_channels, num_heads, mlp_ratio, True) for _ in range(num_refiner_blocks)]
        )
        self.context_refiner = nn.ModuleList(
            [
                TripoSplatTransformerBlock(model_channels, num_heads, mlp_ratio, False)
                for _ in range(num_refiner_blocks)
            ]
        )
        camera_layers = []
        for index in range(num_refiner_blocks - 1):
            camera_layers.extend(
                [
                    nn.Linear(cam_channels if index == 0 else model_channels, model_channels),
                    nn.GELU(approximate="tanh"),
                ]
            )
        camera_layers.append(nn.Linear(cam_channels if num_refiner_blocks == 1 else model_channels, model_channels))
        self.cam_refiner = nn.Sequential(*camera_layers)
        self.blocks = nn.ModuleList(
            [TripoSplatTransformerBlock(model_channels, num_heads, mlp_ratio, True) for _ in range(num_blocks)]
        )
        self.shift_table = nn.Parameter(torch.randn(1, 2, model_channels) / model_channels**0.5)
        self.out_layer = nn.Linear(model_channels, out_channels)
        self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_latents: torch.Tensor,
        camera_latents: torch.Tensor,
        return_dict: bool = True,
    ) -> TripoSplatTransformer3DModelOutput | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor`):
                Gaussian latents of shape `(batch, q_token_length, in_channels)`.
            timestep (`torch.Tensor`):
                Timesteps of shape `(batch,)`, scaled to the scheduler's training timestep range.
            encoder_hidden_states (`torch.Tensor`):
                Normalized DINOv3 features of shape `(batch, image_tokens, cond_channels)`.
            image_latents (`torch.Tensor`):
                Packed VAE features of shape `(batch, image_tokens, cond2_channels)`.
            camera_latents (`torch.Tensor`):
                Camera latents of shape `(batch, 1, cam_channels)`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a structured output or a tuple.

        Returns:
            `TripoSplatTransformer3DModelOutput` or `tuple`: Gaussian and camera velocity predictions.
        """
        latent_length = hidden_states.shape[1]
        hidden_states = self.input_layer(hidden_states.to(self.dtype))
        conditioning = self.cond_embedder(encoder_hidden_states.to(self.dtype))
        conditioning = conditioning + self.cond_embedder2(image_latents.to(self.dtype)).to(conditioning.device)
        timestep_emb = self.t_embedder(timestep, hidden_states.dtype)
        modulation = self.adaLN_modulation(timestep_emb)
        hidden_states = hidden_states + self.pos_embedder(self.pos_pe.to(hidden_states.device)).to(self.dtype)
        for block, rotary_layer in zip(self.noise_refiner, self.noise_repo_layers):
            rotary_emb = rotary_layer(hidden_states)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb, modulation)
            else:
                hidden_states = block(hidden_states, rotary_emb, modulation)
        for block, rotary_layer in zip(self.context_refiner, self.context_repo_layers):
            rotary_emb = rotary_layer(conditioning)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                conditioning = self._gradient_checkpointing_func(block, conditioning, rotary_emb)
            else:
                conditioning = block(conditioning, rotary_emb)
        camera = self.cam_refiner(camera_latents.to(self.dtype))
        hidden_states = torch.cat(
            [hidden_states, conditioning.to(hidden_states.device), camera.to(hidden_states.device)], dim=1
        )
        for block, rotary_layer in zip(self.blocks, self.repo_layers):
            rotary_emb = rotary_layer(hidden_states)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb, modulation)
            else:
                hidden_states = block(hidden_states, rotary_emb, modulation)
        camera = F.layer_norm(hidden_states[:, -camera.shape[1] :].float(), hidden_states.shape[-1:]).to(self.dtype)
        hidden_states = F.layer_norm(hidden_states[:, :latent_length].float(), hidden_states.shape[-1:]).to(self.dtype)
        shift, scale = (
            self.shift_table.to(hidden_states.device) + timestep_emb.to(hidden_states.device)[:, None]
        ).chunk(2, dim=1)
        hidden_states = self.out_layer(hidden_states * (1 + scale) + shift)
        camera = self.cam_out_layer(camera * (1 + scale) + shift).to(hidden_states.device)
        if not return_dict:
            return hidden_states, camera
        return TripoSplatTransformer3DModelOutput(sample=hidden_states, camera_sample=camera)
