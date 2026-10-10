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
class TripoSplatGaussianDecoderOutput(BaseOutput):
    """Gaussian parameters shaped `(batch, num_gaussians, 14)`.

    Args:
        sample (`torch.Tensor`):
            Columns contain xyz position, degree-zero SH color, positive scale, wxyz rotation, and opacity.
    """

    sample: torch.Tensor


class TripoSplatDecoderAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "TripoSplatDecoderAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            query, key, value = attn.to_qkv(hidden_states).unflatten(-1, (3, attn.num_heads, attn.head_dim)).unbind(2)
        else:
            query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
            key, value = attn.to_kv(encoder_hidden_states).unflatten(-1, (2, attn.num_heads, attn.head_dim)).unbind(2)
        query = attn.q_norm(query)
        key = attn.k_norm(key)
        hidden_states = dispatch_attention_fn(
            query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        return attn.to_out(hidden_states.flatten(-2))


class TripoSplatDecoderAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = TripoSplatDecoderAttnProcessor
    _available_processors = [TripoSplatDecoderAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, channels: int, num_heads: int, ctx_channels: int | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        if ctx_channels is None:
            self.to_qkv = nn.Linear(channels, 3 * channels, bias=True)
        else:
            self.to_q = nn.Linear(channels, channels, bias=True)
            self.to_kv = nn.Linear(ctx_channels, 2 * channels, bias=True)
        self.q_norm = TripoSplatDecoderMultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = TripoSplatDecoderMultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)
        self.set_processor(self._default_processor_cls())

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states)


class TripoSplatDecoderMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return (F.normalize(x.float(), dim=-1) * self.gamma.float() * self.scale).to(origin_dtype)


class TripoSplatDecoderPcdAbsolutePositionEmbedderV2(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10) -> None:
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        N, D = x.shape
        logs = torch.linspace(0.0, float(self.max_res), steps=self.freq_dim, device=x.device, dtype=x.dtype)
        frequencies = torch.pow(2.0, logs)
        ang = x.unsqueeze(-1) * frequencies * torch.pi
        embed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device, dtype=embed.dtype)], dim=-1
            )
        return embed.to(orig_dtype)


class TripoSplatDecoderLevelEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequencies = torch.exp(-math.log(1024) * torch.arange(128, dtype=torch.float32, device="cpu") / 128)

    def forward(self, levels: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        angles = levels[:, None].float() * self.frequencies.to(levels.device)[None] * 2 * torch.pi
        embeddings = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.mlp(embeddings.to(dtype))


class TripoSplatDecoderModulatedTransformerCrossOnlyBlock(nn.Module):
    def __init__(self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(channels, elementwise_affine=False, eps=1e-06)
        self.norm2 = FP32LayerNorm(channels, elementwise_affine=False, eps=1e-06)
        self.cross_attn = TripoSplatDecoderAttention(channels, ctx_channels=ctx_channels, num_heads=num_heads)
        self.mlp = FeedForward(channels, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.cross_attn(h, context) * gate_msa.unsqueeze(1)
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(h) * gate_mlp.unsqueeze(1)
        return x


class TripoSplatDecoderOctreeProbabilityFixedlenDecoder(nn.Module):
    def __init__(
        self,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        max_voxel_level: int,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.max_voxel_level = max_voxel_level
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.l_embedder = TripoSplatDecoderLevelEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))
        self.blocks = nn.ModuleList(
            [
                TripoSplatDecoderModulatedTransformerCrossOnlyBlock(
                    model_channels, ctx_channels=cond_channels, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(model_channels, 8)
        self.in_proj = nn.Linear(3, model_channels)
        self.pos_embedder = TripoSplatDecoderPcdAbsolutePositionEmbedderV2(channels=model_channels, in_channels=3)

    def forward(
        self, cond: torch.Tensor, num_points: int, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        B = cond.shape[0]
        device = cond.device
        child_offset = torch.tensor(
            [[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]], dtype=torch.long, device=device
        )
        prev_coords_int = torch.zeros(B, 1, 3, dtype=torch.long, device=device)
        prev_counts = torch.full((B, 1), num_points, dtype=torch.long, device=device)
        batch_indices_range = torch.arange(B, device=device).unsqueeze(1)
        for lv in range(1, self.max_voxel_level + 1):
            res_p = 1 << lv - 1
            res = 1 << lv
            parent_coords_norm = (prev_coords_int.float() + 0.5) / res_p
            res_tensor = torch.full((B,), res, dtype=torch.long, device=device)
            hidden_states = self.in_proj(parent_coords_norm.to(cond.dtype))
            position_embeds = self.pos_embedder(parent_coords_norm.reshape(-1, 3)).reshape(B, -1, self.model_channels)
            hidden_states = hidden_states + position_embeds.to(device=hidden_states.device, dtype=cond.dtype)
            hidden_states = self.input_layer(hidden_states)
            level_embeds = self.l_embedder(res_tensor, hidden_states.dtype)
            modulation = self.adaLN_modulation(level_embeds)
            for block in self.blocks:
                hidden_states = block(hidden_states, modulation, cond)
            hidden_states = F.layer_norm(hidden_states.float(), hidden_states.shape[-1:]).to(cond.dtype)
            pred_logits = self.out_proj(hidden_states)
            pred_probs = torch.softmax(pred_logits, dim=-1).to(device)
            counts = prev_counts.flatten().to(device=device, dtype=torch.long)
            probabilities = pred_probs.reshape(counts.numel(), -1).float().clamp_min_(0)
            probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min_(1)
            sampled = torch.zeros_like(probabilities, dtype=torch.long)
            cdf = probabilities.cumsum(dim=1).clamp(max=1.0 - 1e-12)
            unique_counts, inverse_indices = counts.unique(sorted=False, return_inverse=True)
            random_device = device if generator is None else generator.device
            for index, count in enumerate(unique_counts.tolist()):
                if count == 0:
                    continue
                rows = (inverse_indices == index).nonzero(as_tuple=False).squeeze(1)
                start = torch.rand(
                    (rows.numel(), 1), generator=generator, device=random_device, dtype=probabilities.dtype
                ).to(device) / float(count)
                grid = torch.arange(count, device=device, dtype=probabilities.dtype)[None, :] / float(count)
                samples = (start + grid).clamp(max=1.0 - 1e-12)
                child_indices = torch.searchsorted(cdf.index_select(0, rows), samples).clamp_max(
                    probabilities.shape[1] - 1
                )
                child_counts = torch.zeros(
                    rows.numel(), probabilities.shape[1], dtype=probabilities.dtype, device=device
                )
                child_counts.scatter_add_(1, child_indices, torch.ones_like(child_indices, dtype=child_counts.dtype))
                sampled.index_copy_(0, rows, child_counts.to(torch.long))
            sampled = sampled.reshape(*prev_counts.shape, -1).flatten(1, 2)
            child_coords_int = (prev_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :]).flatten(1, 2)
            mask = sampled > 0
            max_valid = mask.sum(dim=1).max().item()
            scatter_indices = mask.cumsum(dim=1) - 1
            valid_scatter_indices = scatter_indices[mask]
            valid_batch_indices = batch_indices_range.expand_as(mask)[mask]
            next_prev_coords_int = torch.zeros(B, max_valid, 3, dtype=child_coords_int.dtype, device=device)
            next_prev_coords_int[valid_batch_indices, valid_scatter_indices] = child_coords_int[mask]
            next_prev_counts = torch.zeros(B, max_valid, dtype=sampled.dtype, device=device)
            next_prev_counts[valid_batch_indices, valid_scatter_indices] = sampled[mask]
            prev_coords_int = next_prev_coords_int
            prev_counts = next_prev_counts
        res = 1 << self.max_voxel_level
        coords_int = torch.repeat_interleave(prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), dim=0).reshape(
            B, num_points, -1
        )
        coords_norm = coords_int.float()
        random_device = device if generator is None else generator.device
        jitter = torch.rand(coords_norm.shape, generator=generator, device=random_device, dtype=coords_norm.dtype).to(
            device
        )
        coords_norm = (coords_norm + jitter) / res
        return {"points": coords_norm}


class TripoSplatDecoderTransformerCrossBlock(nn.Module):
    def __init__(self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(channels, elementwise_affine=False, eps=1e-06)
        self.norm2 = FP32LayerNorm(channels, elementwise_affine=True, eps=1e-06)
        self.norm3 = FP32LayerNorm(channels, elementwise_affine=False, eps=1e-06)
        self.self_attn = TripoSplatDecoderAttention(channels, num_heads=num_heads)
        self.cross_attn = TripoSplatDecoderAttention(channels, ctx_channels=ctx_channels, num_heads=num_heads)
        self.mlp = FeedForward(channels, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class TripoSplatDecoderElasticGaussianFixedlenDecoder(nn.Module):
    def __init__(
        self,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        gaussians_per_point: int,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.layout = {
            "_xyz": {"shape": (gaussians_per_point, 3), "size": gaussians_per_point * 3},
            "_features_dc": {"shape": (gaussians_per_point, 1, 3), "size": gaussians_per_point * 3},
            "_scaling": {"shape": (gaussians_per_point, 3), "size": gaussians_per_point * 3},
            "_rotation": {"shape": (gaussians_per_point, 4), "size": gaussians_per_point * 4},
            "_opacity": {"shape": (gaussians_per_point, 1), "size": gaussians_per_point},
            "_offset_scale": {"shape": (gaussians_per_point, 1), "size": gaussians_per_point},
        }
        out_channels = 0
        for layout in self.layout.values():
            layout["range"] = (out_channels, out_channels + layout["size"])
            out_channels += layout["size"]
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                TripoSplatDecoderTransformerCrossBlock(
                    model_channels, ctx_channels=cond_channels, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(num_blocks)
            ]
        )
        self.in_proj = nn.Linear(3, model_channels)
        self.pos_embedder = TripoSplatDecoderPcdAbsolutePositionEmbedderV2(channels=model_channels, in_channels=3)
        self.out_proj = nn.Linear(model_channels, out_channels)
        perturbation = []
        for index in range(gaussians_per_point):
            point = [index / gaussians_per_point]
            for base in (2, 3):
                value = 0.0
                inverse_base = 1.0 / base
                factor = inverse_base
                remaining = index
                while remaining > 0:
                    value += (remaining % base) * factor
                    remaining //= base
                    factor *= inverse_base
                point.append(value)
            perturbation.append(point)
        perturbation = torch.tensor(perturbation).float()
        perturbation = torch.atanh((perturbation * 2 - 1) / 1.5)
        self.register_buffer("points_offset_perturbation", perturbation)
        base = torch.tensor(0.05)
        self.register_buffer("base_offset_scale", torch.log(torch.exp(base) - 1.0))

    def forward(self, points: torch.Tensor, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, length, _ = points.shape
        hidden_states = self.in_proj(points.to(cond.dtype))
        position_embeds = self.pos_embedder(points.reshape(-1, 3)).reshape(batch, length, -1)
        hidden_states = hidden_states + position_embeds.to(device=hidden_states.device, dtype=cond.dtype)
        hidden_states = self.input_layer(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, cond)
        hidden_states = F.layer_norm(hidden_states.float(), hidden_states.shape[-1:]).to(cond.dtype)
        features = self.out_proj(hidden_states)
        start, end = self.layout["_offset_scale"]["range"]
        offset_scale = F.softplus(
            features[:, :, start:end].reshape(batch, -1, *self.layout["_offset_scale"]["shape"])
            + self.base_offset_scale.to(features.device)
        )
        start, end = self.layout["_xyz"]["range"]
        offset = features[:, :, start:end].reshape(batch, -1, *self.layout["_xyz"]["shape"])
        offset = offset + self.points_offset_perturbation.to(features.device)
        offset = torch.tanh(offset) * 0.5 * 1.5
        offset = offset * offset_scale
        return {"features": features, "offset": offset}


class TripoSplatGaussianDecoder(ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin):
    """Decode Gaussian latents by sampling an octree and predicting splat parameters.

    Parameters:
        model_channels (`int`, defaults to `1024`):
            Decoder hidden width.
        cond_channels (`int`, defaults to `16`):
            Width of the denoised Gaussian latents.
        num_octree_blocks (`int`, defaults to `4`):
            Number of octree probability transformer blocks.
        num_gaussian_blocks (`int`, defaults to `16`):
            Number of Gaussian parameter transformer blocks.
        num_heads (`int`, defaults to `16`):
            Number of attention heads.
        mlp_ratio (`float`, defaults to `4.0`):
            Feed-forward hidden width relative to `model_channels`.
        gaussians_per_point (`int`, defaults to `32`):
            Number of Gaussians predicted for each sampled point.
        max_voxel_level (`int`, defaults to `8`):
            Number of octree subdivision levels.
    """

    _skip_layerwise_casting_patterns = ["norm", "input_layer", "in_proj", "l_embedder"]
    _no_split_modules = [
        "TripoSplatDecoderModulatedTransformerCrossOnlyBlock",
        "TripoSplatDecoderTransformerCrossBlock",
    ]
    _repeated_blocks = [
        "TripoSplatDecoderModulatedTransformerCrossOnlyBlock",
        "TripoSplatDecoderTransformerCrossBlock",
    ]

    @register_to_config
    def __init__(
        self,
        model_channels: int = 1024,
        cond_channels: int = 16,
        num_octree_blocks: int = 4,
        num_gaussian_blocks: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        gaussians_per_point: int = 32,
        max_voxel_level: int = 8,
    ) -> None:
        super().__init__()
        self.octree = TripoSplatDecoderOctreeProbabilityFixedlenDecoder(
            model_channels=model_channels,
            cond_channels=cond_channels,
            num_blocks=num_octree_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_voxel_level=max_voxel_level,
        )
        self.gs = TripoSplatDecoderElasticGaussianFixedlenDecoder(
            model_channels=model_channels,
            cond_channels=cond_channels,
            num_blocks=num_gaussian_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            gaussians_per_point=gaussians_per_point,
        )
        scaling_bias = torch.tensor(0.004, dtype=torch.float32, device="cpu")
        self.scaling_bias = scaling_bias + torch.log(-torch.expm1(-scaling_bias))
        opacity_bias = torch.tensor(0.1, dtype=torch.float32, device="cpu")
        self.opacity_bias = torch.log(opacity_bias / (1 - opacity_bias))
        self.identity_rotation = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device="cpu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_gaussians: int = 262144,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
    ) -> TripoSplatGaussianDecoderOutput | tuple[torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor`):
                Denoised Gaussian latents of shape `(batch, latent_tokens, cond_channels)`.
            num_gaussians (`int`, defaults to `262144`):
                Number of Gaussians to decode, divisible by `gaussians_per_point`.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator or one generator per batch item for octree sampling and point jitter.
            return_dict (`bool`, defaults to `True`):
                Whether to return a structured output or a tuple.

        Returns:
            `TripoSplatGaussianDecoderOutput` or `tuple`: Parameters of shape `(batch, num_gaussians, 14)` in xyz,
            degree-zero spherical harmonic color, scale, wxyz rotation, and opacity order.
        """
        if num_gaussians <= 0 or num_gaussians % self.config.gaussians_per_point:
            raise ValueError("num_gaussians must be a positive multiple of gaussians_per_point.")
        hidden_states = hidden_states.to(self.dtype)
        batch_size = hidden_states.shape[0]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Pass one generator per batch item.")
        points = []
        for index in range(batch_size):
            item_generator = generator[index] if isinstance(generator, list) else generator
            prediction = self.octree(
                hidden_states[index : index + 1],
                num_points=num_gaussians // self.config.gaussians_per_point,
                generator=item_generator,
            )
            points.append(prediction["points"])
        points = torch.cat(points, dim=0)
        prediction = self.gs(points, hidden_states)
        features = prediction["features"]
        positions = (points.to(features.device)[:, :, None] + prediction["offset"]).flatten(1, 2) - 0.5
        parameters = {}
        for name in ("_features_dc", "_scaling", "_rotation", "_opacity"):
            layout = self.gs.layout[name]
            start, end = layout["range"]
            parameters[name] = features[..., start:end].reshape(batch_size, -1, *layout["shape"]).flatten(1, 2)
        colors = parameters["_features_dc"].squeeze(2)
        scaling_bias = self.scaling_bias.to(features.device)
        scales = F.softplus(parameters["_scaling"] + scaling_bias)
        scales = (scales.square() + 0.0009**2).sqrt()
        rotations = parameters["_rotation"] * 0.1
        rotations = rotations + self.identity_rotation.to(features.device)
        opacity_bias = self.opacity_bias.to(features.device)
        opacity = torch.sigmoid(parameters["_opacity"] + opacity_bias)
        sample = torch.cat([positions, colors, scales, rotations, opacity], dim=-1)
        if not return_dict:
            return (sample,)
        return TripoSplatGaussianDecoderOutput(sample=sample)
