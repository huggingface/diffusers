# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""Kandinsky 6 video super-resolution transformer."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import _CAN_USE_FLEX_ATTN, AttentionBackendName, dispatch_attention_fn
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype


if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


logger = logging.get_logger(__name__)


# Side of the local token block that NABLA sparse attention groups into one 64-token attention block.
FRACTAL_BLOCK_SIZE = 8


# Copied from diffusers.models.transformers.transformer_kandinsky6.get_freqs
def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    """Return inverse frequencies for rotary position embeddings.

    Args:
        dim (`int`): Number of frequency values to generate.
        max_period (`float`, *optional*, defaults to 10000.0): Maximum period
            used by the frequency schedule.

    Returns:
        `torch.Tensor`: Frequency values in float32.
    """
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


# Copied from diffusers.models.transformers.transformer_kandinsky6.apply_scale_shift
def apply_scale_shift(normed: Tensor, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply an AdaLN-style scale/shift affine to an already-normalized tensor, in fp32, cast back to ``x.dtype``.

    Callers compute ``normed`` themselves (e.g. ``self.some_norm(x.float())``) so the norm-layer call stays visible in
    `forward` instead of being hidden inside this helper.
    """
    if x.ndim > 2 and scale.ndim == 2:
        shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[-1],)
        scale, shift = scale.reshape(shape), shift.reshape(shape)
    return (normed * (scale.float() + 1.0) + shift.float()).to(dtype=x.dtype)


# Copied from diffusers.models.transformers.transformer_kandinsky6.apply_gate_sum
def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor) -> Tensor:
    """Residual gate in fp32, cast back to ``x.dtype``."""
    if x.ndim > 2 and gate.ndim == 2:
        gate = gate.reshape((gate.shape[0],) + (1,) * (x.ndim - 2) + (gate.shape[-1],))
    return (x.float() + gate.float() * out.float()).to(dtype=x.dtype)


# Copied from diffusers.models.transformers.transformer_kandinsky6.apply_rotary
def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """RoPE apply in fp32 (rope tables are fp32), cast back to ``x.dtype``."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope.float() * x_).sum(dim=-1).reshape(*x.shape).to(dtype=x.dtype)


def _local_patch(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, g1, H // g2, g2, W // g3, g3, *x.shape[dim + 3 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 2, d + 4, d + 1, d + 3, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)


def _local_merge(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, H // g2, W // g3, g1, g2, g3, *x.shape[dim + 2 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 3, d + 1, d + 4, d + 2, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)


def nabla_block_mask(
    q: Tensor,
    k: Tensor,
    sta: Tensor,
    thr: float = 0.9,
    block_size: int = 64,
) -> BlockMask:
    """Build a dynamic NABLA block mask from query/key statistics and an STA (Sliding-Tile Attention, see
    https://huggingface.co/papers/2502.04507) prior."""
    from torch.nn.attention.flex_attention import BlockMask

    B, h, S, D = q.shape
    s1 = S // block_size
    qa = q.reshape(B, h, s1, block_size, D).mean(-2)
    ka = k.reshape(B, h, s1, block_size, D).mean(-2).transpose(-2, -1)
    attn_map = torch.softmax((qa @ ka) / math.sqrt(D), dim=-1)

    vals, inds = attn_map.sort(-1)
    mask = (vals.cumsum_(-1) >= 1 - thr).int().gather(-1, inds.argsort(-1))
    mask = torch.logical_or(mask, sta)

    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb),
        kv_inds,
        kv_nb,
        kv_inds,
        BLOCK_SIZE=block_size,
        mask_mod=None,
    )


def sliding_tile_mask(
    num_frames: int, height: int, width: int, window: tuple[int, int, int], device: torch.device
) -> Tensor:
    """Build the static sliding-tile-attention prior over the `(t, h, w)` grid of 8x8 token blocks.

    Every block attends to the blocks within `window` (odd `(t, h, w)` extents) of itself. Returned as a boolean
    `(num_frames * height * width, num_frames * height * width)` matrix in the flattened block order.
    """
    window_t, window_h, window_w = window
    positions = torch.arange(max(num_frames, height, width), device=device)
    distance = (positions[:, None] - positions[None, :]).abs()
    near_t = (distance[:num_frames, :num_frames] <= window_t // 2).flatten()
    near_h = (distance[:height, :height] <= window_h // 2).flatten()
    near_w = (distance[:width, :width] <= window_w // 2).flatten()
    near_hw = (near_h[:, None] & near_w[None, :]).reshape(height, height, width, width).transpose(1, 2).flatten()
    near = (near_t[:, None] & near_hw[None, :]).reshape(num_frames, num_frames, height * width, height * width)
    return near.transpose(1, 2).reshape(num_frames * height * width, num_frames * height * width)


@functools.lru_cache(maxsize=None)
def _warn_uncompiled_flex_attention() -> None:
    logger.warning(
        "Kandinsky 6 SR attention runs PyTorch's flex attention eagerly, which materializes the full attention "
        "matrix and does not fit in memory at video resolutions. Compile the transformer, e.g. with "
        "`transformer.compile_repeated_blocks()`."
    )


class Kandinsky6SRAttnProcessor:
    """Self-attention processor of the SR transformer: NABLA sparse attention over the `sparse_params` block pattern.

    Always dispatches on the `flex` backend: NABLA's sparsity is expressed as a `BlockMask`, which only `flex` can
    consume. That backend also needs to run under `torch.compile` (e.g. `transformer.compile_repeated_blocks()`):
    uncompiled, PyTorch's flex attention falls back to an eager implementation that materializes the full attention
    matrix, which does not fit in memory at video resolutions.
    """

    _parallel_config = None

    def __call__(
        self,
        attn: "Kandinsky6SRAttention",
        hidden_states: Tensor,
        rotary_emb: Tensor,
        sparse_params: dict[str, Any],
    ) -> Tensor:
        query = attn.to_query(hidden_states).unflatten(-1, (attn.num_heads, -1))
        key = attn.to_key(hidden_states).unflatten(-1, (attn.num_heads, -1))
        value = attn.to_value(hidden_states).unflatten(-1, (attn.num_heads, -1))
        query = attn.query_norm(query.float()).type_as(query)
        key = attn.key_norm(key.float()).type_as(key)
        query = apply_rotary(query, rotary_emb)
        key = apply_rotary(key, rotary_emb)

        if not torch.compiler.is_compiling():
            _warn_uncompiled_flex_attention()
        # The block statistics are computed from the `(B, heads, S, D)` layout the mask builder expects.
        attn_mask = nabla_block_mask(
            query.transpose(1, 2),
            key.transpose(1, 2),
            sparse_params["sta_mask"],
            thr=sparse_params["threshold"],
        )
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            backend=AttentionBackendName.FLEX,
            parallel_config=self._parallel_config,
        )
        return attn.out_layer(hidden_states.flatten(2, 3))


class Kandinsky6SRAttention(nn.Module, AttentionModuleMixin):
    """SR self-attention with the Diffusers `set_processor` contract."""

    _default_processor_cls = Kandinsky6SRAttnProcessor
    _available_processors = [Kandinsky6SRAttnProcessor]

    def __init__(self, num_channels: int, head_dim: int, processor: Kandinsky6SRAttnProcessor | None = None):
        super().__init__()
        if num_channels % head_dim:
            raise ValueError("num_channels must be divisible by head_dim")
        self.num_heads = num_channels // head_dim
        self.to_query = nn.Linear(num_channels, num_channels)
        self.to_key = nn.Linear(num_channels, num_channels)
        self.to_value = nn.Linear(num_channels, num_channels)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels)
        self.set_processor(processor or self._default_processor_cls())

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: Tensor,
        sparse_params: dict[str, Any],
    ) -> Tensor:
        return self.processor(self, hidden_states, rotary_emb, sparse_params)


# Copied from diffusers.models.transformers.transformer_kandinsky6.Kandinsky6TimeEmbeddings with Kandinsky6->Kandinsky6SR
class Kandinsky6SRTimeEmbeddings(nn.Module):
    """Sinusoidal timestep embedding with a K6-compatible parameter layout."""

    def __init__(self, model_dim: int, time_dim: int):
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even")
        self.time_proj = Timesteps(model_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(model_dim, time_dim, act_fn="silu")

    def forward(self, timestep: Tensor) -> Tensor:
        # The sinusoidal embedding is float32; `_keep_in_fp32_modules` keeps these layers float32 under
        # `from_pretrained(torch_dtype=...)`, and the cast aligns the input with whatever dtype they hold.
        embed = self.time_proj(timestep)
        embed = embed.to(get_parameter_dtype(self.timestep_embedder))
        return self.timestep_embedder(embed)


# Copied from diffusers.models.transformers.transformer_kandinsky6.Kandinsky6VisualEmbeddings with Kandinsky6->Kandinsky6SR
class Kandinsky6SRVisualEmbeddings(nn.Module):
    """Patch projection for ``[B,T,H,W,C]`` visual tokens."""

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, duration, height, width, channels = x.shape
        p_t, p_h, p_w = self.patch_size
        x = (
            x.view(batch, duration // p_t, p_t, height // p_h, p_h, width // p_w, p_w, channels)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


# Copied from diffusers.models.transformers.transformer_kandinsky6.Kandinsky6Modulation with Kandinsky6->Kandinsky6SR
class Kandinsky6SRModulation(nn.Module):
    """AdaLN modulation projection."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(x.to(get_parameter_dtype(self.out_layer))))


# Copied from diffusers.models.transformers.transformer_kandinsky6.Kandinsky6OutLayer with Kandinsky6->Kandinsky6SR
class Kandinsky6SROutLayer(nn.Module):
    """Projects visual hidden states back to packed latent patches."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky6SRModulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim)

    def forward(self, visual_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        condition_shape = (scale.shape[0],) + (1,) * (visual_embed.ndim - 2) + (scale.shape[-1],)
        x = apply_scale_shift(
            self.norm(visual_embed.float()),
            visual_embed,
            scale.reshape(condition_shape),
            shift.reshape(condition_shape),
        )
        x = self.out_layer(x)

        batch, duration, height, width = x.shape[:4]
        p_t, p_h, p_w = self.patch_size
        return (
            x.view(batch, duration, height, width, -1, p_t, p_h, p_w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )


# Copied from diffusers.models.transformers.transformer_kandinsky6.Kandinsky6RoPE3D with Kandinsky6RoPE3D->Kandinsky6SRRoPE3D
class Kandinsky6SRRoPE3D(nn.Module):
    """3-D Rotary Position Embedding — used for video spatial-temporal tokens (T, H, W)."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period
        for i, (d, mp) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(d // 2, max_period)
            self.register_buffer(
                f"angles_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype), freq), persistent=False
            )

    def forward(
        self,
        pos: tuple[Tensor, Tensor, Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        # `pos` holds one index tensor per axis; the grid size is implied by their lengths.
        num_frames, height, width = (int(axis_pos.shape[0]) for axis_pos in pos)
        angles_t = self.angles_0[pos[0]] / scale_factor[0]  # (T, d//2)
        angles_h = self.angles_1[pos[1]] / scale_factor[1]  # (H, d//2)
        angles_w = self.angles_2[pos[2]] / scale_factor[2]  # (W, d//2)

        angles = torch.cat(
            [
                angles_t.view(num_frames, 1, 1, -1).expand(num_frames, height, width, -1),
                angles_h.view(1, height, 1, -1).expand(num_frames, height, width, -1),
                angles_w.view(1, 1, width, -1).expand(num_frames, height, width, -1),
            ],
            dim=-1,
        )
        cos, sin = torch.cos(angles), torch.sin(angles)
        rope = torch.stack([cos, -sin, sin, cos], dim=-1)  # (T, H, W, total_dim, 4)
        rope = rope.view(*rope.shape[:-1], 2, 2)  # (T, H, W, total_dim, 2, 2)
        return rope.unsqueeze(-4)  # (T, H, W, 1, total_dim, 2, 2)


class Kandinsky6SRTransformerBlock(nn.Module):
    """AdaLN-modulated self-attention + feed-forward block of the SR transformer."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int):
        super().__init__()
        self.visual_modulation = Kandinsky6SRModulation(time_dim, model_dim, 6)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky6SRAttention(model_dim, head_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, inner_dim=ff_dim, activation_fn="gelu", bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        rotary_emb: Tensor,
        sparse_params: dict[str, Any],
    ) -> Tensor:
        self_attention_params, feed_forward_params = torch.chunk(self.visual_modulation(temb), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attention_params, 3, dim=-1)
        hidden_states = apply_gate_sum(
            hidden_states,
            self.self_attention(
                apply_scale_shift(self.self_attention_norm(hidden_states.float()), hidden_states, scale, shift),
                rotary_emb,
                sparse_params,
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(feed_forward_params, 3, dim=-1)
        return apply_gate_sum(
            hidden_states,
            self.feed_forward(
                apply_scale_shift(self.feed_forward_norm(hidden_states.float()), hidden_states, scale, shift)
            ),
            gate,
        )


class Kandinsky6SRTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    r"""
    Text-free diffusion transformer for Kandinsky 6 video super-resolution.

    The model denoises tiles of the K-VAE latent video. Its input concatenates the noisy latent with the anchor latent
    and the anchor mask that condition the super-resolution (`2 * in_visual_dim + 1` channels), and its output holds
    `out_visual_dim` channels: `in_visual_dim` for a plain velocity checkpoint, or a widened `n_grid * in_visual_dim`
    for the [`PiflowScheduler`] distilled checkpoints. Video self-attention runs through the NABLA sparse block pattern
    on the `flex` attention backend, which needs the token grid (`height` and `width` after `patch_size`) to be
    divisible by 8.

    Args:
        in_visual_dim (`int`, defaults to `64`):
            Number of latent channels of the K-VAE.
        out_visual_dim (`int`, defaults to `640`):
            Number of output channels.
        time_dim (`int`, defaults to `512`):
            Dimension of the timestep embedding.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 1, 1)`):
            Patch size as `(temporal, height, width)`.
        model_dim (`int`, defaults to `1792`):
            Hidden dimension of the transformer.
        ff_dim (`int`, defaults to `7168`):
            Inner dimension of the feed-forward networks.
        num_visual_blocks (`int`, defaults to `32`):
            Number of transformer blocks.
        axes_dims (`tuple[int, int, int]`, defaults to `(16, 24, 24)`):
            RoPE dimensions per `(t, h, w)` axis; their sum is the attention head dimension.
        scale_factor (`tuple[float, float, float]`, defaults to `(1.0, 2.0, 2.0)`):
            Per-axis RoPE frequency scaling applied to the token positions.
        nabla_threshold (`float`, defaults to `0.8`):
            Cumulative-attention threshold of the NABLA block selection.
        nabla_window (`tuple[int, int, int]`, defaults to `(11, 7, 7)`):
            Odd `(t, h, w)` extents of the sliding-tile prior that every 8x8 token block always attends to.
        tile_sizes (`tuple[tuple[int, int], ...]`, defaults to `((512, 512), (512, 768), (768, 512))`):
            Pixel `(height, width)` sizes of the video tiles the model was trained on. [`Kandinsky6SRPipeline`] refines
            every tile at the size whose aspect ratio is closest to the input video's.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Kandinsky6SRTransformerBlock"]
    _repeated_blocks = ["Kandinsky6SRTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # Timestep embeddings, the pooled-text bias they add to, and every AdaLN modulation projection stay float32
    # under `from_pretrained(torch_dtype=...)`.
    _keep_in_fp32_modules = ["time_embeddings", "pooled_bias", "modulation"]

    @register_to_config
    def __init__(
        self,
        in_visual_dim: int = 64,
        out_visual_dim: int = 640,
        time_dim: int = 512,
        patch_size: tuple[int, int, int] = (1, 1, 1),
        model_dim: int = 1792,
        ff_dim: int = 7168,
        num_visual_blocks: int = 32,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0),
        nabla_threshold: float = 0.8,
        nabla_window: tuple[int, int, int] = (11, 7, 7),
        tile_sizes: tuple[tuple[int, int], ...] = ((512, 512), (512, 768), (768, 512)),
    ) -> None:
        super().__init__()
        if not _CAN_USE_FLEX_ATTN:
            raise ImportError(
                "Kandinsky6SRTransformer3DModel requires PyTorch>=2.5.0 with `torch.nn.attention.flex_attention`"
                " for its NABLA sparse attention."
            )
        head_dim = sum(axes_dims)

        self.time_embeddings = Kandinsky6SRTimeEmbeddings(model_dim, time_dim)
        # The SR checkpoints were trained with an empty caption: the pooled-text projection of that caption is a
        # constant, folded into this bias by the checkpoint conversion.
        self.pooled_bias = nn.Parameter(torch.zeros(time_dim))
        self.visual_embeddings = Kandinsky6SRVisualEmbeddings(2 * in_visual_dim + 1, model_dim, patch_size)
        self.visual_rope_embeddings = Kandinsky6SRRoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [Kandinsky6SRTransformerBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_visual_blocks)]
        )
        self.out_layer = Kandinsky6SROutLayer(model_dim, time_dim, out_visual_dim, patch_size)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[Tensor]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_frames, height, width, 2 * in_visual_dim + 1)`):
                Latent tiles in the `(B, T, H, W, C)` layout: the noisy latent, the anchor latent and the anchor mask
                concatenated along the channel axis.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion timestep on the `[0, num_train_timesteps]` scale.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The prediction of shape `(batch_size, num_frames, height, width, out_visual_dim)`.
        """
        checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing
        device = hidden_states.device

        temb = self.time_embeddings(timestep) + self.pooled_bias

        # 1. Patchify the latents and build the rotary embeddings of the token grid
        visual_embed = self.visual_embeddings(hidden_states)
        visual_shape = visual_embed.shape[1:4]
        visual_rope_pos = tuple(torch.arange(size, device=device) for size in visual_shape)
        visual_rope = self.visual_rope_embeddings(visual_rope_pos, self.config.scale_factor)

        # 2. Flatten the grid into a token sequence; NABLA groups 8x8 spatial neighbourhoods into attention blocks
        num_frames, height, width = visual_shape
        if height % FRACTAL_BLOCK_SIZE or width % FRACTAL_BLOCK_SIZE:
            raise ValueError(
                f"NABLA attention needs the token grid {(height, width)} to be divisible by {FRACTAL_BLOCK_SIZE}"
            )
        group_size = (1, FRACTAL_BLOCK_SIZE, FRACTAL_BLOCK_SIZE)
        visual_embed = _local_patch(visual_embed, visual_shape, group_size, dim=1).flatten(1, 2)
        visual_rope = _local_patch(visual_rope, visual_shape, group_size, dim=0).flatten(0, 1)
        sta_mask = sliding_tile_mask(
            num_frames, height // FRACTAL_BLOCK_SIZE, width // FRACTAL_BLOCK_SIZE, self.config.nabla_window, device
        )
        sparse_params = {"sta_mask": sta_mask, "threshold": self.config.nabla_threshold}

        # 3. Transformer blocks
        for block in self.visual_transformer_blocks:
            if checkpoint:
                visual_embed = self._gradient_checkpointing_func(block, visual_embed, temb, visual_rope, sparse_params)
            else:
                visual_embed = block(visual_embed, temb, visual_rope, sparse_params)

        # 4. Restore the grid and project back to the latent space
        visual_embed = _local_merge(
            visual_embed.reshape(visual_embed.shape[0], -1, FRACTAL_BLOCK_SIZE**2, visual_embed.shape[-1]),
            visual_shape,
            group_size,
            dim=1,
        )
        output = self.out_layer(visual_embed, temb)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
