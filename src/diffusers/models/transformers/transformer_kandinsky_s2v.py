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
"""Kandinsky S2V 1.0 Diffusers transformer."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype


logger = logging.get_logger(__name__)

# The first latent frame takes a single audio token, so the rest need at least one frame.
_MIN_DURATION = 2


# Copied from diffusers.models.transformers.transformer_kandinsky.get_freqs
def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)
    return freqs


def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """RoPE apply in fp32 (rope tables are fp32), cast back to ``x.dtype``."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope.float() * x_).sum(dim=-1).reshape(*x.shape).to(dtype=x.dtype)


def split_lengths(total: int, groups: int, device: torch.device) -> Tensor:
    """Split ``total`` items into ``groups`` with boundaries at ``total * i // groups``."""
    bounds = torch.arange(groups + 1, device=device) * total // groups
    return torch.diff(bounds)


def audio_group_lengths(num_audio_tokens: int, duration: int, device: torch.device) -> Tensor:
    """First latent frame gets one audio token, the remaining frames split the rest evenly."""
    if duration < _MIN_DURATION:
        raise ValueError(f"duration must be >= {_MIN_DURATION} to split audio tokens, got {duration}")
    # Fewer tokens than frames leaves empty groups, whose attention rows are fully
    # masked; SDPA returns NaN for those and it spreads through every later block.
    if num_audio_tokens < duration:
        raise ValueError(
            f"{num_audio_tokens} audio tokens cannot cover {duration} latent frames: every frame needs at least one"
        )
    rest = split_lengths(num_audio_tokens - 1, duration - 1, device)
    first = torch.ones(1, dtype=rest.dtype, device=device)
    return torch.cat([first, rest])


def pad_groups(x: Tensor, lengths: Tensor, max_len: int) -> tuple[Tensor, Tensor]:
    """Scatter a packed ``(S, ...)`` tensor into ``(G, max_len, ...)`` plus a validity mask."""
    positions = torch.arange(max_len, device=x.device)
    valid = positions[None, :] < lengths[:, None]
    out = x.new_zeros(lengths.shape[0], max_len, *x.shape[1:])
    out[valid] = x
    return out, valid


class KandinskyS2VTimeEmbeddings(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0):
        super().__init__()
        if model_dim % 2 != 0:
            raise ValueError(f"model_dim must be even, got {model_dim}")
        self.model_dim = model_dim
        self.max_period = max_period
        # A plain attribute rather than a buffer: it must survive a meta-device build.
        self.freqs = get_freqs(model_dim // 2, max_period)

        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    # Not copied from transformer_kandinsky6: that version casts the result to the
    # weight dtype, while native S2V keeps the time embedding in fp32 all the way
    # into the AdaLN path. The cast would shift rounding in bf16.
    def forward(self, time: Tensor) -> Tensor:
        args = torch.outer(time.to(torch.float32), self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.out_layer(self.activation(self.in_layer(time_embed)))


class KandinskyS2VGeneralEmbeddings(nn.Module):
    """Linear projection plus LayerNorm; used for text, pooled text and audio."""

    def __init__(self, dim: int, model_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, embed: Tensor) -> Tensor:
        embed = self.in_layer(embed)
        # Norm weights are fp32 and activations bf16; native generation reconciles
        # that through autocast, which a Diffusers pipeline does not set up.
        return self.norm(embed.float()).type_as(embed)


class KandinskyS2VVisualEmbeddings(nn.Module):
    """Patch projection for ``[T,H,W,C]`` or ``[B,T,H,W,C]`` visual tokens."""

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        batched = bool(x.ndim == 5)
        offset = 1 if batched else 0
        if not batched and x.ndim != 4:
            raise ValueError("visual input must have shape (T,H,W,C) or (B,T,H,W,C)")

        shape = x.shape
        duration, height, width, channels = shape[offset:]
        p_t, p_h, p_w = self.patch_size
        if batched:
            x = (
                x.view(
                    shape[0],
                    duration // p_t,
                    p_t,
                    height // p_h,
                    p_h,
                    width // p_w,
                    p_w,
                    channels,
                )
                .permute(0, 1, 3, 5, 2, 4, 6, 7)
                .flatten(4, 7)
            )
        else:
            x = (
                x.view(duration // p_t, p_t, height // p_h, p_h, width // p_w, p_w, channels)
                .permute(0, 2, 4, 1, 3, 5, 6)
                .flatten(3, 6)
            )
        return self.in_layer(x)


# Copied from diffusers.models.transformers.transformer_kandinsky6.Kandinsky6Modulation with Kandinsky6->KandinskyS2V
class KandinskyS2VModulation(nn.Module):
    """AdaLN modulation projection."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(x.to(get_parameter_dtype(self.out_layer))))


# Copied from diffusers.models.transformers.transformer_kandinsky.Kandinsky5FeedForward with Kandinsky5->KandinskyS2V
class KandinskyS2VFeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


class KandinskyS2VAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(  # noqa: PLR0913
        self,
        attn: KandinskyS2VAttention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        rotary_emb: Tensor | None = None,
        q_lengths: Tensor | None = None,
        kv_lengths: Tensor | None = None,
    ) -> Tensor:
        cond = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        shape, cond_shape = hidden_states.shape[:-1], cond.shape[:-1]

        query = attn.to_query(hidden_states).reshape(*shape, attn.num_heads, -1)
        key = attn.to_key(cond).reshape(*cond_shape, attn.num_heads, -1)
        value = attn.to_value(cond).reshape(*cond_shape, attn.num_heads, -1)

        query = attn.query_norm(query.float()).type_as(query)
        key = attn.key_norm(key.float()).type_as(key)

        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb).type_as(query)
            key = apply_rotary(key, rotary_emb).type_as(key)

        if q_lengths is not None:
            if kv_lengths is None:
                raise ValueError("kv_lengths is required when q_lengths is given")
            # Block-diagonal attention: query group i attends exactly to key group i. The groups are uneven, so they
            # are padded into a batch axis and masked rather than reshaped.
            query, q_valid = pad_groups(query, q_lengths, int(q_lengths.max()))
            kv_max = int(kv_lengths.max())
            key, kv_valid = pad_groups(key, kv_lengths, kv_max)
            value, _ = pad_groups(value, kv_lengths, kv_max)
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=kv_valid[:, None, None, :],
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )[q_valid]
        else:
            hidden_states = dispatch_attention_fn(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            ).squeeze(0)

        return attn.out_layer(hidden_states.flatten(-2, -1))


class KandinskyS2VAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = KandinskyS2VAttnProcessor
    _available_processors = [KandinskyS2VAttnProcessor]

    def __init__(self, num_channels: int, head_dim: int, processor: Any | None = None):
        super().__init__()
        if num_channels % head_dim != 0:
            raise ValueError(f"num_channels={num_channels} not divisible by head_dim={head_dim}")
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)
        self.set_processor(processor if processor is not None else self._default_processor_cls())

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        accepted = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused = [k for k in kwargs if k not in accepted]
        if unused:
            logger.warning(
                f"attention_processor_kwargs {unused} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return self.processor(self, hidden_states, **kwargs)


# Not copied from transformer_kandinsky6: that OutLayer takes one time embedding,
# while this one also takes `time_embed_idx` to pick a modulation row per token —
# memory and the reference frame are modulated at t=0, the noisy tail at t.
class KandinskyS2VOutLayer(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        if patch_size[0] != 1:
            # Re-expanding the temporal patch needs cu_seqlens; every shipped S2V
            # config uses patch_size[0] == 1.
            raise ValueError(f"temporal patch size must be 1, got {patch_size[0]}")
        self.patch_size = patch_size
        self.modulation = KandinskyS2VModulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(self, visual_embed: Tensor, time_embed: Tensor, time_embed_idx: Tensor) -> Tensor:
        """``(T, H, W, model_dim)`` -> ``(T*p0, H*p1, W*p2, visual_dim)``."""
        shift, scale = (
            p.float()[:, None, None].index_select(0, time_embed_idx)
            for p in torch.chunk(self.modulation(time_embed), 2, dim=-1)
        )
        visual_embed = (self.norm(visual_embed.float()) * (scale + 1.0) + shift).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        duration, height, width, _ = x.shape
        return (
            x.view(duration, height, width, -1, *self.patch_size)
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )


# Not copied from transformer_kandinsky6: its tables are buffers, which
# `.to(torch.bfloat16)` casts down.
class KandinskyS2VRoPE1D(nn.Module):
    def __init__(self, dim: int, max_pos: int = 1024, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.max_period = max_period
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        # A plain attribute rather than a buffer, so `.to(torch.bfloat16)` leaves the table in fp32.
        self.args = torch.outer(pos, freq)

    def forward(self, pos: Tensor) -> Tensor:
        args = self.args.to(device=pos.device)[pos]
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)


# Not copied from transformer_kandinsky6: its tables are buffers, and its temporal
# table is built over arange(max_pos), which would silently wrap a negative index
# around the end of the tensor. FramePack memory sits at negative temporal positions.
class KandinskyS2VRoPE3D(nn.Module):
    """3-D RoPE; the temporal axis spans negative positions for memory frames."""

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
        self.offsets_t = max_pos[0]

        # Plain attributes rather than buffers, so `.to(torch.bfloat16)` leaves the tables in fp32.
        self.args = []
        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos, strict=True)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = (
                torch.arange(-ax_max_pos, ax_max_pos, dtype=freq.dtype)
                if i == 0
                else torch.arange(ax_max_pos, dtype=freq.dtype)
            )
            self.args.append(torch.outer(pos, freq))

    def forward(
        self,
        shape: tuple[int, int, int],
        pos: list[Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        duration, height, width = shape
        args_0, args_1, args_2 = (
            args.to(device=axis_pos.device) for args, axis_pos in zip(self.args, pos, strict=True)
        )
        args_t = args_0[pos[0] + self.offsets_t] / scale_factor[0]
        args_h = args_1[pos[1]] / scale_factor[1]
        args_w = args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)


class KandinskyS2VTransformerTextEncoderBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int):
        super().__init__()
        self.text_modulation = KandinskyS2VModulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = KandinskyS2VAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = KandinskyS2VFeedForward(model_dim, ff_dim)

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor, time_embed_idx: Tensor) -> Tensor:
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = (
            self.self_attention_norm(x.float()) * (scale.float().index_select(0, time_embed_idx) + 1.0)
            + shift.float().index_select(0, time_embed_idx)
        ).type_as(x)
        out = self.self_attention(out, rotary_emb=rope)
        x = (x.float() + gate.float().index_select(0, time_embed_idx) * out.float()).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = (
            self.feed_forward_norm(x.float()) * (scale.float().index_select(0, time_embed_idx) + 1.0)
            + shift.float().index_select(0, time_embed_idx)
        ).type_as(x)
        out = self.feed_forward(out)
        return (x.float() + gate.float().index_select(0, time_embed_idx) * out.float()).type_as(x)


class KandinskyS2VTransformerAudioEncoderBlock(nn.Module):
    """Plain pre-norm transformer block — the audio tower is not time-modulated."""

    def __init__(self, model_dim: int, ff_dim: int, head_dim: int):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.self_attention = KandinskyS2VAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.feed_forward = KandinskyS2VFeedForward(model_dim, ff_dim)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        x = x + self.self_attention(self.self_attention_norm(x.float()).type_as(x), rotary_emb=rope)
        return x + self.feed_forward(self.feed_forward_norm(x.float()).type_as(x))


class KandinskyS2VTransformerDecoderBlock(nn.Module):
    """Visual block; ``enable_audio`` adds the frame-aligned speech adapter."""

    def __init__(  # noqa: PLR0913
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        enable_audio: bool = False,
    ):
        super().__init__()
        self.visual_modulation = KandinskyS2VModulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = KandinskyS2VAttention(model_dim, head_dim)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = KandinskyS2VAttention(model_dim, head_dim)

        self.enable_audio = enable_audio
        if enable_audio:
            self.cross_attention_norm_v2a = nn.LayerNorm(model_dim, elementwise_affine=True)
            self.cross_attention_v2a = KandinskyS2VAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = KandinskyS2VFeedForward(model_dim, ff_dim)

    def forward(  # noqa: PLR0913
        self,
        visual_embed: Tensor,
        text_embed: Tensor,
        audio_embed: Tensor,
        time_embed: Tensor,
        rope: Tensor,
        time_embed_idx: Tensor,
        num_visual_tokens: int,
        visual_group_lengths: Tensor,
        audio_group_lengths: Tensor,
    ) -> Tensor:
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = (
            self.self_attention_norm(visual_embed.float()) * (scale.float().index_select(0, time_embed_idx) + 1.0)
            + shift.float().index_select(0, time_embed_idx)
        ).type_as(visual_embed)
        out = self.self_attention(out, rotary_emb=rope)
        visual_embed = (visual_embed.float() + gate.float().index_select(0, time_embed_idx) * out.float()).type_as(
            visual_embed
        )

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        out = (
            self.cross_attention_norm(visual_embed.float()) * (scale.float().index_select(0, time_embed_idx) + 1.0)
            + shift.float().index_select(0, time_embed_idx)
        ).type_as(visual_embed)
        out = self.cross_attention(out, encoder_hidden_states=text_embed)
        visual_embed = (visual_embed.float() + gate.float().index_select(0, time_embed_idx) * out.float()).type_as(
            visual_embed
        )

        if self.enable_audio:
            # Speech conditions only the noisy tail, frame by frame. The residual is
            # ungated, as in native generation with audio_adaln=False.
            head, noisy = visual_embed[:-num_visual_tokens], visual_embed[-num_visual_tokens:]
            out = self.cross_attention_v2a(
                self.cross_attention_norm_v2a(noisy.float()).type_as(noisy),
                encoder_hidden_states=audio_embed,
                q_lengths=visual_group_lengths,
                kv_lengths=audio_group_lengths,
            )
            visual_embed = torch.cat([head, noisy + out], dim=0)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = (
            self.feed_forward_norm(visual_embed.float()) * (scale.float().index_select(0, time_embed_idx) + 1.0)
            + shift.float().index_select(0, time_embed_idx)
        ).type_as(visual_embed)
        out = self.feed_forward(out)
        return (visual_embed.float() + gate.float().index_select(0, time_embed_idx) * out.float()).type_as(
            visual_embed
        )


class KandinskyS2VTransformer3DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """Speech-to-video diffusion transformer.

    Parameter names match the training checkpoints, so a converted `.safetensors` loads with no key remapping.
    Inference is batch size 1: the sequence is packed and the speech cross-attention is frame-aligned, so `forward`
    takes a leading batch axis of 1 and squeezes it.

    The visual stream is one packed sequence ``[memory | ref | visual]``:

    * ``memory`` — FramePack context from the previous chunk at 4x / 2x / 1x temporal downsampling, placed at negative
      temporal RoPE positions;
    * ``ref`` — the reference frame, pinned to temporal RoPE index ``index_rope_ref_image``;
    * ``visual`` — the noisy latent being denoised.

    ``memory`` and ``ref`` are clean, so they are modulated with ``t = 0`` while ``visual`` uses the real timestep; the
    modulation holds both rows and an index selects per token.
    """

    _repeated_blocks = [
        "KandinskyS2VTransformerTextEncoderBlock",
        "KandinskyS2VTransformerAudioEncoderBlock",
        "KandinskyS2VTransformerDecoderBlock",
    ]
    _no_split_modules = _repeated_blocks
    # The checkpoint keeps the time embedding and every normalisation in fp32, and
    # AdaLN modulation runs in fp32; this stops `torch_dtype=` from flattening that.
    # Matched both as substrings and as exact path components, hence the compound
    # names beside the bare ones.
    _keep_in_fp32_modules = [
        "time_embeddings",
        "modulation",
        "text_modulation",
        "visual_modulation",
        "norm",
        "self_attention_norm",
        "cross_attention_norm",
        "cross_attention_norm_v2a",
        "feed_forward_norm",
        "query_norm",
        "key_norm",
    ]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(  # noqa: PLR0913
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        in_audio_dim: int = 3584,
        time_dim: int = 512,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        frame_buckets: tuple[int, int, int] = (2, 4, 24),
        enable_framepack: bool = True,
        index_rope_ref_image: int = 40,
        patch_proj1x_coef: tuple[int, int, int] = (1, 1, 1),
        patch_proj2x_coef: tuple[int, int, int] = (2, 2, 2),
        patch_proj4x_coef: tuple[int, int, int] = (4, 4, 4),
        model_dim: int = 1792,
        ff_dim: int = 7168,
        num_text_blocks: int = 2,
        num_audio_blocks: int = 2,
        num_visual_blocks: int = 32,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        audio_inject_layers: tuple[int, ...] = (),
        text_rope_max_pos: int = 4096,
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.frame_buckets = frame_buckets
        self.enable_framepack = enable_framepack
        self.index_rope_ref_image = index_rope_ref_image
        self.audio_inject_layers = tuple(audio_inject_layers)

        self.time_embeddings = KandinskyS2VTimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = KandinskyS2VGeneralEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = KandinskyS2VGeneralEmbeddings(in_text_dim2, time_dim)
        self.audio_embeddings = KandinskyS2VGeneralEmbeddings(in_audio_dim, model_dim)

        self.visual_embeddings = KandinskyS2VVisualEmbeddings(in_visual_dim, model_dim, patch_size)
        self.visual_embeddings_proj1x = KandinskyS2VVisualEmbeddings(
            in_visual_dim,
            model_dim,
            (
                patch_size[0] * patch_proj1x_coef[0],
                patch_size[1] * patch_proj1x_coef[1],
                patch_size[2] * patch_proj1x_coef[2],
            ),
        )
        self.visual_embeddings_proj2x = KandinskyS2VVisualEmbeddings(
            in_visual_dim,
            model_dim,
            (
                patch_size[0] * patch_proj2x_coef[0],
                patch_size[1] * patch_proj2x_coef[1],
                patch_size[2] * patch_proj2x_coef[2],
            ),
        )
        self.visual_embeddings_proj4x = KandinskyS2VVisualEmbeddings(
            in_visual_dim,
            model_dim,
            (
                patch_size[0] * patch_proj4x_coef[0],
                patch_size[1] * patch_proj4x_coef[1],
                patch_size[2] * patch_proj4x_coef[2],
            ),
        )
        # 0 = visual, 1 = reference frame, 2 = memory
        self.mask_embeddings = nn.Embedding(3, model_dim)

        self.text_rope_embeddings = KandinskyS2VRoPE1D(head_dim, max_pos=text_rope_max_pos)
        self.text_transformer_blocks = nn.ModuleList(
            [
                KandinskyS2VTransformerTextEncoderBlock(model_dim, time_dim, ff_dim, head_dim)
                for _ in range(num_text_blocks)
            ]
        )

        self.audio_rope_embeddings = KandinskyS2VRoPE1D(head_dim)
        self.audio_transformer_blocks = nn.ModuleList(
            [KandinskyS2VTransformerAudioEncoderBlock(model_dim, ff_dim, head_dim) for _ in range(num_audio_blocks)]
        )

        self.visual_rope_embeddings = KandinskyS2VRoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                KandinskyS2VTransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    enable_audio=(i in self.audio_inject_layers),
                )
                for i in range(num_visual_blocks)
            ]
        )

        self.out_layer = KandinskyS2VOutLayer(model_dim, time_dim, out_visual_dim, patch_size)
        self.gradient_checkpointing = False

    def forward(  # noqa: PLR0913, PLR0915
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        audio_hidden_states: Tensor,
        reference_hidden_states: Tensor,
        memory_hidden_states: tuple[Tensor, Tensor, Tensor],
        timestep: Tensor,
        pooled_projections: Tensor,
        visual_rope_pos: list[Tensor],
        reference_rope_pos: list[Tensor],
        memory_rope_pos: tuple[list[Tensor], list[Tensor], list[Tensor]],
        text_rope_pos: Tensor,
        audio_rope_pos: Tensor,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[Tensor]:
        """Denoise one chunk.

        Args:
            hidden_states: noisy latent, ``(1, T, H, W, C)``.
            encoder_hidden_states: text embeddings, ``(1, S_text, in_text_dim)``.
            audio_hidden_states: speech features, ``(1, S_audio, in_audio_dim)``.
            reference_hidden_states: reference frame latent, ``(1, 1, H, W, C)``.
            memory_hidden_states: FramePack memory at 4x / 2x / 1x, each ``(1, T_i, H_i, W_i, C)``.
            timestep: current timestep, ``(1,)``.
            pooled_projections: pooled CLIP text embedding, ``(1, in_text_dim2)``.
            visual_rope_pos: per-axis RoPE positions of the noisy latent.
            reference_rope_pos: per-axis RoPE positions of the reference frame.
            memory_rope_pos: per-axis RoPE positions of each FramePack memory scale.
            text_rope_pos: 1-D RoPE positions of the text tokens.
            audio_rope_pos: latent frame each speech token belongs to.
            scale_factor: RoPE scale per axis, set by the resolution bucket.
            return_dict: return a ``Transformer2DModelOutput`` instead of a tuple.

        Batch size must be 1: the sequence is packed and the speech cross-attention is frame-aligned.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`: the predicted velocity, shaped like
            ``hidden_states``.
        """
        if hidden_states.shape[0] != 1:
            raise ValueError(f"batch size must be 1, got {hidden_states.shape[0]}")

        x = hidden_states.squeeze(0)
        ref = reference_hidden_states.squeeze(0)
        memory = tuple(m.squeeze(0) for m in memory_hidden_states)
        text_embed = encoder_hidden_states.squeeze(0)
        audio_embed = audio_hidden_states.squeeze(0)

        device = x.device
        duration = x.shape[0]

        text_embed = self.text_embeddings(text_embed)
        audio_embed = self.audio_embeddings(audio_embed)

        time_embed = self.time_embeddings(timestep)
        time_embed_clean = self.time_embeddings(torch.zeros_like(timestep))
        pooled = self.pooled_text_embeddings(pooled_projections)
        time_embed = time_embed + pooled
        time_embed_clean = time_embed_clean + pooled
        # row 0 -> memory / ref (t = 0), row 1 -> visual (t)
        time_embed_full = torch.stack((time_embed_clean, time_embed), dim=1).flatten(0, 1)

        text_rope = self.text_rope_embeddings(text_rope_pos)
        text_time_embed_idx = torch.zeros(text_embed.shape[0], dtype=torch.int32, device=device)
        for block in self.text_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                text_embed = self._gradient_checkpointing_func(
                    block, text_embed, time_embed, text_rope, text_time_embed_idx
                )
            else:
                text_embed = block(text_embed, time_embed, text_rope, text_time_embed_idx)

        audio_rope = self.audio_rope_embeddings(audio_rope_pos)
        for block in self.audio_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                audio_embed = self._gradient_checkpointing_func(block, audio_embed, audio_rope)
            else:
                audio_embed = block(audio_embed, audio_rope)

        visual_embed = self.visual_embeddings(x)
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor).flatten(0, 2)
        visual_embed = visual_embed.flatten(0, 2)

        ref_embed = self.visual_embeddings(ref)
        ref_rope = self.visual_rope_embeddings(ref_embed.shape[:-1], reference_rope_pos, scale_factor).flatten(0, 2)
        ref_embed = ref_embed.flatten(0, 2)

        # The FramePack memory at its three temporal scales.
        memory_4x, memory_2x, memory_1x = memory
        rope_pos_4x, rope_pos_2x, rope_pos_1x = memory_rope_pos
        memory_parts = [(self.visual_embeddings_proj1x(memory_1x), rope_pos_1x)]
        if self.enable_framepack:
            memory_parts = [
                (self.visual_embeddings_proj4x(memory_4x), rope_pos_4x),
                (self.visual_embeddings_proj2x(memory_2x), rope_pos_2x),
                *memory_parts,
            ]
        memory_embed = torch.cat([embed.flatten(0, 2) for embed, _ in memory_parts], dim=0)
        memory_rope = torch.cat(
            [
                self.visual_rope_embeddings(embed.shape[:-1], rope_pos, scale_factor).flatten(0, 2)
                for embed, rope_pos in memory_parts
            ],
            dim=0,
        )

        visual_embed = visual_embed + self.mask_embeddings(torch.tensor(0, device=device))
        ref_embed = ref_embed + self.mask_embeddings(torch.tensor(1, device=device))
        memory_embed = memory_embed + self.mask_embeddings(torch.tensor(2, device=device))

        num_visual_tokens = visual_embed.shape[0]
        num_clean_tokens = memory_embed.shape[0] + ref_embed.shape[0]
        visual_embed_full = torch.cat([memory_embed, ref_embed, visual_embed], dim=0)
        visual_rope_full = torch.cat([memory_rope, ref_rope, visual_rope], dim=0)
        time_embed_idx_full = torch.cat(
            [
                torch.zeros(num_clean_tokens, dtype=torch.int32, device=device),
                torch.ones(num_visual_tokens, dtype=torch.int32, device=device),
            ]
        )

        visual_groups = split_lengths(num_visual_tokens, duration, device)
        audio_groups = audio_group_lengths(audio_embed.shape[0], duration, device)

        for block in self.visual_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                visual_embed_full = self._gradient_checkpointing_func(
                    block,
                    visual_embed_full,
                    text_embed,
                    audio_embed,
                    time_embed_full,
                    visual_rope_full,
                    time_embed_idx_full,
                    num_visual_tokens,
                    visual_groups,
                    audio_groups,
                )
            else:
                visual_embed_full = block(
                    visual_embed_full,
                    text_embed,
                    audio_embed,
                    time_embed_full,
                    visual_rope_full,
                    time_embed_idx_full,
                    num_visual_tokens,
                    visual_groups,
                    audio_groups,
                )

        visual_embed = visual_embed_full[-num_visual_tokens:].reshape(*visual_shape, self.model_dim)
        visual_time_embed_idx = torch.zeros(visual_shape[0], dtype=torch.int32, device=device)
        out = self.out_layer(visual_embed, time_embed, visual_time_embed_idx).unsqueeze(0)

        if not return_dict:
            return (out,)
        return Transformer2DModelOutput(sample=out)
