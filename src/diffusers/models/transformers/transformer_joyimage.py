# Copyright 2025 The JoyImage Team and The HuggingFace Team. All rights reserved.
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

import inspect
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# Rotary position embedding utilities
# ---------------------------------------------------------------------------


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    if len(x) == dim:
        return tuple(x)
    raise ValueError(f"Expected length {dim} or int, but got {x}")


def _get_meshgrid_nd(start, *args, dim=2):
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    return torch.stack(grid, dim=0)


def _reshape_for_broadcast(freqs_cis, x: torch.Tensor, head_first: bool = False):
    ndim = x.ndim
    if isinstance(freqs_cis, tuple):
        if head_first:
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)

    if head_first:
        shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def _rotate_half(x: torch.Tensor):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def _apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _reshape_for_broadcast(freqs_cis, xq, head_first)
    cos, sin = cos.to(xq.device), sin.to(xq.device)
    xq_out = (xq.float() * cos + _rotate_half(xq.float()) * sin).type_as(xq)
    xk_out = (xk.float() * cos + _rotate_half(xk.float()) * sin).type_as(xk)
    return xq_out, xk_out


def _get_1d_rotary_pos_embed(
    dim: int,
    pos,
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
):
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    freqs = torch.outer(pos.float() * interpolation_factor, freqs)

    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin

    return torch.polar(torch.ones_like(freqs), freqs)


def _get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    txt_rope_size=None,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
):
    rope_dim_list = list(rope_dim_list)
    grid = _get_meshgrid_nd(start, *args, dim=len(rope_dim_list))

    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [float(theta_rescale_factor)] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [float(theta_rescale_factor[0])] * len(rope_dim_list)

    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [float(interpolation_factor)] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [float(interpolation_factor[0])] * len(rope_dim_list)

    embs = []
    for i in range(len(rope_dim_list)):
        emb = _get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        vis_emb = (torch.cat([emb[0] for emb in embs], dim=1), torch.cat([emb[1] for emb in embs], dim=1))
    else:
        vis_emb = torch.cat(embs, dim=1)

    if txt_rope_size is None:
        return vis_emb, None

    embs_txt = []
    vis_max_ids = grid.view(-1).max().item()
    grid_txt = torch.arange(txt_rope_size) + vis_max_ids + 1
    for i in range(len(rope_dim_list)):
        emb = _get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid_txt,
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs_txt.append(emb)

    if use_real:
        txt_emb = (torch.cat([emb[0] for emb in embs_txt], dim=1), torch.cat([emb[1] for emb in embs_txt], dim=1))
    else:
        txt_emb = torch.cat(embs_txt, dim=1)

    return vis_emb, txt_emb


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------


class JoyImageModulate(nn.Module):
    """Wan-style learnable modulation table.

    Produces `factor` modulation vectors by adding the conditioning signal to a
    learnable parameter table.
    """

    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor, hidden_size, dtype=dtype, device=device) / hidden_size**0.5,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


# ---------------------------------------------------------------------------
# Attention processor
# ---------------------------------------------------------------------------


class JoyImageAttnProcessor:
    """Attention processor for JoyImage double-stream joint attention.

    Implements the joint attention computation where text and image streams are
    processed together.  The :class:`JoyImageAttention` module stores fused QKV
    projections (``img_attn_qkv`` / ``txt_attn_qkv``).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        pass

    def __call__(
        self,
        attn: "JoyImageAttention",
        hidden_states: torch.Tensor,  # image stream  (B, S_img, D)
        encoder_hidden_states: torch.Tensor = None,  # text stream  (B, S_txt, D)
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("JoyImageAttnProcessor requires encoder_hidden_states (text stream)")

        heads = attn.heads

        # image stream: fused QKV -> split
        img_qkv = attn.img_attn_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        # text stream: fused QKV -> split
        txt_qkv = attn.txt_attn_qkv(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        # reshape to multi-head: (B, S, H, D)
        img_query = img_query.unflatten(-1, (heads, -1))
        img_key = img_key.unflatten(-1, (heads, -1))
        img_value = img_value.unflatten(-1, (heads, -1))

        txt_query = txt_query.unflatten(-1, (heads, -1))
        txt_key = txt_key.unflatten(-1, (heads, -1))
        txt_value = txt_value.unflatten(-1, (heads, -1))

        # QK norm
        img_query = attn.img_attn_q_norm(img_query)
        img_key = attn.img_attn_k_norm(img_key)
        txt_query = attn.txt_attn_q_norm(txt_query)
        txt_key = attn.txt_attn_k_norm(txt_key)

        # RoPE (custom implementation)
        if image_rotary_emb is not None:
            vis_freqs, txt_freqs = image_rotary_emb
            if vis_freqs is not None:
                img_query, img_key = _apply_rotary_emb(img_query, img_key, vis_freqs, head_first=False)
            if txt_freqs is not None:
                txt_query, txt_key = _apply_rotary_emb(txt_query, txt_key, txt_freqs, head_first=False)

        # concatenate for joint attention: [img, txt]
        joint_query = torch.cat([img_query, txt_query], dim=1)
        joint_key = torch.cat([img_key, txt_key], dim=1)
        joint_value = torch.cat([img_value, txt_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # split back
        img_attn_output = joint_hidden_states[:, : hidden_states.shape[1], :]
        txt_attn_output = joint_hidden_states[:, hidden_states.shape[1] :, :]

        # output projections
        img_attn_output = attn.img_attn_proj(img_attn_output)
        txt_attn_output = attn.txt_attn_proj(txt_attn_output)

        return img_attn_output, txt_attn_output


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------


class JoyImageAttention(nn.Module, AttentionModuleMixin):
    """Joint attention module for JoyImage double-stream blocks.

    Wraps the fused QKV projections, QK norms, and output projections for both
    image and text streams. Delegates the actual attention computation to a
    pluggable :class:`JoyImageAttnProcessor`.
    """

    _default_processor_cls = JoyImageAttnProcessor
    _available_processors = [JoyImageAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        processor=None,
    ):
        super().__init__()

        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.img_attn_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.img_attn_q_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.img_attn_k_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.img_attn_proj = nn.Linear(inner_dim, dim, bias=True)

        self.txt_attn_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.txt_attn_q_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.txt_attn_k_norm = nn.RMSNorm(attention_head_dim, eps=eps)
        self.txt_attn_proj = nn.Linear(inner_dim, dim, bias=True)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, image_rotary_emb, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _apply_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * gate.unsqueeze(1)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


@maybe_allow_in_graph
class JoyImageTransformerBlock(nn.Module):
    """Double-stream transformer block for JoyImage.

    Each block processes an image stream and a text stream jointly through
    shared attention, following the SD3 / Flux double-stream pattern with
    WAN-style modulation.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        mlp_hidden_dim = int(dim * mlp_width_ratio)

        # image stream
        self.img_mod = JoyImageModulate(dim, factor=6)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        # text stream
        self.txt_mod = JoyImageModulate(dim, factor=6)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        # ---- joint attention ----
        self.attn = JoyImageAttention(dim, num_attention_heads, attention_head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # modulation
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(temb)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(temb)

        # --- attention ---
        img_modulated = _modulate(self.img_norm1(hidden_states), img_mod1_shift, img_mod1_scale)
        txt_modulated = _modulate(self.txt_norm1(encoder_hidden_states), txt_mod1_shift, txt_mod1_scale)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + _apply_gate(img_attn, img_mod1_gate)
        encoder_hidden_states = encoder_hidden_states + _apply_gate(txt_attn, txt_mod1_gate)

        # --- FFN ---
        hidden_states = hidden_states + _apply_gate(
            self.img_mlp(_modulate(self.img_norm2(hidden_states), img_mod2_shift, img_mod2_scale)),
            img_mod2_gate,
        )
        encoder_hidden_states = encoder_hidden_states + _apply_gate(
            self.txt_mlp(_modulate(self.txt_norm2(encoder_hidden_states), txt_mod2_shift, txt_mod2_scale)),
            txt_mod2_gate,
        )

        return hidden_states, encoder_hidden_states


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class JoyImageTransformer3DModel(ModelMixin, ConfigMixin, AttentionMixin):
    """JoyImage Transformer model for image generation / editing.

    Dual-stream DiT architecture with WAN-style conditioning embeddings and
    custom rotary position embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["JoyImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        text_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        num_layers: int = 20,
        rope_dim_list: list[int] = [16, 56, 56],
        rope_type: str = "rope",
        theta: int = 256,
        # legacy config.json keys (kept for backward compatibility)
        heads_num: int | None = None,
        mm_double_blocks_depth: int | None = None,
        text_states_dim: int | None = None,
        rope_theta: int | None = None,
    ):
        super().__init__()

        # --- backward-compatible parameter mapping ---
        if heads_num is not None:
            num_attention_heads = heads_num
        if mm_double_blocks_depth is not None:
            num_layers = mm_double_blocks_depth
        if text_states_dim is not None:
            text_dim = text_states_dim
        if rope_theta is not None:
            theta = rope_theta

        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_dim_list = rope_dim_list
        self.rope_type = rope_type
        self.theta = theta

        attention_head_dim = hidden_size // num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        # image projection
        self.img_in = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # condition embedder (re-uses WAN implementation)
        from .transformer_wan import WanTimeTextImageEmbedding

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_dim,
        )

        # double-stream blocks
        self.double_blocks = nn.ModuleList(
            [
                JoyImageTransformerBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        # output head
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, self.out_channels * math.prod(patch_size))

    # ------------------------------------------------------------------
    # RoPE helper
    # ------------------------------------------------------------------

    def get_rotary_pos_embed(
        self,
        vis_rope_size: list[int],
        txt_rope_size: int | None = None,
    ):
        target_ndim = 3
        if len(vis_rope_size) != target_ndim:
            vis_rope_size = [1] * (target_ndim - len(vis_rope_size)) + list(vis_rope_size)

        head_dim = self.hidden_size // self.num_attention_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal head_dim"

        vis_freqs, txt_freqs = _get_nd_rotary_pos_embed(
            rope_dim_list,
            vis_rope_size,
            txt_rope_size=txt_rope_size,
            theta=self.theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return vis_freqs, txt_freqs

    # ------------------------------------------------------------------
    # Unpatchify
    # ------------------------------------------------------------------

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(x.shape[0], t, h, w, pt, ph, pw, c)
        x = torch.einsum("nthwopqc->nctohpwq", x)
        return x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        return_dict: bool = True,
    ):
        # handle multi-item input (b, n, c, t, h, w)
        is_multi_item = hidden_states.ndim == 6
        num_items = 0
        if is_multi_item:
            num_items = hidden_states.shape[1]
            if num_items > 1:
                assert self.patch_size[0] == 1, "For multi-item input, patch_size[0] must be 1"
                hidden_states = torch.cat(
                    [hidden_states[:, -1:], hidden_states[:, :-1]], dim=1
                )
            # rearrange: (b, n, c, t, h, w) -> (b, c, n*t, h, w)
            b, n, c, t, h, w = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).reshape(b, c, n * t, h, w)

        batch_size, _, ot, oh, ow = hidden_states.shape
        tt = ot // self.patch_size[0]
        th = oh // self.patch_size[1]
        tw = ow // self.patch_size[2]

        # patchify
        img = self.img_in(hidden_states).flatten(2).transpose(1, 2)

        # condition embeddings
        _, vec, txt, _ = self.condition_embedder(timestep, encoder_hidden_states)
        if vec.shape[-1] > self.hidden_size:
            vec = vec.unflatten(1, (6, -1))

        txt_seq_len = txt.shape[1]

        # RoPE
        vis_freqs, txt_freqs = self.get_rotary_pos_embed(
            vis_rope_size=[tt, th, tw],
            txt_rope_size=txt_seq_len if self.rope_type == "mrope" else None,
        )

        # main loop
        for block in self.double_blocks:
            img, txt = block(
                hidden_states=img,
                encoder_hidden_states=txt,
                temb=vec,
                image_rotary_emb=(vis_freqs, txt_freqs),
            )

        # final layer
        img = self.proj_out(self.norm_out(img))
        img = self.unpatchify(img, tt, th, tw)

        # un-multi-item: (b, c, n*t, h, w) -> (b, n, c, t, h, w)
        if is_multi_item:
            c_out = img.shape[1]
            img = img.reshape(batch_size, c_out, num_items, -1, oh, ow)
            img = img.permute(0, 2, 1, 3, 4, 5)  # (b, n, c, t, h, w)
            if num_items > 1:
                img = torch.cat([img[:, 1:], img[:, :1]], dim=1)

        if not return_dict:
            return (img,)
        return Transformer2DModelOutput(sample=img)


class JoyImageEditTransformer3DModel(JoyImageTransformer3DModel):
    """Alias kept for backward compatibility with pipeline imports."""

    pass