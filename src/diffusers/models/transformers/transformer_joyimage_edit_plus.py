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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _apply_rotary_emb_batched(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for batched [B, S, D] freqs."""
    cos, sin = freqs_cis[0].to(xq.device), freqs_cis[1].to(xq.device)

    # batched: [B, S, D] -> [B, S, 1, D]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    def _rotate_half(x):
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    xq_out = (xq.float() * cos + _rotate_half(xq) * sin).type_as(xq)
    xk_out = (xk.float() * cos + _rotate_half(xk) * sin).type_as(xk)
    return xq_out, xk_out


# Copied from diffusers.models.transformers.transformer_joyimage.JoyImageModulate with JoyImage->JoyImageEditPlus
class JoyImageEditPlusModulate(nn.Module):
    """Wan-style learnable modulation table.

    Produces `factor` modulation vectors by adding the conditioning signal to a learnable parameter table.
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


class JoyImageEditPlusAttnProcessor:
    """Attention processor that supports batched RoPE embeddings for edit-plus multi-image input."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        pass

    def __call__(
        self,
        attn: "JoyImageEditPlusAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("JoyImageEditPlusAttnProcessor requires encoder_hidden_states")

        heads = attn.heads

        img_qkv = attn.img_attn_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        txt_qkv = attn.txt_attn_qkv(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        img_query = img_query.unflatten(-1, (heads, -1))
        img_key = img_key.unflatten(-1, (heads, -1))
        img_value = img_value.unflatten(-1, (heads, -1))

        txt_query = txt_query.unflatten(-1, (heads, -1))
        txt_key = txt_key.unflatten(-1, (heads, -1))
        txt_value = txt_value.unflatten(-1, (heads, -1))

        img_query = attn.img_attn_q_norm(img_query)
        img_key = attn.img_attn_k_norm(img_key)
        txt_query = attn.txt_attn_q_norm(txt_query)
        txt_key = attn.txt_attn_k_norm(txt_key)

        if image_rotary_emb is not None:
            vis_freqs, txt_freqs = image_rotary_emb
            if vis_freqs is not None:
                img_query, img_key = _apply_rotary_emb_batched(img_query, img_key, vis_freqs)
            if txt_freqs is not None:
                txt_query, txt_key = _apply_rotary_emb_batched(txt_query, txt_key, txt_freqs)

        joint_query = torch.cat([img_query, txt_query], dim=1)
        joint_key = torch.cat([img_key, txt_key], dim=1)
        joint_value = torch.cat([img_value, txt_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        img_attn_output = joint_hidden_states[:, : hidden_states.shape[1], :]
        txt_attn_output = joint_hidden_states[:, hidden_states.shape[1] :, :]

        img_attn_output = attn.img_attn_proj(img_attn_output)
        txt_attn_output = attn.txt_attn_proj(txt_attn_output)

        return img_attn_output, txt_attn_output


# Copied from diffusers.models.transformers.transformer_joyimage.JoyImageAttention with JoyImage->JoyImageEditPlus
class JoyImageEditPlusAttention(nn.Module, AttentionModuleMixin):
    """Joint attention module for JoyImage Edit Plus double-stream blocks."""

    _default_processor_cls = JoyImageEditPlusAttnProcessor
    _available_processors = [JoyImageEditPlusAttnProcessor]
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
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {}
        if "attention_mask" in attn_parameters:
            kwargs["attention_mask"] = attention_mask
        return self.processor(self, hidden_states, encoder_hidden_states, image_rotary_emb, **kwargs)


# Copied from diffusers.models.transformers.transformer_joyimage.JoyImageTransformerBlock with JoyImage->JoyImageEditPlus
class JoyImageEditPlusTransformerBlock(nn.Module):
    """Double-stream transformer block for JoyImage Edit Plus."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        mlp_hidden_dim = int(dim * mlp_width_ratio)

        # image stream
        self.img_mod = JoyImageEditPlusModulate(dim, factor=6)
        self.img_norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        # text stream
        self.txt_mod = JoyImageEditPlusModulate(dim, factor=6)
        self.txt_norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        # joint attention
        self.attn = JoyImageEditPlusAttention(dim, num_attention_heads, attention_head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        img_normed = self.img_norm1(hidden_states)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        img_modulated = img_normed * (1 + img_mod1_scale.unsqueeze(1)) + img_mod1_shift.unsqueeze(1)
        txt_modulated = txt_normed * (1 + txt_mod1_scale.unsqueeze(1)) + txt_mod1_shift.unsqueeze(1)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + img_attn * img_mod1_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + txt_attn * txt_mod1_gate.unsqueeze(1)

        # --- FFN ---
        img_ffn_normed = self.img_norm2(hidden_states)
        txt_ffn_normed = self.txt_norm2(encoder_hidden_states)
        img_ffn_input = img_ffn_normed * (1 + img_mod2_scale.unsqueeze(1)) + img_mod2_shift.unsqueeze(1)
        txt_ffn_input = txt_ffn_normed * (1 + txt_mod2_scale.unsqueeze(1)) + txt_mod2_shift.unsqueeze(1)
        img_ffn_output = self.img_mlp(img_ffn_input)
        txt_ffn_output = self.txt_mlp(txt_ffn_input)
        hidden_states = hidden_states + img_ffn_output * img_mod2_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + txt_ffn_output * txt_mod2_gate.unsqueeze(1)

        return hidden_states, encoder_hidden_states


# Copied from diffusers.models.transformers.transformer_joyimage.JoyImageTimeTextImageEmbedding with JoyImage->JoyImageEditPlus
class JoyImageEditPlusTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class JoyImageEditPlusTransformer3DModel(ModelMixin, ConfigMixin, AttentionMixin):
    r"""
    JoyImage Edit Plus Transformer for multi-image editing.

    Uses a patchify+padding approach where each reference image and the target noise are independently
    patchified and concatenated into a flat patch sequence. Supports variable-resolution reference images.

    Input format: `[B, max_patches, C, pt, ph, pw]` (6D padded patches).

    Args:
        patch_size (`list`, defaults to `[1, 2, 2]`):
            Patch size for patchifying the latent input along `(t, h, w)` dimensions.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input latent.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        hidden_size (`int`, defaults to `3072`):
            The dimensionality of the hidden representations.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads.
        text_dim (`int`, defaults to `4096`):
            The dimensionality of the text encoder output.
        mlp_width_ratio (`float`, defaults to `4.0`):
            The ratio of MLP hidden dimension to `hidden_size`.
        num_layers (`int`, defaults to `20`):
            The number of double-stream transformer blocks.
        rope_dim_list (`list[int]`, defaults to `[16, 56, 56]`):
            The dimensions for 3D rotary positional embeddings along `(t, h, w)`.
        rope_type (`str`, defaults to `"rope"`):
            The type of rotary positional embedding.
        theta (`int`, defaults to `256`):
            The base frequency for rotary embeddings.
    """

    _skip_layerwise_casting_patterns = ["img_in", "condition_embedder", "norm"]
    _no_split_modules = ["JoyImageEditPlusTransformerBlock"]
    _supports_gradient_checkpointing = True
    _keep_in_fp32_modules = [
        "time_embedder",
        "norm1",
        "norm2",
        "norm_out",
    ]
    _repeated_blocks = ["JoyImageEditPlusTransformerBlock"]

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
    ):
        super().__init__()

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

        self.img_in = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = JoyImageEditPlusTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                JoyImageEditPlusTransformerBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, self.out_channels * math.prod(patch_size))

        self.gradient_checkpointing = False

        # Set batched-RoPE-aware attention processor on all blocks
        for block in self.double_blocks:
            block.attn.set_processor(JoyImageEditPlusAttnProcessor())

    def _get_rotary_pos_embed_for_range(
        self,
        start: tuple[int, int, int],
        stop: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate 3D RoPE for a spatial range [start, stop)."""
        head_dim = self.hidden_size // self.num_attention_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // 3] * 3

        grids = []
        for i in range(3):
            grids.append(torch.arange(start[i], stop[i], dtype=torch.float32))

        mesh = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=0)

        cos_parts, sin_parts = [], []
        for i, dim in enumerate(rope_dim_list):
            pos = mesh[i].reshape(-1)
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
            angles = torch.outer(pos, freqs)
            cos_parts.append(angles.cos().repeat_interleave(2, dim=1))
            sin_parts.append(angles.sin().repeat_interleave(2, dim=1))

        return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        shape_list: list[list[tuple[int, int, int]]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor | tuple:
        """
        Args:
            hidden_states: [B, max_patches, C, pt, ph, pw] - patchified latent input.
            timestep: [B] - diffusion timestep.
            encoder_hidden_states: [B, L, D] - text encoder outputs.
            encoder_hidden_states_mask: [B, L] - attention mask for text tokens.
            shape_list: Per-sample list of (t, h, w) tuples for each component (target + references).
            return_dict: Whether to return a dict or tuple.
        """
        batch_size, max_num_patches, channels, pt, ph, pw = hidden_states.shape
        device = hidden_states.device

        # 1. Condition embeddings
        _, vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        vec = vec.unflatten(1, (6, -1))

        # 2. Patchify via Conv3d: flatten (B, N) -> apply conv -> reshape back
        x = hidden_states.reshape(batch_size * max_num_patches, channels, pt, ph, pw)
        x = self.img_in(x)  # (B*N, D, 1, 1, 1)
        img = x.reshape(batch_size, max_num_patches, -1)

        # 3. Build per-component RoPE with temporal offsets
        sample_cos_list, sample_sin_list = [], []

        for i in range(batch_size):
            s_cos_parts, s_sin_parts = [], []
            current_t_offset = 0

            for thw in shape_list[i]:
                t, h, w = thw
                start = (current_t_offset, 0, 0)
                stop = (current_t_offset + t, h, w)
                cos_emb, sin_emb = self._get_rotary_pos_embed_for_range(start, stop)
                s_cos_parts.append(cos_emb)
                s_sin_parts.append(sin_emb)
                current_t_offset += t

            s_cos = torch.cat(s_cos_parts, dim=0).to(device)
            s_sin = torch.cat(s_sin_parts, dim=0).to(device)

            actual_len = s_cos.shape[0]
            pad_len = max_num_patches - actual_len
            if pad_len > 0:
                s_cos = F.pad(s_cos, (0, 0, 0, pad_len), value=1.0)
                s_sin = F.pad(s_sin, (0, 0, 0, pad_len), value=0.0)

            sample_cos_list.append(s_cos)
            sample_sin_list.append(s_sin)

        vis_freqs = (torch.stack(sample_cos_list), torch.stack(sample_sin_list))

        # 4. Build attention mask: [B, 1, 1, img_seq + txt_seq]
        attention_mask = None
        if encoder_hidden_states_mask is not None:
            img_mask = torch.zeros(batch_size, max_num_patches, device=device, dtype=encoder_hidden_states_mask.dtype)
            for i in range(batch_size):
                actual_len = sum(t * h * w for t, h, w in shape_list[i])
                img_mask[i, :actual_len] = 1.0
            full_mask = torch.cat([img_mask, encoder_hidden_states_mask], dim=1)
            attention_mask = full_mask.unsqueeze(1).unsqueeze(1).bool()

        # 5. Run double blocks
        for block in self.double_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img, txt = self._gradient_checkpointing_func(block, img, txt, vec, (vis_freqs, None), attention_mask)
            else:
                img, txt = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    temb=vec,
                    image_rotary_emb=(vis_freqs, None),
                    attention_mask=attention_mask,
                )

        # 6. Output projection + reshape to 6D patches
        img = self.proj_out(self.norm_out(img))
        img = img.reshape(
            batch_size, max_num_patches, pt, ph, pw, self.out_channels
        ).permute(0, 1, 5, 2, 3, 4)  # -> [B, N, C, pt, ph, pw]

        if not return_dict:
            return (img,)
        return Transformer2DModelOutput(sample=img)
