# Copyright 2025 The SkyReels-V2 Team, The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import FeedForward
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


logger = logging.get_logger(__name__)

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")


class SkyReelsV2AttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "SkyReelsV2AttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

        self._flag_ar_attention = False

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # image_context_length is hardcoded for now like in the original code
            image_context_length = 257
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if self._flag_ar_attention:
            is_self_attention = encoder_hidden_states == hidden_states
            hidden_states = F.scaled_dot_product_attention(
                query.to(torch.bfloat16) if is_self_attention else query,
                key.to(torch.bfloat16) if is_self_attention else key,
                value.to(torch.bfloat16) if is_self_attention else value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def set_ar_attention(self):
        self._flag_ar_attention = True


class SkyReelsV2ImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class SkyReelsV2TimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = SkyReelsV2ImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class SkyReelsV2RotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class SkyReelsV2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=SkyReelsV2AttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=SkyReelsV2AttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if temb.dim() == 3:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)
        elif temb.dim() == 4:
            e = (self.scale_shift_table.unsqueeze(2) + temb.float()).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in e]

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attention_mask
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states  # TODO: check .to(torch.bfloat16)

    def set_ar_attention(self):
        self.attn1.processor.set_ar_attention()


class Head(nn.Module):
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        with amp.autocast("cuda", dtype=torch.float32):
            if e.dim() == 2:
                modulation = self.modulation  # 1, 2, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)

            elif e.dim() == 3:
                modulation = self.modulation.unsqueeze(2)  # 1, 2, seq, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
                e = [ei.squeeze(1) for ei in e]
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class SkyReelsV2Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan-based SkyReels-V2 model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `16`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `4096`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `8192`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `32`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`str`, *optional*, defaults to `"rms_norm"`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        inject_sample_info (`bool`, defaults to `False`):
            Whether to inject sample information into the model.
        image_dim (`int`, *optional*):
            The dimension of the image embeddings.
        added_kv_proj_dim (`int`, *optional*):
            The dimension of the added key/value projection.
        rope_max_seq_len (`int`, defaults to `1024`):
            The maximum sequence length for the rotary embeddings.
        pos_embed_seq_len (`int`, *optional*):
            The sequence length for the positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 8192,
        num_layers: int = 32,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        inject_sample_info: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.num_frame_per_block = 1
        self.flag_causal_attention = False
        self.enable_teacache = False

        # 1. Patch & position embedding
        self.rope = SkyReelsV2RotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = SkyReelsV2TimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SkyReelsV2TransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim=inner_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, inner_dim)
            self.fps_projection = nn.Sequential(
                nn.Linear(inner_dim, inner_dim), nn.SiLU(), nn.Linear(inner_dim, inner_dim * 6)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        # TODO: check here
        if y is not None:
            hidden_states = torch.cat([hidden_states, y], dim=1)

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.tensor(hidden_states.shape[2:], dtype=torch.long)

        if self.flag_causal_attention:
            frame_num, height, width = grid_sizes
            block_num = frame_num // self.num_frame_per_block
            range_tensor = torch.arange(block_num, device=hidden_states.device).view(-1, 1)
            range_tensor = range_tensor.repeat(1, self.num_frame_per_block).flatten()
            causal_mask = range_tensor.unsqueeze(0) <= range_tensor.unsqueeze(1)  # f, f
            causal_mask = causal_mask.view(frame_num, 1, 1, frame_num, 1, 1)
            causal_mask = causal_mask.repeat(1, height, width, 1, height, width)
            causal_mask = causal_mask.reshape(frame_num * height * width, frame_num * height * width)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # TODO: check here
        if timestep.dim() == 2:
            b, f = timestep.shape
            _flag_df = True
        else:
            _flag_df = False

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if self.inject_sample_info:
            fps = torch.tensor(fps, dtype=torch.long, device=hidden_states.device)

            fps_emb = self.fps_embedding(fps).float()
            if _flag_df:
                timestep_proj = timestep_proj + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(
                    timestep.shape[1], 1, 1
                )
            else:
                timestep_proj = timestep_proj + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

        if _flag_df:
            temb = temb.view(b, f, 1, 1, self.dim)
            timestep_proj = timestep_proj.view(b, f, 1, 1, 6, self.dim)
            temb = temb.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1).flatten(1, 3)
            timestep_proj = timestep_proj.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).flatten(1, 3)
            timestep_proj = timestep_proj.transpose(1, 2).contiguous()

        if self.enable_teacache:
            modulated_inp = timestep_proj if self.use_ref_steps else temb
            # teacache
            if self.cnt % 2 == 0:  # even -> condition
                self.is_even = True
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_even += rescale_func(
                        ((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean())
                        .cpu()
                        .item()
                    )
                    if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                        should_calc_even = False
                    else:
                        should_calc_even = True
                        self.accumulated_rel_l1_distance_even = 0
                self.previous_e0_even = modulated_inp.clone()

            else:  # odd -> unconditon
                self.is_even = False
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_odd += rescale_func(
                        ((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean())
                        .cpu()
                        .item()
                    )
                    if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                        should_calc_odd = False
                    else:
                        should_calc_odd = True
                        self.accumulated_rel_l1_distance_odd = 0
                self.previous_e0_odd = modulated_inp.clone()

        if self.enable_teacache:
            if self.is_even:
                if not should_calc_even:
                    hidden_states += self.previous_residual_even
                else:
                    ori_hidden_states = hidden_states.clone()
                    for block in self.blocks:
                        hidden_states = block(
                            hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask
                        )
                    self.previous_residual_even = hidden_states - ori_hidden_states
            else:
                if not should_calc_odd:
                    hidden_states += self.previous_residual_odd
                else:
                    ori_hidden_states = hidden_states.clone()
                    for block in self.blocks:
                        hidden_states = block(
                            hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask
                        )
                    self.previous_residual_odd = hidden_states - ori_hidden_states

            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def set_ar_attention(self, causal_block_size):
        self.num_frame_per_block = causal_block_size
        self.flag_causal_attention = True
        for block in self.blocks:
            block.set_ar_attention()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21, frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format [1 latent frame] [1 latent frame] ... [1 latent
        frame] We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask

    def initialize_teacache(
        self, enable_teacache=True, num_steps=25, teacache_thresh=0.15, use_ret_steps=False, ckpt_dir=""
    ):
        self.enable_teacache = enable_teacache
        print("using teacache")
        self.cnt = 0
        self.num_steps = num_steps
        self.teacache_thresh = teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.use_ref_steps = use_ret_steps
        if "I2V" in ckpt_dir:
            if use_ret_steps:
                if "540P" in ckpt_dir:
                    self.coefficients = [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01]
                if "720P" in ckpt_dir:
                    self.coefficients = [8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02]
                self.ret_steps = 5 * 2
                self.cutoff_steps = num_steps * 2
            else:
                if "540P" in ckpt_dir:
                    self.coefficients = [-3.02331670e02, 2.23948934e02, -5.25463970e01, 5.87348440e00, -2.01973289e-01]
                if "720P" in ckpt_dir:
                    self.coefficients = [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
                self.ret_steps = 1 * 2
                self.cutoff_steps = num_steps * 2 - 2
        else:
            if use_ret_steps:
                if "1.3B" in ckpt_dir:
                    self.coefficients = [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02]
                if "14B" in ckpt_dir:
                    self.coefficients = [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01]
                self.ret_steps = 5 * 2
                self.cutoff_steps = num_steps * 2
            else:
                if "1.3B" in ckpt_dir:
                    self.coefficients = [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01]
                if "14B" in ckpt_dir:
                    self.coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
                self.ret_steps = 1 * 2
                self.cutoff_steps = num_steps * 2 - 2
