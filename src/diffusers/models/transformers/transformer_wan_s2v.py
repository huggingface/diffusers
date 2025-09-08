# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import FP32LayerNorm
from .transformer_wan import (
    WanAttention,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def torch_dfs(model: nn.Module, parent_name="root"):
    module_names, modules = [], []
    current_name = parent_name if parent_name else "root"
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f"{parent_name}.{name}"
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                # dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                n = hidden_states.size(2)
                # loop over samples
                output = []
                for i in range(hidden_states.size(0)):
                    s = hidden_states.size(1)
                    x_i = torch.view_as_complex(hidden_states[i, :s].to(torch.float64).reshape(s, n, -1, 2))
                    freqs_i = freqs[i, :s]
                    # apply rotary embedding
                    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
                    x_i = torch.cat([x_i, hidden_states[i, s:]])
                    # append to collection
                    output.append(x_i)
                return torch.stack(output).type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        output_dim (`int`, *optional*): Output dimension for the layer.
        norm_elementwise_affine (`bool`, defaults to `False`): Whether to use elementwise affine in LayerNorm.
        norm_eps (`float`, defaults to `1e-5`): Epsilon value for LayerNorm.
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        output_dim = output_dim or embedding_dim * 2

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.linear(self.silu(temb))

        shift, scale = temb.chunk(2, dim=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]

        x = self.norm(x) * (1 + scale) + shift
        return x


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)  # T

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_attention_heads: int, need_global: bool = True):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_attention_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        residual = x.clone()
        batch_size, num_channels, seq_len = x.shape
        x = self.conv1_local(x)
        x = x.unflatten(1, (self.num_attention_heads, -1)).permute(0, 1, 3, 2).flatten(0, 1)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)
        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(residual)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)

        return x, x_local


class CausalAudioEncoder(nn.Module):
    def __init__(self, dim=5120, num_layers=25, out_dim=2048, num_audio_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim, hidden_dim=out_dim, num_attention_heads=num_audio_token, need_global=need_global
        )
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length
        weights = self.act(self.weights)
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioInjector(nn.Module):
    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=[0, 27],
        enable_adain=False,
        adain_dim=2048,
        need_adain_ont=False,
        eps=1e-6,
        added_kv_proj_dim=None,
    ):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanS2VTransformerBlock):
                for inject_id in inject_layer:
                    if f"transformer_blocks.{inject_id}" in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        # Cross-attention
        self.injector = nn.ModuleList(
            [
                WanAttention(
                    dim=dim,
                    heads=num_heads,
                    dim_head=dim // num_heads,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    cross_attention_dim_head=dim // num_heads,
                    processor=WanAttnProcessor(),
                )
                for _ in range(audio_injector_id)
            ]
        )

        self.injector_pre_norm_feat = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(audio_injector_id)]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(audio_injector_id)]
        )

        if enable_adain:
            self.injector_adain_layers = nn.ModuleList(
                [AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim) for _ in range(audio_injector_id)]
            )
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)]
                )


class FramePackMotioner(nn.Module):
    def __init__(
        self,
        inner_dim=1024,
        num_attention_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
        zip_frame_buckets=[
            1,
            2,
            16,
        ],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
        drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
        patch_size=(1, 2, 2),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.inner_dim = inner_dim
        self.num_attention_heads = num_attention_heads
        if (inner_dim % num_attention_heads) != 0 or (inner_dim // num_attention_heads) % 2 != 0:
            raise ValueError(
                "inner_dim must be divisible by num_attention_heads and inner_dim // num_attention_heads must be even"
            )
        self.drop_mode = drop_mode

        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)

        self.rope = WanS2VRotaryPosEmbed(
            inner_dim // num_attention_heads,
            patch_size=patch_size,
            max_seq_len=1024,
            num_attention_heads=num_attention_heads,
        )

    def forward(self, motion_latents, add_last_motion=2):
        latent_height, latent_width = motion_latents.shape[3], motion_latents.shape[4]
        padd_latent = torch.zeros(
            (motion_latents.shape[0], 16, self.zip_frame_buckets.sum(), latent_height, latent_width),
            device=motion_latents.device,
            dtype=motion_latents.dtype,
        )
        overlap_frame = min(padd_latent.shape[2], motion_latents.shape[2])
        if overlap_frame > 0:
            padd_latent[:, :, -overlap_frame:] = motion_latents[:, :, -overlap_frame:]

        if add_last_motion < 2 and self.drop_mode != "drop":
            zero_end_frame = self.zip_frame_buckets[: len(self.zip_frame_buckets) - add_last_motion - 1].sum()
            padd_latent[:, :, -zero_end_frame:] = 0

        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_latent[
            :, :, -self.zip_frame_buckets.sum() :, :, :
        ].split(list(self.zip_frame_buckets)[::-1], dim=2)  # 16, 2, 1

        # Patchify
        clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
        clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
        clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

        if add_last_motion < 2 and self.drop_mode == "drop":
            clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
            clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

        motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        # RoPE
        start_time_id = -(self.zip_frame_buckets[:1].sum())
        end_time_id = start_time_id + self.zip_frame_buckets[0]
        grid_sizes = (
            []
            if add_last_motion < 2 and self.drop_mode == "drop"
            else [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0),
                    torch.tensor([end_time_id, latent_height // 2, latent_width // 2]).unsqueeze(0),
                    torch.tensor([self.zip_frame_buckets[0], latent_height // 2, latent_width // 2]).unsqueeze(0),
                ]
            ]
        )

        start_time_id = -(self.zip_frame_buckets[:2].sum())
        end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
        grid_sizes_2x = (
            []
            if add_last_motion < 1 and self.drop_mode == "drop"
            else [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0),
                    torch.tensor([end_time_id, latent_height // 4, latent_width // 4]).unsqueeze(0),
                    torch.tensor([self.zip_frame_buckets[1], latent_height // 2, latent_width // 2]).unsqueeze(0),
                ]
            ]
        )

        start_time_id = -(self.zip_frame_buckets[:3].sum())
        end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
        grid_sizes_4x = [
            [
                torch.tensor([start_time_id, 0, 0]).unsqueeze(0),
                torch.tensor([end_time_id, latent_height // 8, latent_width // 8]).unsqueeze(0),
                torch.tensor([self.zip_frame_buckets[2], latent_height // 2, latent_width // 2]).unsqueeze(0),
            ]
        ]

        grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

        motion_rope_emb = self.rope(
            motion_lat.detach().view(
                motion_lat.shape[0],
                motion_lat.shape[1],
                self.num_attention_heads,
                self.inner_dim // self.num_attention_heads,
            ),
            grid_sizes=grid_sizes,
        )

        return motion_lat, motion_rope_emb


class WanTimeTextAudioPoseEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        audio_embed_dim: int,
        pose_embed_dim: int,
        patch_size: Tuple[int],
        enable_adain: bool,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.causal_audio_encoder = CausalAudioEncoder(
            dim=audio_embed_dim, out_dim=dim, num_audio_token=4, need_global=enable_adain
        )
        self.pose_embedder = nn.Conv3d(pose_embed_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        pose_hidden_states: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = get_parameter_dtype(self.time_embedder)
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        audio_hidden_states = self.causal_audio_encoder(audio_hidden_states)

        pose_hidden_states = self.pose_embedder(pose_hidden_states)

        return temb, timestep_proj, encoder_hidden_states, audio_hidden_states, pose_hidden_states


class WanS2VRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        num_attention_heads: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.num_attention_heads = num_attention_heads

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs = []

        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=freqs_dtype
            )
            freqs.append(freq)

        self.freqs = torch.cat(freqs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
        grid_sizes: Optional[List[List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if grid_sizes is None:
            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.patch_size
            ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

            grid_sizes = torch.tensor([ppf, pph, ppw]).unsqueeze(0).repeat(batch_size, 1)
            grid_sizes = [torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]

            image_grid_sizes = [
                # The start index
                torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                # The end index
                torch.tensor([31, image_latents.shape[3] // p_h, image_latents.shape[4] // p_w])
                .unsqueeze(0)
                .repeat(batch_size, 1),
                # The range
                torch.tensor([1, image_latents.shape[3] // p_h, image_latents.shape[4] // p_w])
                .unsqueeze(0)
                .repeat(batch_size, 1),
            ]

            grids = [grid_sizes, image_grid_sizes]
            S = ppf * pph * ppw + image_latents.shape[3] // p_h * image_latents.shape[4] // p_w
        else:  # FramePack's RoPE
            batch_size, S, _, _ = hidden_states.shape
            grids = grid_sizes

        split_sizes = [
            self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
            self.attention_head_dim // 6,
            self.attention_head_dim // 6,
        ]

        freqs = self.freqs.split(split_sizes, dim=1)

        # Loop over samples
        output = torch.view_as_complex(
            torch.zeros(
                (batch_size, S, self.num_attention_heads, self.attention_head_dim // 2, 2),
                device=hidden_states.device,
                dtype=torch.float64,
            )
        )
        seq_bucket = [0]
        for g in grids:
            if type(g) is not list:
                g = [torch.zeros_like(g), g]
            batch_size = g[0].shape[0]
            for i in range(batch_size):
                f_o, h_o, w_o = g[0][i]

                f, h, w = g[1][i]
                t_f, t_h, t_w = g[2][i]
                seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
                seq_len = int(seq_f * seq_h * seq_w)
                if seq_len > 0:
                    if t_f > 0:
                        # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                        if f_o >= 0:
                            f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                        else:
                            f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                        h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                        w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                        assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                        freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                        freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                        freqs_i = torch.cat(
                            [
                                freqs_0.expand(seq_f, seq_h, seq_w, -1),
                                freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                                freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                            ],
                            dim=-1,
                        ).reshape(seq_len, 1, -1)

                    # apply rotary embedding
                    output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
            seq_bucket.append(seq_bucket[-1] + seq_len)

        return output


@maybe_allow_in_graph
class WanS2VTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
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
        temb: Tuple[torch.Tensor, torch.Tensor],
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        seg_idx = temb[1].item()
        seg_idx = min(max(0, seg_idx), hidden_states.shape[1])
        seg_idx = [0, seg_idx, hidden_states.shape[1]]
        temb = temb[0]
        # temb: batch_size, 6, 2, inner_dim
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(2) + temb.float()
        ).chunk(6, dim=1)
        # batch_size, 1, seq_len, inner_dim
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)

        norm_hidden_states = self.norm1(hidden_states.float())
        parts = []
        for i in range(2):
            parts.append(
                norm_hidden_states[:, seg_idx[i] : seg_idx[i + 1]] * (1 + scale_msa[:, i : i + 1])
                + shift_msa[:, i : i + 1]
            )
        norm_hidden_states = torch.cat(parts, dim=1).type_as(hidden_states)

        # 1. Self-attention
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        z = []
        for i in range(2):
            z.append(attn_output[:, seg_idx[i] : seg_idx[i + 1]] * gate_msa[:, i : i + 1])
        attn_output = torch.cat(z, dim=1)
        hidden_states = (hidden_states.float() + attn_output).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm3_hidden_states = self.norm3(hidden_states.float())
        parts = []
        for i in range(2):
            parts.append(
                norm3_hidden_states[:, seg_idx[i] : seg_idx[i + 1]] * (1 + c_scale_msa[:, i : i + 1])
                + c_shift_msa[:, i : i + 1]
            )
        norm3_hidden_states = torch.cat(parts, dim=1).type_as(hidden_states)
        ff_output = self.ffn(norm3_hidden_states)
        z = []
        for i in range(2):
            z.append(ff_output[:, seg_idx[i] : seg_idx[i + 1]] * c_gate_msa[:, i : i + 1])
        ff_output = torch.cat(z, dim=1)
        hidden_states = (hidden_states.float() + ff_output.float()).type_as(hidden_states)

        return hidden_states


class WanS2VTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan2.2-S2V model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        zero_timestep (`bool`, defaults to `True`):
            Whether to assign 0 value timestep to image/motion
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanS2VTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3", "causal_audio_encoder"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanS2VTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        audio_dim: int = 1280,
        audio_inject_layers: List[int] = [0, 4, 8, 12, 16, 20, 24, 27],
        enable_adain: bool = True,
        adain_mode: str = "attn_norm",
        pose_dim: int = 1280,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        enable_framepack: bool = False,
        framepack_drop_mode: str = "padd",
        add_last_motion: bool = False,
        zero_timestep: bool = True,
    ) -> None:
        super().__init__()

        self.inner_dim = inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanS2VRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len, num_attention_heads)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=inner_dim,
                num_attention_heads=num_attention_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode,
                patch_size=patch_size,
            )

        self.trainable_condition_mask = nn.Embedding(3, inner_dim)

        # 2. Condition Embeddings
        self.condition_embedder = WanTimeTextAudioPoseEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            audio_embed_dim=audio_dim,
            pose_embed_dim=pose_dim,
            patch_size=patch_size,
            enable_adain=enable_adain,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanS2VTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Audio Injector
        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector(
            all_modules,
            all_modules_names,
            dim=inner_dim,
            num_heads=num_attention_heads,
            inject_layer=audio_inject_layers,
            enable_adain=enable_adain,
            adain_dim=inner_dim,
            need_adain_ont=adain_mode != "attn_norm",
            eps=eps,
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def process_motion(self, motion_latents, drop_motion_frames=False):
        if drop_motion_frames or motion_latents[0].shape[1] == 0:
            return [], []
        self.latent_motion_frames = motion_latents[0].shape[1]
        mot = [self.patch_embedding(m.unsqueeze(0)) for m in motion_latents]
        batch_size = len(mot)

        mot_remb = []
        flattern_mot = []
        for bs in range(batch_size):
            height, width = mot[bs].shape[3], mot[bs].shape[4]
            flat_mot = mot[bs].flatten(2).transpose(1, 2).contiguous()
            motion_grid_sizes = [
                [
                    torch.tensor([-self.latent_motion_frames, 0, 0]).unsqueeze(0),
                    torch.tensor([0, height, width]).unsqueeze(0),
                    torch.tensor([self.latent_motion_frames, height, width]).unsqueeze(0),
                ]
            ]
            motion_rope_emb = self.rope(
                flat_mot.detach().view(
                    1,
                    flat_mot.shape[1],
                    self.config.num_attention_heads,
                    self.inner_dim // self.config.num_attention_heads,
                ),
                motion_grid_sizes,
            )
            mot_remb.append(motion_rope_emb)
            flattern_mot.append(flat_mot)
        return flattern_mot, mot_remb

    def process_motion_frame_pack(self, motion_latents, drop_motion_frames=False, add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)

        if drop_motion_frames:
            return flattern_mot[:,:,:0], mot_remb[:,:,:0]
        else:
            return flattern_mot, mot_remb

    def inject_motion(
        self,
        hidden_states,
        seq_lens,
        rope_embs,
        mask_input,
        motion_latents,
        drop_motion_frames=False,
        add_last_motion=True,
    ):
        # Inject the motion frames token to the hidden states
        if self.config.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(motion_latents, drop_motion_frames, add_last_motion)
        else:
            mot, mot_remb = self.process_motion(motion_latents, drop_motion_frames)

        if len(mot) > 0:
            hidden_states = torch.cat([hidden_states, mot], dim=1)
            seq_lens = seq_lens + torch.tensor([mot.shape[1]], dtype=torch.long)
            rope_embs = torch.cat([rope_embs, mot_remb], dim=1)
            mask_input = torch.cat(
                [
                    mask_input,
                    2
                    * torch.ones(
                        [1, hidden_states.shape[1] - mask_input.shape[1]],
                        device=mask_input.device,
                        dtype=mask_input.dtype,
                    ),
                ],
                dim=1,
            )
        return hidden_states, seq_lens, rope_embs, mask_input

    def after_transformer_block(
        self,
        block_idx,
        hidden_states,
        original_sequence_length,
        merged_audio_emb_num_frames,
        attn_audio_emb,
        audio_emb_global,
    ):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]

            input_hidden_states = hidden_states[:, :original_sequence_length].clone()  # B (F H W) C
            input_hidden_states = input_hidden_states.unflatten(1, (merged_audio_emb_num_frames, -1)).flatten(0, 1)

            if self.config.enable_adain and self.config.adain_mode == "attn_norm":
                attn_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](
                    input_hidden_states, temb=audio_emb_global[:, 0]
                )
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)

            residual_out = self.audio_injector.injector[audio_attn_id](
                attn_hidden_states,
                attn_audio_emb,
            )
            residual_out = residual_out.unflatten(0, (-1, merged_audio_emb_num_frames)).flatten(1, 2)
            hidden_states[:, :original_sequence_length] = hidden_states[:, :original_sequence_length] + residual_out

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        motion_latents: torch.Tensor,
        audio_embeds: torch.Tensor,
        image_latents: torch.Tensor,
        pose_latents: torch.Tensor,
        motion_frames: List[int] = [17, 5],
        drop_motion_frames: bool = False,
        add_last_motion: int = 2,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Parameters:
            audio_embeds:
                The input audio embedding [B, num_wav2vec_layer, C_a, T_a].
            motion_frames:
                The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5].
            add_last_motion:
                For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame)
                will be added. For frame packing, the behavior depends on the value of add_last_motion: add_last_motion
                = 0: Only the farthest part of the latent (i.e., clean_latents_4x) is included. add_last_motion = 1:
                Both clean_latents_2x and clean_latents_4x are included. add_last_motion = 2: All motion-related
                latents are used.
            drop_motion_frames:
                Bool, whether drop the motion frames info.
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
        add_last_motion = self.config.add_last_motion * add_last_motion

        # 1. Rotary position embeddings
        rotary_emb = self.rope(hidden_states, image_latents)

        # 2. Patch embeddings
        hidden_states = self.patch_embedding(hidden_states)
        image_latents = self.patch_embedding(image_latents)

        # 3. Condition embeddings
        audio_embeds = torch.cat(
            [audio_embeds[..., 0].unsqueeze(-1).repeat(1, 1, 1, motion_frames[0]), audio_embeds], dim=-1
        )

        if self.config.zero_timestep:
            timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])

        temb, timestep_proj, encoder_hidden_states, audio_hidden_states, pose_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, audio_embeds, pose_latents
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if self.config.enable_adain:
            audio_emb_global, audio_emb = audio_hidden_states
            audio_emb_global = audio_emb_global[:, motion_frames[1] :].clone()
        else:
            audio_emb = audio_hidden_states
        merged_audio_emb = audio_emb[:, motion_frames[1] :, :]

        hidden_states = hidden_states + pose_hidden_states

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        image_latents = image_latents.flatten(2).transpose(1, 2)

        sequence_length = torch.tensor([hidden_states.shape[1]], dtype=torch.long)
        original_sequence_length = sequence_length
        sequence_length = sequence_length + torch.tensor([image_latents.shape[1]], dtype=torch.long)
        hidden_states = torch.cat([hidden_states, image_latents], dim=1)

        # Initialize masks to indicate noisy latent, image latent, and motion latent.
        # However, at this point, only the first two (noisy and image latents) are marked;
        # the marking of motion latent will be implemented inside `inject_motion`.
        mask_input = (
            torch.zeros([1, hidden_states.shape[1]], dtype=torch.long, device=hidden_states.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        mask_input[:, :, original_sequence_length:] = 1

        hidden_states, sequence_length, rotary_emb, mask_input = self.inject_motion(
            hidden_states,
            sequence_length,
            rotary_emb,
            mask_input,
            motion_latents,
            drop_motion_frames,
            add_last_motion,
        )

        hidden_states = torch.cat(hidden_states)
        rotary_emb = torch.cat(rotary_emb)
        mask_input = torch.cat(mask_input)

        hidden_states = hidden_states + self.trainable_condition_mask(mask_input).to(hidden_states.dtype)

        if self.config.zero_timestep:
            temb = temb[:-1]
            zero_timestep_proj = timestep_proj[-1:]
            timestep_proj = timestep_proj[:-1]
            timestep_proj = torch.cat(
                [timestep_proj.unsqueeze(2), zero_timestep_proj.unsqueeze(2).repeat(timestep_proj.shape[0], 1, 1, 1)],
                dim=2,
            )
            timestep_proj = [timestep_proj, original_sequence_length]
        else:
            timestep_proj = timestep_proj.unsqueeze(2).repeat(1, 1, 2, 1)
            timestep_proj = [timestep_proj, 0]

        merged_audio_emb_num_frames = merged_audio_emb.shape[1]  # B F N C
        attn_audio_emb = merged_audio_emb.flatten(0, 1).to(hidden_states.dtype)
        audio_emb_global = audio_emb_global.flatten(0, 1).to(hidden_states.dtype)

        # 5. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                hidden_states = self.after_transformer_block(
                    block_idx,
                    hidden_states,
                    original_sequence_length,
                    merged_audio_emb_num_frames,
                    attn_audio_emb,
                    audio_emb_global,
                )
        else:
            for block_idx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                hidden_states = self.after_transformer_block(
                    block_idx,
                    hidden_states,
                    original_sequence_length,
                    merged_audio_emb_num_frames,
                    attn_audio_emb,
                    audio_emb_global,
                )

        hidden_states = hidden_states[:, :original_sequence_length]

        # 6. Output norm, projection & unpatchify
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
