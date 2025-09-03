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
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm
from .transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanRotaryPosEmbed,
    WanTransformerBlock,
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


def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if type(grid_sizes) is not list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if type(g) is not list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

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
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
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
    def __init__(self, in_dim: int, hidden_dim: int, num_attention_heads=int, need_global=True):
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
        x = (
            x.unflatten(1, (self.num_attention_heads, -1))
            .permute(0, 1, 3, 2)
            .reshape(batch_size * self.num_attention_heads, seq_len, num_channels)
        )
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
            if isinstance(mod, WanTransformerBlock):
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

        head_dim = inner_dim // num_attention_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

    def forward(self, motion_latents, add_last_motion=2):
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height, lat_width).to(
                device=m.device, dtype=m.dtype
            )
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[: self.zip_frame_buckets.__len__() - add_last_motion - 1].sum()
                padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[
                :, :, -self.zip_frame_buckets.sum() :, :, :
            ].split(list(self.zip_frame_buckets)[::-1], dim=2)  # 16, 2, 1

            # patchify
            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = (
                []
                if add_last_motion < 2 and self.drop_mode == "drop"
                else [
                    [
                        torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2])
                        .unsqueeze(0)
                        .repeat(1, 1),
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
                        torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2])
                        .unsqueeze(0)
                        .repeat(1, 1),
                    ]
                ]
            )

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2])
                    .unsqueeze(0)
                    .repeat(1, 1),
                ]
            ]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None,
            )

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


class WanTimeTextAudioPoseEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        audio_embed_dim: int,
        enable_adain: bool = True,
        pose_embed_dim: Optional[int] = None,
        patch_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_embed_dim, out_dim=dim, num_audio_token=4, need_global=enable_adain
        )

        self.pose_embedder = None
        if pose_embed_dim is not None:
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

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        audio_hidden_states = self.casual_audio_encoder(audio_hidden_states)

        if self.pose_embedder is not None:
            pose_hidden_states = self.pose_embedder(pose_hidden_states)

        return temb, timestep_proj, encoder_hidden_states, audio_hidden_states, pose_hidden_states


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
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

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
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        enable_framepack: bool = False,
        framepack_drop_mode: str = "padd",
        add_last_motion: bool = False,
        zero_timestep: bool = True,
    ) -> None:
        super().__init__()

        self.add_last_motion = add_last_motion
        self.inner_dim = inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=inner_dim,
                num_attention_heads=num_attention_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode,
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
                    inner_dim, ffn_dim, num_attention_heads, cross_attn_norm, eps, added_kv_proj_dim
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
                    torch.tensor([-self.latent_motion_frames, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([0, height, width]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([self.latent_motion_frames, height, width]).unsqueeze(0).repeat(1, 1),
                ]
            ]
            motion_rope_emb = rope_precompute(
                flat_mot.detach().view(
                    1, flat_mot.shape[1], self.num_attention_heads, self.inner_dim // self.num_attention_heads
                ),
                motion_grid_sizes,
                self.freqs,
                start=None,
            )
            mot_remb.append(motion_rope_emb)
            flattern_mot.append(flat_mot)
        return flattern_mot, mot_remb

    def process_motion_frame_pack(self, motion_latents, drop_motion_frames=False, add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
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
        if self.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(motion_latents, drop_motion_frames, add_last_motion)
        else:
            mot, mot_remb = self.process_motion(motion_latents, drop_motion_frames)

        if len(mot) > 0:
            hidden_states = [torch.cat([u, m], dim=1) for u, m in zip(hidden_states, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot], dtype=torch.long)
            rope_embs = [torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
            mask_input = [
                torch.cat([m, 2 * torch.ones([1, u.shape[1] - m.shape[1]], device=m.device, dtype=m.dtype)], dim=1)
                for m, u in zip(mask_input, hidden_states)
            ]
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

            if self.enbale_adain and self.adain_mode == "attn_norm":
                attn_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](
                    input_hidden_states, temb=audio_emb_global[:, 0]
                )
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)

            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=torch.ones(
                    attn_hidden_states.shape[0], dtype=torch.long, device=attn_hidden_states.device
                )
                * attn_audio_emb.shape[1],
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
        image_latents: torch.Tensor = None,
        pose_latents: torch.Tensor = None,
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
        add_last_motion = self.add_last_motion * add_last_motion

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)

        # 3. Condition embeddings
        audio_embeds = torch.cat([audio_embeds[..., 0].repeat(1, 1, 1, motion_frames[0]), audio_embeds], dim=-1)

        if self.config.zero_timestep:
            timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])

        temb, timestep_proj, encoder_hidden_states, audio_hidden_states, pose_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, audio_embeds, pose_latents
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if self.enable_adain:
            audio_emb_global, audio_emb = audio_hidden_states
            audio_emb_global = audio_emb_global[:, motion_frames[1] :].clone()
        else:
            audio_emb = audio_hidden_states
        merged_audio_emb = audio_emb[:, motion_frames[1] :, :]

        hidden_states = hidden_states + pose_hidden_states
        grid_sizes = torch.tensor(
            [post_patch_num_frames, post_patch_height, post_patch_width], dtype=torch.long
        ).unsqueeze(0)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        sequence_length = hidden_states.shape[1].to(torch.long)
        original_sequence_length = sequence_length

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

        image_latents = self.patch_embedding(image_latents)
        image_latents = image_latents.flatten(2).transpose(1, 2)
        image_grid_sizes = [
            [
                # The start index
                torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                # The end index
                torch.tensor([31, height, width]).unsqueeze(0).repeat(batch_size, 1),
                # The range
                torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
            ]
        ]

        sequence_length = sequence_length + image_latents.shape[1].to(torch.long)
        grid_sizes = grid_sizes + image_grid_sizes

        hidden_states = torch.cat([hidden_states, image_latents], dim=1)

        # Initialize masks to indicate noisy latent, image latent, and motion latent.
        # However, at this point, only the first two (noisy and image latents) are marked;
        # the marking of motion latent will be implemented inside `inject_motion`.
        mask_input = torch.zeros([1, hidden_states.shape[1]], dtype=torch.long, device=hidden_states.device)
        mask_input[:, original_sequence_length:] = 1

        # Rotary position embedding
        rotary_emb = self.rope(hidden_states)

        hidden_states, sequence_length, pre_compute_freqs, mask_input = self.inject_motion(
            hidden_states,
            sequence_length,
            # pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames,
            add_last_motion,
        )

        hidden_states = hidden_states + self.trainable_condition_mask(mask_input).to(hidden_states.dtype)

        merged_audio_emb_num_frames = merged_audio_emb.shape[1]  # B F N C
        attn_audio_emb = merged_audio_emb.flatten(0, 1)
        audio_emb_global = audio_emb_global.flatten(0, 1)

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
