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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import FeedForward
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm
from .transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanRotaryPosEmbed,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CausalConv1d(nn.Module):

    def __init__(self,
                 chan_in,
                 chan_out,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 pad_mode='replicate',
                 **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_heads=int,
                 need_global=True,
                 dtype=None,
                 device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(
            in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(
                in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(
            hidden_dim // 4,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim,
                                          **factory_kwargs)

        self.norm1 = nn.LayerNorm(
            hidden_dim // 4,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)

        self.norm2 = nn.LayerNorm(
            hidden_dim // 2,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)

        self.norm3 = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)

        return x, x_local


class CausalAudioEncoder(nn.Module):

    def __init__(self,
                 dim=5120,
                 num_layers=25,
                 out_dim=2048,
                 video_rate=8,
                 num_audio_token=4,
                 need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_audio_token,
            need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        with amp.autocast(dtype=torch.float32):
            # features B * num_layers * dim * video_length
            weights = self.act(self.weights)
            weights_sum = weights.sum(dim=1, keepdims=True)
            weighted_feat = ((features * weights) / weights_sum).sum(
                dim=1)  # b dim f
            weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
            res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioInjector(nn.Module):

    def __init__(self,
                 all_modules,
                 all_modules_names,
                 dim=2048,
                 num_heads=32,
                 inject_layer=[0, 27],
                 root_net=None,
                 enable_adain=False,
                 adain_dim=2048,
                 need_adain_ont=False):
        super().__init__()
        num_injector_layers = len(inject_layer)
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([
            AudioCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=True,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(
                    output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1)
                for _ in range(audio_injector_id)
            ])
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)])



class MotionerTransformers(nn.Module, PeftAdapterMixin):

    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        self_attn_block="SelfAttention",
        motion_token_num=1024,
        enable_tsm=False,
        motion_stride=4,
        expand_ratio=2,
        trainable_token_pos_emb=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.enable_tsm = enable_tsm
        self.motion_stride = motion_stride
        self.expand_ratio = expand_ratio
        self.sample_c = self.patch_size[0]

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # blocks
        self.blocks = nn.ModuleList([
            MotionerAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
                self_attn_block=self_attn_block) for _ in range(num_layers)
        ])

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        self.gradient_checkpointing = False

        self.motion_side_len = int(math.sqrt(motion_token_num))
        assert self.motion_side_len**2 == motion_token_num
        self.token = nn.Parameter(
            torch.zeros(1, motion_token_num, dim).contiguous())

        self.trainable_token_pos_emb = trainable_token_pos_emb
        if trainable_token_pos_emb:
            x = torch.zeros([1, motion_token_num, num_heads, d])
            x[..., ::2] = 1

            gride_sizes = [[
                torch.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([1, self.motion_side_len,
                              self.motion_side_len]).unsqueeze(0).repeat(1, 1),
                torch.tensor([1, self.motion_side_len,
                              self.motion_side_len]).unsqueeze(0).repeat(1, 1),
            ]]
            token_freqs = rope_apply(x, gride_sizes, self.freqs)
            token_freqs = token_freqs[0, :, 0].reshape(motion_token_num, -1, 2)
            token_freqs = token_freqs * 0.01
            self.token_freqs = torch.nn.Parameter(token_freqs)

    def after_patch_embedding(self, x):
        return x

    def forward(
        self,
        x,
    ):
        """
        x:              A list of videos each with shape [C, T, H, W].
        t:              [B].
        context:        A list of text embeddings each with shape [L, C].
        """
        # params
        motion_frames = x[0].shape[1]
        device = self.patch_embedding.weight.device
        freqs = self.freqs
        if freqs.device != device:
            freqs = freqs.to(device)

        if self.trainable_token_pos_emb:
            with amp.autocast(dtype=torch.float64):
                token_freqs = self.token_freqs.to(torch.float64)
                token_freqs = token_freqs / token_freqs.norm(
                    dim=-1, keepdim=True)
                freqs = [freqs, torch.view_as_complex(token_freqs)]

        if self.enable_tsm:
            sample_idx = [
                sample_indices(
                    u.shape[1],
                    stride=self.motion_stride,
                    expand_ratio=self.expand_ratio,
                    c=self.sample_c) for u in x
            ]
            x = [
                torch.flip(torch.flip(u, [1])[:, idx], [1])
                for idx, u in zip(sample_idx, x)
            ]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = self.after_patch_embedding(x)

        seq_f, seq_h, seq_w = x[0].shape[-3:]
        batch_size = len(x)
        if not self.enable_tsm:
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            grid_sizes = [[
                torch.zeros_like(grid_sizes), grid_sizes, grid_sizes
            ]]
            seq_f = 0
        else:
            grid_sizes = []
            for idx in sample_idx[0][::-1][::self.sample_c]:
                tsm_frame_grid_sizes = [[
                    torch.tensor([idx, 0,
                                  0]).unsqueeze(0).repeat(batch_size, 1),
                    torch.tensor([idx + 1, seq_h,
                                  seq_w]).unsqueeze(0).repeat(batch_size, 1),
                    torch.tensor([1, seq_h,
                                  seq_w]).unsqueeze(0).repeat(batch_size, 1),
                ]]
                grid_sizes += tsm_frame_grid_sizes
            seq_f = sample_idx[0][-1] + 1

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat([u for u in x])

        batch_size = len(x)

        token_grid_sizes = [[
            torch.tensor([seq_f, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor(
                [seq_f + 1, self.motion_side_len,
                 self.motion_side_len]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor(
                [1 if not self.trainable_token_pos_emb else -1, seq_h,
                 seq_w]).unsqueeze(0).repeat(batch_size, 1),
        ]  # 第三行代表rope emb的想要覆盖到的范围
                           ]

        grid_sizes = grid_sizes + token_grid_sizes
        token_unpatch_grid_sizes = torch.stack([
            torch.tensor([1, 32, 32], dtype=torch.long)
            for b in range(batch_size)
        ])
        token_len = self.token.shape[1]
        token = self.token.clone().repeat(x.shape[0], 1, 1).contiguous()
        seq_lens = seq_lens + torch.tensor([t.size(0) for t in token],
                                           dtype=torch.long)
        x = torch.cat([x, token], dim=1)
        # arguments
        kwargs = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )

        # grad ckpt args
        def create_custom_forward(module, return_dict=None):

            def custom_forward(*inputs, **kwargs):
                if return_dict is not None:
                    return module(*inputs, **kwargs, return_dict=return_dict)
                else:
                    return module(*inputs, **kwargs)

            return custom_forward

        ckpt_kwargs: Dict[str, Any] = ({
            "use_reentrant": False
        } if is_torch_version(">=", "1.11.0") else {})

        for idx, block in enumerate(self.blocks):
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    **ckpt_kwargs,
                )
            else:
                x = block(x, **kwargs)
        # head
        out = x[:, -token_len:]
        return out

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))


class FramePackMotioner(nn.Module):

    def __init__(
            self,
            inner_dim=1024,
            num_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
            zip_frame_buckets=[
                1, 2, 16
            ],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
            drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Conv3d(
            16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(
            16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(
            16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(
            zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        assert (inner_dim %
                num_heads) == 0 and (inner_dim // num_heads) % 2 == 0
        d = inner_dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2):
        motion_frames = motion_latents[0].shape[1]
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height,
                                   lat_width).to(
                                       device=m.device, dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[:self.zip_frame_buckets.
                                                        __len__() -
                                                        add_last_motion -
                                                        1].sum()
                padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum(
            ):, :, :].split(
                list(self.zip_frame_buckets)[::-1], dim=2)  # 16, 2 ,1

            # patchfy
            clean_latents_post = self.proj(clean_latents_post).flatten(
                2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(
                2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(
                2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :
                                                        0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :
                                                    0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = torch.cat(
                [clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else \
                        [
                            [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
                        ]

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else \
            [
                [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
            ]

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [[
                torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 8,
                              lat_width // 8]).unsqueeze(0).repeat(1, 1),
                torch.tensor([
                    self.zip_frame_buckets[2], lat_height // 2, lat_width // 2
                ]).unsqueeze(0).repeat(1, 1),
            ]]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads,
                                         self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None)

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
        self.casual_audio_encoder = CausalAudioEncoder(dim=audio_embed_dim, out_dim=dim, num_audio_token=4, need_global=enable_adain)

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
        apply_input_projection: bool = False,
        apply_output_projection: bool = False,
    ):
        super().__init__()

        # 1. Input projection
        self.proj_in = None
        if apply_input_projection:
            self.proj_in = nn.Linear(dim, dim)

        # 2. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=WanAttnProcessor(),
        )

        # 3. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. Output projection
        self.proj_out = None
        if apply_output_projection:
            self.proj_out = nn.Linear(dim, dim)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(control_hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            control_hidden_states
        )
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        control_hidden_states = (control_hidden_states.float() + attn_output * gate_msa).type_as(control_hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(control_hidden_states.float()).type_as(control_hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        control_hidden_states = control_hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(control_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            control_hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        control_hidden_states = (control_hidden_states.float() + ff_output.float() * c_gate_msa).type_as(
            control_hidden_states
        )

        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states = self.proj_out(control_hidden_states)

        return conditioning_states, control_hidden_states


class WanS2VTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

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
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanS2VTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
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
        enable_adain: bool = True,
        pose_dim: int = 1280,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        enable_motioner: bool = False,
        enable_framepack: bool = False,
        add_last_motion: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # init motioner
        if enable_motioner and enable_framepack:
            raise ValueError(
                "enable_motioner and enable_framepack are mutually exclusive, please set one of them to False"
            )
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        if enable_motioner:
            motioner_dim = 2048
            self.motioner = MotionerTransformers(
                patch_size=(2, 4, 4),
                dim=motioner_dim,
                ffn_dim=motioner_dim,
                freq_dim=256,
                out_dim=16,
                num_heads=16,
                num_layers=13,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=False,
                eps=1e-6,
                motion_token_num=motion_token_num,
                enable_tsm=enable_tsm,
                motion_stride=4,
                expand_ratio=2,
            )
            self.zip_motion_out = torch.nn.Sequential(
                WanLayerNorm(motioner_dim),
                zero_module(nn.Linear(motioner_dim, self.dim)))

        self.enable_framepack = enable_framepack
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode)

        self.trainable_cond_mask = nn.Embedding(3, self.dim)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
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

        self.audio_injector = AudioInjector(
            all_modules,
            all_modules_names,
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            need_adain_ont=adain_mode != "attn_norm",
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def inject_motion(self,
                      x,
                      seq_lens,
                      rope_embs,
                      mask_input,
                      motion_latents,
                      drop_motion_frames=False,
                      add_last_motion=True):
        # inject the motion frames token to the hidden states
        if self.enable_motioner:
            mot, mot_remb = self.process_motion_transformer_motioner(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion)
        elif self.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion)
        else:
            mot, mot_remb = self.process_motion(
                motion_latents, drop_motion_frames=drop_motion_frames)

        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot],
                                               dtype=torch.long)
            rope_embs = [
                torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)
            ]
            mask_input = [
                torch.cat([
                    m, 2 * torch.ones([1, u.shape[1] - m.shape[1]],
                                      device=m.device,
                                      dtype=m.dtype)
                ],
                          dim=1) for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        motion_latents: torch.Tensor,
        image_latents: torch.Tensor = None,
        pose_latents: torch.Tensor = None,
        audio_embeds: torch.Tensor = None,
        motion_frames: List[int] = None,
        drop_motion_frames: bool = False,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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

        # 1. Rotary position embedding
        rotary_emb = self.rope(hidden_states)

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 3. Time embedding
        temb, timestep_proj, encoder_hidden_states, audio_hidden_states, pose_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, audio_embeds, pose_latents
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Image embedding

        # 5. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for i, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

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
