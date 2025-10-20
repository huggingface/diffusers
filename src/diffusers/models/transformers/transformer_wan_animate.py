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
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import AttentionMixin
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin, get_parameter_dtype
from ..normalization import FP32LayerNorm
from .transformer_wan import (
    WanImageEmbedding,
    WanRotaryPosEmbed,
    WanTransformerBlock,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        downsample: bool = False,
        bias: bool = True,
        activate: bool = True,
    ):
        super().__init__()

        self.downsample = downsample
        self.activate = activate

        if activate:
            self.act = nn.LeakyReLU(0.2)
            self.bias_leaky_relu = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

        if downsample:
            factor = 2
            blur_kernel = (1, 3, 3, 1)
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            # Create blur kernel
            blur_kernel_tensor = torch.tensor(blur_kernel, dtype=torch.float32)
            blur_kernel_2d = blur_kernel_tensor[None, :] * blur_kernel_tensor[:, None]
            blur_kernel_2d /= blur_kernel_2d.sum()

            self.blur_conv = nn.Conv2d(
                in_channel,
                in_channel,
                blur_kernel_2d.shape[0],
                padding=(pad0, pad1),
                groups=in_channel,
                bias=False,
            )

            # Set the kernel weights
            with torch.no_grad():
                # Expand kernel for groups
                kernel_expanded = blur_kernel_2d.unsqueeze(0).unsqueeze(0).expand(in_channel, 1, -1, -1)
                self.blur_conv.weight.copy_(kernel_expanded)

            stride = 2
            padding = 0
        else:
            stride = 1
            padding = kernel_size // 2

        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias and not activate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            input = self.blur_conv(input)

        input = self.conv2d(input)

        if self.activate:
            input = self.act(input + self.bias_leaky_relu) * 2**0.5

        return input


class ResBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class WanAnimateMotionEmbedder(nn.Module):
    def __init__(self, size: int = 512, style_dim: int = 512, motion_dim: int = 20):
        super().__init__()

        # Appearance encoder: conv layers
        channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64, 512: 32, 1024: 16}
        log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs.append(nn.Conv2d(in_channel, style_dim, 4, padding=0, bias=False))

        # Motion encoder: linear layers
        linears = []
        for _ in range(4):
            linears.append(nn.Linear(style_dim, style_dim))
        linears.append(nn.Linear(style_dim, motion_dim))
        self.linears = nn.Sequential(*linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(512, 20))

    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        # Appearance encoding through convs
        for conv in self.convs:
            face_image = conv(face_image)
        face_image = face_image.squeeze(-1).squeeze(-1)

        # Motion feature extraction
        motion_feat = self.linears(face_image)

        # Motion synthesis via QR decomposition
        weight = self.motion_synthesis_weight + 1e-8
        Q = torch.linalg.qr(weight.to(torch.float32))[0]

        input_diag = torch.diag_embed(motion_feat)  # Alpha, diagonal matrix
        out = torch.matmul(input_diag, Q.T)
        out = torch.sum(out, dim=1).to(motion_feat.dtype)
        return out


class WanAnimateFaceEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, kernel_size: int = 3, eps: float = 1e-6):
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)

        self.conv1_local = nn.Conv1d(in_dim, 1024 * num_heads, kernel_size=kernel_size, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 8, eps, elementwise_affine=False)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(1024, 1024, kernel_size, stride=2)
        self.conv3 = nn.Conv1d(1024, 1024, kernel_size, stride=2)

        self.out_proj = nn.Linear(1024, hidden_dim)
        self.norm1 = nn.LayerNorm(1024, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(1024, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(1024, eps, elementwise_affine=False)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, channels, num_frames = x.shape

        x = F.pad(x, self.time_causal_padding, mode="replicate")
        x = self.conv1_local(x)
        x = x.unflatten(1, (-1, channels)).flatten(0, 1).permute(0, 2, 1)

        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)

        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        return x_local


class WanTimeTextImageMotionFaceEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int,
        motion_encoder_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.image_embedder = WanImageEmbedding(image_embed_dim, dim)
        self.motion_embedder = WanAnimateMotionEmbedder(size=512, style_dim=512, motion_dim=20)
        self.face_embedder = WanAnimateFaceEmbedder(in_dim=motion_encoder_dim, hidden_dim=dim, num_heads=4)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        face_pixel_values: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = get_parameter_dtype(self.time_embedder)
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        # Motion vector computation from face pixel values
        batch_size, channels, num_face_frames, height, width = face_pixel_values.shape
        # Rearrange from (B, C, T, H, W) to (B*T, C, H, W)
        face_pixel_values_flat = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        # Extract motion features using motion embedder
        motion_vec = self.motion_embedder(face_pixel_values_flat)
        motion_vec = motion_vec.view(batch_size, num_face_frames, -1)

        # Encode motion vectors through face embedder
        motion_vec = self.face_embedder(motion_vec)

        # Add padding at the beginning (prepend zeros)
        batch_size, T_motion, N_motion, C_motion = motion_vec.shape
        pad_face = torch.zeros(batch_size, 1, N_motion, C_motion, dtype=motion_vec.dtype, device=motion_vec.device)
        motion_vec = torch.cat([pad_face, motion_vec], dim=1)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image, motion_vec


class WanAnimateFaceBlock(nn.Module):
    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num

        self.linear1_kv = nn.Linear(hidden_size, hidden_size * 2)
        self.linear1_q = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.q_norm = nn.RMSNorm(head_dim, eps)
        self.k_norm = nn.RMSNorm(head_dim, eps)

        self.pre_norm_feat = nn.LayerNorm(hidden_size, eps, elementwise_affine=False)
        self.pre_norm_motion = nn.LayerNorm(hidden_size, eps, elementwise_affine=False)

    def set_attention_backend(self, backend):
        """Set the attention backend for this face block."""
        self._attention_backend = backend

    def set_parallel_config(self, config):
        """Set the parallel configuration for this face block."""
        self._parallel_config = config

    def forward(
        self,
        x: torch.Tensor,
        motion_vec: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, C = motion_vec.shape
        T_comp = T

        x_motion = self.pre_norm_motion(motion_vec)
        x_feat = self.pre_norm_feat(x)

        kv = self.linear1_kv(x_motion)
        q = self.linear1_q(x_feat)

        k, v = kv.view(B, T, N, 2, self.heads_num, -1).permute(3, 0, 1, 2, 4, 5)
        q = q.unflatten(2, (self.heads_num, -1))

        q = self.q_norm(q.float()).type_as(q)
        k = self.k_norm(k.float()).type_as(k)

        k = k.flatten(0, 1)
        v = v.flatten(0, 1)

        q = q.unflatten(1, (T_comp, -1)).flatten(0, 1)

        attn = dispatch_attention_fn(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn = attn.unflatten(0, (B, T_comp)).flatten(1, 2)

        output = self.linear2(attn)

        return output


class WanAnimateTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the WanAnimate model.

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
        image_dim (`int`, *optional*, defaults to `1280`):
            The number of channels to use for the image embedding. If `None`, no projection is used.
        added_kv_proj_dim (`int`, *optional*, defaults to `5120`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanAnimateTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "motion_synthesis_weight",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = 1280,
        added_kv_proj_dim: Optional[int] = 5120,
        rope_max_seq_len: int = 1024,
        motion_encoder_dim: int = 512,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.pose_patch_embedding = nn.Conv3d(16, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageMotionFaceEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            motion_encoder_dim=motion_encoder_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.face_adapter = nn.ModuleList(
            [
                WanAnimateFaceBlock(
                    inner_dim,
                    num_attention_heads,
                )
                for _ in range(num_layers // 5)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def set_attention_backend(self, backend: str):
        """
        Set the attention backend for the transformer and all face adapter blocks.

        Args:
            backend (`str`): The attention backend to use (e.g., 'flash', 'sdpa', 'xformers').
        """
        from ..attention_dispatch import AttentionBackendName

        # Validate backend
        available_backends = {x.value for x in AttentionBackendName.__members__.values()}
        if backend not in available_backends:
            raise ValueError(f"`{backend=}` must be one of the following: " + ", ".join(available_backends))

        backend_enum = AttentionBackendName(backend.lower())

        # Call parent ModelMixin method to set backend for all attention modules
        super().set_attention_backend(backend)

        # Also set backend for all face adapter blocks (which use dispatch_attention_fn directly)
        for face_block in self.face_adapter:
            face_block.set_attention_backend(backend_enum)

    def set_parallel_config(self, config):
        """
        Set the parallel configuration for all face adapter blocks.

        Args:
            config: The parallel configuration to use.
        """
        for face_block in self.face_adapter:
            face_block.set_parallel_config(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pose_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        face_pixel_values: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
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
        pose_hidden_states = self.pose_patch_embedding(pose_hidden_states)
        # Add pose embeddings to hidden states
        hidden_states[:, :, 1:] = hidden_states[:, :, 1:] + pose_hidden_states[:, :, 1:]
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # sequence_length = int(math.ceil(np.prod([post_patch_num_frames, post_patch_height, post_patch_width]) // 4))
        # hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(hidden_states.shape[0], sequence_length - hidden_states.shape[1], hidden_states.shape[2])], dim=1)
        pose_hidden_states = pose_hidden_states.flatten(2).transpose(1, 2)

        # 3. Condition embeddings (time, text, image, motion)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image, motion_vec = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, face_pixel_values
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Image embedding
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 5. Transformer blocks with face adapter integration
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )

                # Face adapter integration: apply after every 5th block (0, 5, 10, 15, ...)
                if block_idx % 5 == 0:
                    face_adapter_output = self.face_adapter[block_idx // 5](hidden_states, motion_vec)
                    hidden_states = face_adapter_output + hidden_states
        else:
            for block_idx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

                # Face adapter integration: apply after every 5th block (0, 5, 10, 15, ...)
                if block_idx % 5 == 0:
                    face_adapter_output = self.face_adapter[block_idx // 5](hidden_states, motion_vec)
                    hidden_states = face_adapter_output + hidden_states

        # 6. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

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
