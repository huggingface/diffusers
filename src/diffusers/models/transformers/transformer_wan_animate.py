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


class FusedLeakyReLU(nn.Module):
    """
    Fused LeakyRelu with scale factor and channel-wise bias.
    """

    def __init__(self, negative_slope: float = 0.2, scale: float = 2**0.5, bias_channels: Optional[int] = None):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.channels = bias_channels

        if self.channels is not None:
            self.bias = nn.Parameter(torch.zeros(self.channels,))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.bias is not None:
            # Expand self.bias to have all singleton dims except at self.channel_dim
            expanded_shape = [1] * x.ndim
            expanded_shape[channel_dim] = self.bias.shape[0]
            bias = self.bias.reshape(*expanded_shape)
            x = x + bias
        return F.leaky_relu(x, self.negative_slope) * self.scale


class MotionConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        blur_kernel: Optional[Tuple[int, ...]] = None,
        blur_upsample_factor: int = 1,
        use_activation: bool = True,
    ):
        super().__init__()
        self.use_activation = use_activation
        self.in_channels = in_channels

        # Handle blurring (applying a FIR filter with the given kernel) if available
        self.blur = False
        if blur_kernel is not None:
            p = (len(blur_kernel) - stride) + (kernel_size - 1)
            self.blur_padding = ((p + 1) // 2, p // 2)

            kernel = torch.tensor(blur_kernel)
            # Convert kernel to 2D if necessary
            if kernel.ndim == 1:
                kernel = kernel[None, :] * kernel[:, None]
            # Normalize kernel
            kernel = kernel / kernel.sum()
            if blur_upsample_factor > 1:
                kernel = kernel * (blur_upsample_factor ** 2)
            self.register_buffer("blur_kernel", kernel, persistent=False)
            self.blur = True

        # Main Conv2d parameters (with scale factor)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        # If using an activation function, the bias will be fused into the activation
        if bias and not self.use_activation:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        if self.use_activation:
            self.act_fn = FusedLeakyReLU(bias_channels=out_channels)
        else:
            self.act_fn = None

    def forward(self, input: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        # Apply blur if using
        if self.blur:
            # NOTE: the original implementation uses a 2D upfirdn operation with the upsampling and downsampling rates
            # set to 1, which should be equivalent to a 2D convolution
            expanded_kernel = self.blur_kernel[None, None, :, :].expand(self.in_channels, 1, -1, -1)
            x = F.conv2d(input, expanded_kernel, padding=self.blur_padding, groups=self.in_channels)

        # Main Conv2D with scaling
        x = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

        # Activation with fused bias, if using
        if self.use_activation:
            x = self.act_fn(x, channel_dim=channel_dim)
        return x

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class MotionLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        use_activation: bool = False,
    ):
        super().__init__()
        self.use_activation = use_activation

        # Linear weight with scale factor
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = (1 / math.sqrt(in_dim))

        # If an activation is present, the bias will be fused to it
        if bias and not self.use_activation:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        if self.use_activation:
            self.act_fn = FusedLeakyReLU(bias_channels=out_dim)
        else:
            self.act_fn = None

    def forward(self, input: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        out = F.linear(input, self.weight * self.scale, bias=self.bias)
        if self.use_activation:
            out = self.act_fn(out, channel_dim=channel_dim)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class MotionEncoderResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        kernel_size_skip: int = 1,
        blur_kernel: Tuple[int, ...] = (1, 3, 3, 1),
        downsample_factor: int = 2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        # 3 x 3 Conv + fused leaky ReLU
        self.conv1 = MotionConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            use_activation=True,
        )

        # 3 x 3 Conv that downsamples 2x + fused leaky ReLU
        self.conv2 = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=0,
            blur_kernel=blur_kernel,
            use_activation=True,
        )

        # 1 x 1 Conv that downsamples 2x in skip connection
        self.conv_skip = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_skip,
            stride=self.downsample_factor,
            padding=0,
            bias=False,
            blur_kernel=blur_kernel,
            use_activation=False,
        )

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        x_out = self.conv1(x, channel_dim)
        x_out = self.conv2(x_out, channel_dim)

        x_skip = self.conv_skip(x, channel_dim)

        x_out = (x_out + x_skip) / math.sqrt(2)
        return x_out


class WanAnimateMotionEncoder(nn.Module):
    def __init__(
        self, size: int = 512, style_dim: int = 512, motion_dim: int = 20, out_dim: int = 512, motion_blocks: int = 5
    ):
        super().__init__()

        # Appearance encoder: conv layers
        channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64, 512: 32, 1024: 16}
        log_size = int(math.log(size, 2))

        self.conv_in = MotionConv2d(3, channels[size], 1, use_activation=True)

        self.res_blocks = nn.ModuleList()
        in_channels = channels[size]
        for i in range(log_size, 2, -1):
            out_channels = channels[2 ** (i - 1)]
            self.res_blocks.append(MotionEncoderResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.conv_out = MotionConv2d(in_channels, style_dim, 4, padding=0, bias=False, use_activation=False)

        # Motion encoder: linear layers
        # NOTE: there are no activations in between the linear layers here, which is weird but I believe matches the
        # original code.
        linears = [MotionLinear(style_dim, style_dim) for _ in range(motion_blocks - 1)]
        linears.append(MotionLinear(style_dim, motion_dim))
        self.motion_network = nn.Sequential(*linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, face_image: torch.Tensor, channel_dim: int = 1, upcast_to_fp32: bool = True) -> torch.Tensor:
        # Appearance encoding through convs
        face_image = self.conv_in(face_image, channel_dim)
        for block in self.res_blocks:
            face_image = block(face_image, channel_dim)
        face_image = self.conv_out(face_image, channel_dim)
        face_image = face_image.squeeze(-1).squeeze(-1)

        # Motion feature extraction
        motion_feat = self.motion_network(face_image, channel_dim)

        # Motion synthesis via QR decomposition
        weight = self.motion_synthesis_weight + 1e-8
        if upcast_to_fp32:
            original_motion_dtype = motion_feat.dtype
            motion_feat = motion_feat.to(torch.float32)
            weight = weight.to(torch.float32)

        Q = torch.linalg.qr(weight)[0]

        motion_feat_diag = torch.diag_embed(motion_feat)  # Alpha, diagonal matrix
        motion_decomposition = torch.matmul(motion_feat_diag, Q.T)
        motion_vec = torch.sum(motion_decomposition, dim=1)
        if upcast_to_fp32:
            motion_vec = motion_vec.to(dtype=original_motion_dtype)

        return motion_vec


class WanAnimateFaceEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        kernel_size: int = 3,
        eps: float = 1e-6,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.pad_mode = pad_mode

        self.act = nn.SiLU()

        self.conv1_local = nn.Conv1d(in_dim, hidden_dim * num_heads, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)

        self.norm1 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)

        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels = x.shape

        # Reshape to channels-first to apply causal Conv1d over frame dim
        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv1_local(x)  # [B, C, T] --> [B, N * C, T]
        x = x.unflatten(1, (-1, channels)).flatten(0, 1)  # [B, N * C, T] --> [B * N, C, T]
        # Reshape back to channels-last to apply LayerNorm over channel dim
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.act(x)

        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)

        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)

        x = self.out_proj(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)  # [B * N, T, C_out] --> [B, T, N, C_out]

        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)  # [B, T, N, C_out] --> [B, T, N + 1, C_out]
        x_local = x.clone()

        return x_local


class WanTimeTextImageMotionFaceEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 5120,  # num_attention_heads * attention_head_dim = 40 * 128 = 5120
        time_freq_dim: int = 256,
        time_proj_dim: int = 30720,  # (40 * 128) * 6 = 30720
        text_embed_dim: int = 4096,
        image_embed_dim: int = 1280,
        motion_encoder_size: int = 512,
        motion_style_dim: int = 512,
        motion_dim: int = 20,
        motion_encoder_dim: int = 512,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.image_embedder = WanImageEmbedding(image_embed_dim, dim)
        self.motion_embedder = WanAnimateMotionEncoder(
            size=motion_encoder_size, style_dim=motion_style_dim, motion_dim=motion_dim, out_dim=motion_encoder_dim
        )
        self.face_embedder = WanAnimateFaceEncoder(in_dim=motion_encoder_dim, out_dim=dim, num_heads=4)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        face_pixel_values: Optional[torch.Tensor] = None,
        upcast_to_fp32: bool = True,
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
        motion_vec = self.motion_embedder(face_pixel_values_flat, upcast_to_fp32=upcast_to_fp32)
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
        motion_encoder_size: int = 512,
        motion_style_dim: int = 512,
        motion_dim: int = 20,
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
            motion_encoder_size=motion_encoder_size,
            motion_style_dim=motion_style_dim,
            motion_dim=motion_dim,
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
