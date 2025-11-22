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
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES = {
    "4": 512,
    "8": 512,
    "16": 512,
    "32": 512,
    "64": 256,
    "128": 128,
    "256": 64,
    "512": 32,
    "1024": 16,
}


# Copied from diffusers.models.transformers.transformer_wan._get_qkv_projections
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


# Copied from diffusers.models.transformers.transformer_wan._get_added_kv_projections
def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


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
            self.bias = nn.Parameter(
                torch.zeros(
                    self.channels,
                )
            )
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
                kernel = kernel * (blur_upsample_factor**2)
            self.register_buffer("blur_kernel", kernel, persistent=False)
            self.blur = True

        # Main Conv2d parameters (with scale factor)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

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

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        # Apply blur if using
        if self.blur:
            # NOTE: the original implementation uses a 2D upfirdn operation with the upsampling and downsampling rates
            # set to 1, which should be equivalent to a 2D convolution
            expanded_kernel = self.blur_kernel[None, None, :, :].expand(self.in_channels, 1, -1, -1)
            x = F.conv2d(x, expanded_kernel, padding=self.blur_padding, groups=self.in_channels)

        # Main Conv2D with scaling
        x = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

        # Activation with fused bias, if using
        if self.use_activation:
            x = self.act_fn(x, channel_dim=channel_dim)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" kernel_size={self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
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
        self.scale = 1 / math.sqrt(in_dim)

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
        return (
            f"{self.__class__.__name__}(in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]},"
            f" bias={self.bias is not None})"
        )


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
        self,
        size: int = 512,
        style_dim: int = 512,
        motion_dim: int = 20,
        out_dim: int = 512,
        motion_blocks: int = 5,
        channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.size = size

        # Appearance encoder: conv layers
        if channels is None:
            channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

        self.conv_in = MotionConv2d(3, channels[str(size)], 1, use_activation=True)

        self.res_blocks = nn.ModuleList()
        in_channels = channels[str(size)]
        log_size = int(math.log(size, 2))
        for i in range(log_size, 2, -1):
            out_channels = channels[str(2 ** (i - 1))]
            self.res_blocks.append(MotionEncoderResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.conv_out = MotionConv2d(in_channels, style_dim, 4, padding=0, bias=False, use_activation=False)

        # Motion encoder: linear layers
        # NOTE: there are no activations in between the linear layers here, which is weird but I believe matches the
        # original code.
        linears = [MotionLinear(style_dim, style_dim) for _ in range(motion_blocks - 1)]
        linears.append(MotionLinear(style_dim, motion_dim))
        self.motion_network = nn.ModuleList(linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, face_image: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if (face_image.shape[-2] != self.size) or (face_image.shape[-1] != self.size):
            raise ValueError(
                f"Face pixel values has resolution ({face_image.shape[-1]}, {face_image.shape[-2]}) but is expected"
                f" to have resolution ({self.size}, {self.size})"
            )

        # Appearance encoding through convs
        face_image = self.conv_in(face_image, channel_dim)
        for block in self.res_blocks:
            face_image = block(face_image, channel_dim)
        face_image = self.conv_out(face_image, channel_dim)
        motion_feat = face_image.squeeze(-1).squeeze(-1)

        # Motion feature extraction
        for linear_layer in self.motion_network:
            motion_feat = linear_layer(motion_feat, channel_dim=channel_dim)

        # Motion synthesis via Linear Motion Decomposition
        weight = self.motion_synthesis_weight + 1e-8
        # Upcast the QR orthogonalization operation to FP32
        original_motion_dtype = motion_feat.dtype
        motion_feat = motion_feat.to(torch.float32)
        weight = weight.to(torch.float32)

        Q = torch.linalg.qr(weight)[0].to(device=motion_feat.device)

        motion_feat_diag = torch.diag_embed(motion_feat)  # Alpha, diagonal matrix
        motion_decomposition = torch.matmul(motion_feat_diag, Q.T)
        motion_vec = torch.sum(motion_decomposition, dim=1)

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
        self.num_heads = num_heads
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
        batch_size = x.shape[0]

        # Reshape to channels-first to apply causal Conv1d over frame dim
        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv1_local(x)  # [B, C, T_padded] --> [B, N * C, T]
        x = x.unflatten(1, (self.num_heads, -1)).flatten(0, 1)  # [B, N * C, T] --> [B * N, C, T]
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

        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1).to(device=x.device)
        x = torch.cat([x, padding], dim=-2)  # [B, T, N, C_out] --> [B, T, N + 1, C_out]

        return x


class WanAnimateFaceBlockAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or"
                f" higher."
            )

    def __call__(
        self,
        attn: "WanAnimateFaceBlockCrossAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # encoder_hidden_states corresponds to the motion vec
        # attention_mask corresponds to the motion mask (if any)
        hidden_states = attn.pre_norm_q(hidden_states)
        encoder_hidden_states = attn.pre_norm_kv(encoder_hidden_states)

        # B --> batch_size, T --> reduced inference segment len, N --> face_encoder_num_heads + 1, C --> attn.dim
        B, T, N, C = encoder_hidden_states.shape

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = query.unflatten(2, (attn.heads, -1))  # [B, S, H * D] --> [B, S, H, D]
        key = key.view(B, T, N, attn.heads, -1)  # [B, T, N, H * D_kv] --> [B, T, N, H, D_kv]
        value = value.view(B, T, N, attn.heads, -1)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # NOTE: the below line (which follows the official code) means that in practice, the number of frames T in
        # encoder_hidden_states (the motion vector after applying the face encoder) must evenly divide the
        # post-patchify sequence length S of the transformer hidden_states. Is it possible to remove this dependency?
        query = query.unflatten(1, (T, -1)).flatten(0, 1)  # [B, S, H, D] --> [B * T, S / T, H, D]
        key = key.flatten(0, 1)  # [B, T, N, H, D_kv] --> [B * T, N, H, D_kv]
        value = value.flatten(0, 1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = hidden_states.unflatten(0, (B, T)).flatten(1, 2)

        hidden_states = attn.to_out(hidden_states)

        if attention_mask is not None:
            # NOTE: attention_mask is assumed to be a multiplicative mask
            attention_mask = attention_mask.flatten(start_dim=1)
            hidden_states = hidden_states * attention_mask

        return hidden_states


class WanAnimateFaceBlockCrossAttention(nn.Module, AttentionModuleMixin):
    """
    Temporally-aligned cross attention with the face motion signal in the Wan Animate Face Blocks.
    """

    _default_processor_cls = WanAnimateFaceBlockAttnProcessor
    _available_processors = [WanAnimateFaceBlockAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-6,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_head_dim = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        # 1. Pre-Attention Norms for the hidden_states (video latents) and encoder_hidden_states (motion vector).
        # NOTE: this is not used in "vanilla" WanAttention
        self.pre_norm_q = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.pre_norm_kv = nn.LayerNorm(dim, eps, elementwise_affine=False)

        # 2. QKV and Output Projections
        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.Linear(self.inner_dim, dim, bias=True)

        # 3. QK Norm
        # NOTE: this is applied after the reshape, so only over dim_head rather than dim_head * heads
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)

        # 4. Set attention processor
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask)


# Copied from diffusers.models.transformers.transformer_wan.WanAttnProcessor
class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None

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

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

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
                parallel_config=self._parallel_config,
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
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# Copied from diffusers.models.transformers.transformer_wan.WanAttention
class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


# Copied from diffusers.models.transformers.transformer_wan.WanImageEmbedding
class WanImageEmbedding(torch.nn.Module):
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


# Copied from diffusers.models.transformers.transformer_wan.WanTimeTextImageEmbedding
class WanTimeTextImageEmbedding(nn.Module):
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
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
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
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


# Copied from diffusers.models.transformers.transformer_wan.WanRotaryPosEmbed
class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


# Copied from diffusers.models.transformers.transformer_wan.WanTransformerBlock
class WanTransformerBlock(nn.Module):
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
    _no_split_modules = ["WanTransformerBlock", "MotionEncoderResBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "motion_synthesis_weight",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: Optional[int] = 36,
        latent_channels: Optional[int] = 16,
        out_channels: Optional[int] = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = 1280,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        motion_encoder_channel_sizes: Optional[Dict[str, int]] = None,  # Start of Wan Animate-specific args
        motion_encoder_size: int = 512,
        motion_style_dim: int = 512,
        motion_dim: int = 20,
        motion_encoder_dim: int = 512,
        face_encoder_hidden_dim: int = 1024,
        face_encoder_num_heads: int = 4,
        inject_face_latents_blocks: int = 5,
        motion_encoder_batch_size: int = 8,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        # Allow either only in_channels or only latent_channels to be set for convenience
        if in_channels is None and latent_channels is not None:
            in_channels = 2 * latent_channels + 4
        elif in_channels is not None and latent_channels is None:
            latent_channels = (in_channels - 4) // 2
        elif in_channels is not None and latent_channels is not None:
            # TODO: should this always be true?
            assert in_channels == 2 * latent_channels + 4, "in_channels should be 2 * latent_channels + 4"
        else:
            raise ValueError("At least one of `in_channels` and `latent_channels` must be supplied.")
        out_channels = out_channels or latent_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.pose_patch_embedding = nn.Conv3d(latent_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # Motion encoder
        self.motion_encoder = WanAnimateMotionEncoder(
            size=motion_encoder_size,
            style_dim=motion_style_dim,
            motion_dim=motion_dim,
            out_dim=motion_encoder_dim,
            channels=motion_encoder_channel_sizes,
        )

        # Face encoder
        self.face_encoder = WanAnimateFaceEncoder(
            in_dim=motion_encoder_dim,
            out_dim=inner_dim,
            hidden_dim=face_encoder_hidden_dim,
            num_heads=face_encoder_num_heads,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_attention_heads,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.face_adapter = nn.ModuleList(
            [
                WanAnimateFaceBlockCrossAttention(
                    dim=inner_dim,
                    heads=num_attention_heads,
                    dim_head=inner_dim // num_attention_heads,
                    eps=eps,
                    cross_attention_dim_head=inner_dim // num_attention_heads,
                    processor=WanAnimateFaceBlockAttnProcessor(),
                )
                for _ in range(num_layers // inject_face_latents_blocks)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        pose_hidden_states: Optional[torch.Tensor] = None,
        face_pixel_values: Optional[torch.Tensor] = None,
        motion_encode_batch_size: Optional[int] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of Wan2.2-Animate transformer model.

        Args:
            hidden_states (`torch.Tensor` of shape `(B, 2C + 4, T + 1, H, W)`):
                Input noisy video latents of shape `(B, 2C + 4, T + 1, H, W)`, where B is the batch size, C is the
                number of latent channels (16 for Wan VAE), T is the number of latent frames in an inference segment, H
                is the latent height, and W is the latent width.
            timestep: (`torch.LongTensor`):
                The current timestep in the denoising loop.
            encoder_hidden_states (`torch.Tensor`):
                Text embeddings from the text encoder (umT5 for Wan Animate).
            encoder_hidden_states_image (`torch.Tensor`):
                CLIP visual features of the reference (character) image.
            pose_hidden_states (`torch.Tensor` of shape `(B, C, T, H, W)`):
                Pose video latents. TODO: description
            face_pixel_values (`torch.Tensor` of shape `(B, C', S, H', W')`):
                Face video in pixel space (not latent space). Typically C' = 3 and H' and W' are the height/width of
                the face video in pixels. Here S is the inference segment length, usually set to 77.
            motion_encode_batch_size (`int`, *optional*):
                The batch size for batched encoding of the face video via the motion encoder. Will default to
                `self.config.motion_encoder_batch_size` if not set.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return the output as a dict or tuple.
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

        # Check that shapes match up
        if pose_hidden_states is not None and pose_hidden_states.shape[2] + 1 != hidden_states.shape[2]:
            raise ValueError(
                f"pose_hidden_states frame dim (dim 2) is {pose_hidden_states.shape[2]} but must be one less than the"
                f" hidden_states's corresponding frame dim: {hidden_states.shape[2]}"
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
        hidden_states[:, :, 1:] = hidden_states[:, :, 1:] + pose_hidden_states
        # Calling contiguous() here is important so that we don't recompile when performing regional compilation
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        # 3. Condition embeddings (time, text, image)
        # Wan Animate is based on Wan 2.1 and thus uses Wan 2.1's timestep logic
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=None
        )

        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Get motion features from the face video
        # Motion vector computation from face pixel values
        batch_size, channels, num_face_frames, height, width = face_pixel_values.shape
        # Rearrange from (B, C, T, H, W) to (B*T, C, H, W)
        face_pixel_values = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        # Extract motion features using motion encoder
        # Perform batched motion encoder inference to allow trading off inference speed for memory usage
        motion_encode_batch_size = motion_encode_batch_size or self.config.motion_encoder_batch_size
        face_batches = torch.split(face_pixel_values, motion_encode_batch_size)
        motion_vec_batches = []
        for face_batch in face_batches:
            motion_vec_batch = self.motion_encoder(face_batch)
            motion_vec_batches.append(motion_vec_batch)
        motion_vec = torch.cat(motion_vec_batches)
        motion_vec = motion_vec.view(batch_size, num_face_frames, -1)

        # Now get face features from the motion vector
        motion_vec = self.face_encoder(motion_vec)

        # Add padding at the beginning (prepend zeros)
        pad_face = torch.zeros_like(motion_vec[:, :1])
        motion_vec = torch.cat([pad_face, motion_vec], dim=1)

        # 5. Transformer blocks with face adapter integration
        for block_idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

            # Face adapter integration: apply after every 5th block (0, 5, 10, 15, ...)
            if block_idx % self.config.inject_face_latents_blocks == 0:
                face_adapter_block_idx = block_idx // self.config.inject_face_latents_blocks
                face_adapter_output = self.face_adapter[face_adapter_block_idx](hidden_states, motion_vec)
                # In case the face adapter and main transformer blocks are on different devices, which can happen when
                # using model parallelism
                face_adapter_output = face_adapter_output.to(device=hidden_states.device)
                hidden_states = face_adapter_output + hidden_states

        # 6. Output norm, projection & unpatchify
        # batch_size, inner_dim
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states_original_dtype = hidden_states.dtype
        hidden_states = self.norm_out(hidden_states.float())
        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (hidden_states * (1 + scale) + shift).to(dtype=hidden_states_original_dtype)

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
