# Copyright 2025 The Framepack Team, The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, get_logger, scale_lora_layers, unscale_lora_layers
from ..cache_utils import CacheMixin
from ..embeddings import get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous
from .transformer_hunyuan_video import (
    HunyuanVideoConditionEmbedding,
    HunyuanVideoPatchEmbed,
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTokenRefiner,
    HunyuanVideoTransformerBlock,
)


logger = get_logger(__name__)  # pylint: disable=invalid-name


class HunyuanVideoFramepackRotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: List[int], theta: float = 256.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, frame_indices: torch.Tensor, height: int, width: int, device: torch.device):
        height = height // self.patch_size
        width = width // self.patch_size
        grid = torch.meshgrid(
            frame_indices.to(device=device, dtype=torch.float32),
            torch.arange(0, height, device=device, dtype=torch.float32),
            torch.arange(0, width, device=device, dtype=torch.float32),
            indexing="ij",
        )  # 3 * [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)

        return freqs_cos, freqs_sin


class FramepackClipVisionProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.down(hidden_states)
        return hidden_states


class HunyuanVideoHistoryPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, inner_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(in_channels, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(in_channels, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    def forward(
        self,
        latents_clean: Optional[torch.Tensor] = None,
        latents_clean_2x: Optional[torch.Tensor] = None,
        latents_clean_4x: Optional[torch.Tensor] = None,
    ):
        if latents_clean is not None:
            latents_clean = self.proj(latents_clean)
            latents_clean = latents_clean.flatten(2).transpose(1, 2)
        if latents_clean_2x is not None:
            latents_clean_2x = _pad_for_3d_conv(latents_clean_2x, (2, 4, 4))
            latents_clean_2x = self.proj_2x(latents_clean_2x)
            latents_clean_2x = latents_clean_2x.flatten(2).transpose(1, 2)
        if latents_clean_4x is not None:
            latents_clean_4x = _pad_for_3d_conv(latents_clean_4x, (4, 8, 8))
            latents_clean_4x = self.proj_4x(latents_clean_4x)
            latents_clean_4x = latents_clean_4x.flatten(2).transpose(1, 2)
        return latents_clean, latents_clean_2x, latents_clean_4x


class HunyuanVideoFramepackTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoHistoryPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
        image_condition_type: Optional[str] = None,
        has_image_proj: int = False,
        image_proj_dim: int = 1152,
        has_clean_x_embedder: int = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)

        # Framepack history projection embedder
        self.clean_x_embedder = None
        if has_clean_x_embedder:
            self.clean_x_embedder = HunyuanVideoHistoryPatchEmbed(in_channels, inner_dim)

        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )

        # Framepack image-conditioning embedder
        self.image_projection = FramepackClipVisionProjection(image_proj_dim, inner_dim) if has_image_proj else None

        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim, pooled_projection_dim, guidance_embeds, image_condition_type
        )

        # 2. RoPE
        self.rope = HunyuanVideoFramepackRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        image_embeds: torch.Tensor,
        indices_latents: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        latents_clean: Optional[torch.Tensor] = None,
        indices_latents_clean: Optional[torch.Tensor] = None,
        latents_history_2x: Optional[torch.Tensor] = None,
        indices_latents_history_2x: Optional[torch.Tensor] = None,
        latents_history_4x: Optional[torch.Tensor] = None,
        indices_latents_history_4x: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
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
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        if indices_latents is None:
            indices_latents = torch.arange(0, num_frames).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.x_embedder(hidden_states)
        image_rotary_emb = self.rope(
            frame_indices=indices_latents, height=height, width=width, device=hidden_states.device
        )

        latents_clean, latents_history_2x, latents_history_4x = self.clean_x_embedder(
            latents_clean, latents_history_2x, latents_history_4x
        )

        if latents_clean is not None and indices_latents_clean is not None:
            image_rotary_emb_clean = self.rope(
                frame_indices=indices_latents_clean, height=height, width=width, device=hidden_states.device
            )
        if latents_history_2x is not None and indices_latents_history_2x is not None:
            image_rotary_emb_history_2x = self.rope(
                frame_indices=indices_latents_history_2x, height=height, width=width, device=hidden_states.device
            )
        if latents_history_4x is not None and indices_latents_history_4x is not None:
            image_rotary_emb_history_4x = self.rope(
                frame_indices=indices_latents_history_4x, height=height, width=width, device=hidden_states.device
            )

        hidden_states, image_rotary_emb = self._pack_history_states(
            hidden_states,
            latents_clean,
            latents_history_2x,
            latents_history_4x,
            image_rotary_emb,
            image_rotary_emb_clean,
            image_rotary_emb_history_2x,
            image_rotary_emb_history_4x,
            post_patch_height,
            post_patch_width,
        )

        temb, _ = self.time_text_embed(timestep, pooled_projections, guidance)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_image = self.image_projection(image_embeds)
        attention_mask_image = encoder_attention_mask.new_ones((batch_size, encoder_hidden_states_image.shape[1]))

        # must cat before (not after) encoder_hidden_states, due to attn masking
        encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([attention_mask_image, encoder_attention_mask], dim=1)

        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.zeros(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )  # [B, N]
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

        if batch_size == 1:
            encoder_hidden_states = encoder_hidden_states[:, : effective_condition_sequence_length[0]]
            attention_mask = None
        else:
            for i in range(batch_size):
                attention_mask[i, : effective_sequence_length[i]] = True
            # [B, 1, 1, N], for broadcasting across attention heads
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )

        hidden_states = hidden_states[:, -original_context_length:]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    def _pack_history_states(
        self,
        hidden_states: torch.Tensor,
        latents_clean: Optional[torch.Tensor] = None,
        latents_history_2x: Optional[torch.Tensor] = None,
        latents_history_4x: Optional[torch.Tensor] = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] = None,
        image_rotary_emb_clean: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_rotary_emb_history_2x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_rotary_emb_history_4x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        height: int = None,
        width: int = None,
    ):
        image_rotary_emb = list(image_rotary_emb)  # convert tuple to list for in-place modification

        if latents_clean is not None and image_rotary_emb_clean is not None:
            hidden_states = torch.cat([latents_clean, hidden_states], dim=1)
            image_rotary_emb[0] = torch.cat([image_rotary_emb_clean[0], image_rotary_emb[0]], dim=0)
            image_rotary_emb[1] = torch.cat([image_rotary_emb_clean[1], image_rotary_emb[1]], dim=0)

        if latents_history_2x is not None and image_rotary_emb_history_2x is not None:
            hidden_states = torch.cat([latents_history_2x, hidden_states], dim=1)
            image_rotary_emb_history_2x = self._pad_rotary_emb(image_rotary_emb_history_2x, height, width, (2, 2, 2))
            image_rotary_emb[0] = torch.cat([image_rotary_emb_history_2x[0], image_rotary_emb[0]], dim=0)
            image_rotary_emb[1] = torch.cat([image_rotary_emb_history_2x[1], image_rotary_emb[1]], dim=0)

        if latents_history_4x is not None and image_rotary_emb_history_4x is not None:
            hidden_states = torch.cat([latents_history_4x, hidden_states], dim=1)
            image_rotary_emb_history_4x = self._pad_rotary_emb(image_rotary_emb_history_4x, height, width, (4, 4, 4))
            image_rotary_emb[0] = torch.cat([image_rotary_emb_history_4x[0], image_rotary_emb[0]], dim=0)
            image_rotary_emb[1] = torch.cat([image_rotary_emb_history_4x[1], image_rotary_emb[1]], dim=0)

        return hidden_states, tuple(image_rotary_emb)

    def _pad_rotary_emb(
        self,
        image_rotary_emb: Tuple[torch.Tensor],
        height: int,
        width: int,
        kernel_size: Tuple[int, int, int],
    ):
        # freqs_cos, freqs_sin have shape [W * H * T, D / 2], where D is attention head dim
        freqs_cos, freqs_sin = image_rotary_emb
        freqs_cos = freqs_cos.unsqueeze(0).permute(0, 2, 1).unflatten(2, (-1, height, width))
        freqs_sin = freqs_sin.unsqueeze(0).permute(0, 2, 1).unflatten(2, (-1, height, width))
        freqs_cos = _pad_for_3d_conv(freqs_cos, kernel_size)
        freqs_sin = _pad_for_3d_conv(freqs_sin, kernel_size)
        freqs_cos = _center_down_sample_3d(freqs_cos, kernel_size)
        freqs_sin = _center_down_sample_3d(freqs_sin, kernel_size)
        freqs_cos = freqs_cos.flatten(2).permute(0, 2, 1).squeeze(0)
        freqs_sin = freqs_sin.flatten(2).permute(0, 2, 1).squeeze(0)
        return freqs_cos, freqs_sin


def _pad_for_3d_conv(x, kernel_size):
    if isinstance(x, (tuple, list)):
        return tuple(_pad_for_3d_conv(i, kernel_size) for i in x)
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def _center_down_sample_3d(x, kernel_size):
    if isinstance(x, (tuple, list)):
        return tuple(_center_down_sample_3d(i, kernel_size) for i in x)
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)
