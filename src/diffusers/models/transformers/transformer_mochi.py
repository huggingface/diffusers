# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
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

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import FeedForward
from ..attention_processor import MochiAttention, MochiAttnProcessor2_0
from ..cache_utils import CacheMixin
from ..embeddings import MochiCombinedTimestepCaptionEmbedding, PatchEmbed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MochiModulatedRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()

        self.eps = eps
        self.norm = RMSNorm(0, eps, False)

    def forward(self, hidden_states, scale=None):
        hidden_states_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        hidden_states = self.norm(hidden_states)

        if scale is not None:
            hidden_states = hidden_states * scale

        hidden_states = hidden_states.to(hidden_states_dtype)

        return hidden_states


class MochiLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        eps=1e-5,
        bias=True,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)
        self.norm = MochiModulatedRMSNorm(eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype

        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        scale = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        x = self.norm(x, (1 + scale.unsqueeze(1).to(torch.float32)))

        return x.to(input_dtype)


class MochiRMSNormZero(nn.Module):
    r"""
    Adaptive RMS Norm used in Mochi.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self, embedding_dim: int, hidden_dim: int, eps: float = 1e-5, elementwise_affine: bool = False
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.norm = RMSNorm(0, eps, False)

    def forward(
        self, hidden_states: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states_dtype = hidden_states.dtype

        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        hidden_states = self.norm(hidden_states.to(torch.float32)) * (1 + scale_msa[:, None].to(torch.float32))
        hidden_states = hidden_states.to(hidden_states_dtype)

        return hidden_states, gate_msa, scale_mlp, gate_mlp


@maybe_allow_in_graph
class MochiTransformerBlock(nn.Module):
    r"""
    Transformer block used in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"swiglu"`):
            Activation function to use in feed-forward.
        context_pre_only (`bool`, defaults to `False`):
            Whether or not to process context-related conditions with additional layers.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        pooled_projection_dim: int,
        qk_norm: str = "rms_norm",
        activation_fn: str = "swiglu",
        context_pre_only: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.context_pre_only = context_pre_only
        self.ff_inner_dim = (4 * dim * 2) // 3
        self.ff_context_inner_dim = (4 * pooled_projection_dim * 2) // 3

        self.norm1 = MochiRMSNormZero(dim, 4 * dim, eps=eps, elementwise_affine=False)

        if not context_pre_only:
            self.norm1_context = MochiRMSNormZero(dim, 4 * pooled_projection_dim, eps=eps, elementwise_affine=False)
        else:
            self.norm1_context = MochiLayerNormContinuous(
                embedding_dim=pooled_projection_dim,
                conditioning_embedding_dim=dim,
                eps=eps,
            )

        self.attn1 = MochiAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=False,
            added_kv_proj_dim=pooled_projection_dim,
            added_proj_bias=False,
            out_dim=dim,
            out_context_dim=pooled_projection_dim,
            context_pre_only=context_pre_only,
            processor=MochiAttnProcessor2_0(),
            eps=1e-5,
        )

        # TODO(aryan): norm_context layers are not needed when `context_pre_only` is True
        self.norm2 = MochiModulatedRMSNorm(eps=eps)
        self.norm2_context = MochiModulatedRMSNorm(eps=eps) if not self.context_pre_only else None

        self.norm3 = MochiModulatedRMSNorm(eps)
        self.norm3_context = MochiModulatedRMSNorm(eps=eps) if not self.context_pre_only else None

        self.ff = FeedForward(dim, inner_dim=self.ff_inner_dim, activation_fn=activation_fn, bias=False)
        self.ff_context = None
        if not context_pre_only:
            self.ff_context = FeedForward(
                pooled_projection_dim,
                inner_dim=self.ff_context_inner_dim,
                activation_fn=activation_fn,
                bias=False,
            )

        self.norm4 = MochiModulatedRMSNorm(eps=eps)
        self.norm4_context = MochiModulatedRMSNorm(eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

        if not self.context_pre_only:
            norm_encoder_hidden_states, enc_gate_msa, enc_scale_mlp, enc_gate_mlp = self.norm1_context(
                encoder_hidden_states, temb
            )
        else:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)

        attn_hidden_states, context_attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=encoder_attention_mask,
        )

        hidden_states = hidden_states + self.norm2(attn_hidden_states, torch.tanh(gate_msa).unsqueeze(1))
        norm_hidden_states = self.norm3(hidden_states, (1 + scale_mlp.unsqueeze(1).to(torch.float32)))
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + self.norm4(ff_output, torch.tanh(gate_mlp).unsqueeze(1))

        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + self.norm2_context(
                context_attn_hidden_states, torch.tanh(enc_gate_msa).unsqueeze(1)
            )
            norm_encoder_hidden_states = self.norm3_context(
                encoder_hidden_states, (1 + enc_scale_mlp.unsqueeze(1).to(torch.float32))
            )
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + self.norm4_context(
                context_ff_output, torch.tanh(enc_gate_mlp).unsqueeze(1)
            )

        return hidden_states, encoder_hidden_states


class MochiRoPE(nn.Module):
    r"""
    RoPE implementation used in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        base_height (`int`, defaults to `192`):
            Base height used to compute interpolation scale for rotary positional embeddings.
        base_width (`int`, defaults to `192`):
            Base width used to compute interpolation scale for rotary positional embeddings.
    """

    def __init__(self, base_height: int = 192, base_width: int = 192) -> None:
        super().__init__()

        self.target_area = base_height * base_width

    def _centers(self, start, stop, num, device, dtype) -> torch.Tensor:
        edges = torch.linspace(start, stop, num + 1, device=device, dtype=dtype)
        return (edges[:-1] + edges[1:]) / 2

    def _get_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        scale = (self.target_area / (height * width)) ** 0.5

        t = torch.arange(num_frames, device=device, dtype=dtype)
        h = self._centers(-height * scale / 2, height * scale / 2, height, device, dtype)
        w = self._centers(-width * scale / 2, width * scale / 2, width, device, dtype)

        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")

        positions = torch.stack([grid_t, grid_h, grid_w], dim=-1).view(-1, 3)
        return positions

    def _create_rope(self, freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        with torch.autocast(freqs.device.type, torch.float32):
            # Always run ROPE freqs computation in FP32
            freqs = torch.einsum("nd,dhf->nhf", pos.to(torch.float32), freqs.to(torch.float32))

        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    def forward(
        self,
        pos_frequencies: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self._get_positions(num_frames, height, width, device, dtype)
        rope_cos, rope_sin = self._create_rope(pos_frequencies, pos)
        return rope_cos, rope_sin


@maybe_allow_in_graph
class MochiTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data introduced in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `48`):
            The number of layers of Transformer blocks to use.
        in_channels (`int`, defaults to `12`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `256`):
            Output dimension of timestep embeddings.
        activation_fn (`str`, defaults to `"swiglu"`):
            Activation function to use in feed-forward.
        max_sequence_length (`int`, defaults to `256`):
            The maximum sequence length of text embeddings supported.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MochiTransformerBlock"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        pooled_projection_dim: int = 1536,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 4096,
        time_embed_dim: int = 256,
        activation_fn: str = "swiglu",
        max_sequence_length: int = 256,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_type=None,
        )

        self.time_embed = MochiCombinedTimestepCaptionEmbedding(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            text_embed_dim=text_embed_dim,
            time_embed_dim=time_embed_dim,
            num_attention_heads=8,
        )

        self.pos_frequencies = nn.Parameter(torch.full((3, num_attention_heads, attention_head_dim // 2), 0.0))
        self.rope = MochiRoPE()

        self.transformer_blocks = nn.ModuleList(
            [
                MochiTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    pooled_projection_dim=pooled_projection_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    context_pre_only=i == num_layers - 1,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            inner_dim,
            inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            norm_type="layer_norm",
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
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
        p = self.config.patch_size

        post_patch_height = height // p
        post_patch_width = width // p

        temb, encoder_hidden_states = self.time_embed(
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_dtype=hidden_states.dtype,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        image_rotary_emb = self.rope(
            self.pos_frequencies,
            num_frames,
            post_patch_height,
            post_patch_width,
            device=hidden_states.device,
            dtype=torch.float32,
        )

        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    encoder_attention_mask,
                    image_rotary_emb,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
