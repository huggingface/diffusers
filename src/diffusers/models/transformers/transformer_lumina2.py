# Copyright 2025 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
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
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import LuminaFeedForward
from ..attention_processor import Attention
from ..embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import LuminaLayerNormContinuous, LuminaRMSNormZero, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        cap_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )

        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, hidden_size, bias=True)
        )

    def forward(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).type_as(hidden_states)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(encoder_hidden_states)
        return time_embed, caption_embed


class Lumina2AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Lumina2Transformer2DModel model. It applies normalization and RoPE on query and key vectors.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key Norm if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Apply proportional attention if true
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # perform Grouped-qurey Attention (GQA)
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class Lumina2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=Lumina2AttnProcessor2_0(),
        )

        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            self.norm1 = LuminaRMSNormZero(
                embedding_dim=dim,
                norm_eps=norm_eps,
                norm_elementwise_affine=True,
            )
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.modulation:
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class Lumina2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], axes_lens: List[int] = (300, 512, 512), patch_size: int = 2):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

        self.freqs_cis = self._precompute_freqs_cis(axes_dim, axes_lens, theta)

    def _precompute_freqs_cis(self, axes_dim: List[int], axes_lens: List[int], theta: int) -> List[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=self.theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = self.freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1).to(device)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, channels, height, width = hidden_states.shape
        p = self.patch_size
        post_patch_height, post_patch_width = height // p, width // p
        image_seq_len = post_patch_height * post_patch_width
        device = hidden_states.device

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()
        seq_lengths = [cap_seq_len + image_seq_len for cap_seq_len in l_effective_cap_len]
        max_seq_len = max(seq_lengths)

        # Create position IDs
        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add caption position ids
            position_ids[i, :cap_seq_len, 0] = torch.arange(cap_seq_len, dtype=torch.int32, device=device)
            position_ids[i, cap_seq_len:seq_len, 0] = cap_seq_len

            # add image position ids
            row_ids = (
                torch.arange(post_patch_height, dtype=torch.int32, device=device)
                .view(-1, 1)
                .repeat(1, post_patch_width)
                .flatten()
            )
            col_ids = (
                torch.arange(post_patch_width, dtype=torch.int32, device=device)
                .view(1, -1)
                .repeat(post_patch_height, 1)
                .flatten()
            )
            position_ids[i, cap_seq_len:seq_len, 1] = row_ids
            position_ids[i, cap_seq_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            batch_size, encoder_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        img_freqs_cis = torch.zeros(
            batch_size, image_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            img_freqs_cis[i, :image_seq_len] = freqs_cis[i, cap_seq_len:seq_len]

        # image patch embeddings
        hidden_states = (
            hidden_states.view(batch_size, channels, post_patch_height, p, post_patch_width, p)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )

        return hidden_states, cap_freqs_cis, img_freqs_cis, freqs_cis, l_effective_cap_len, seq_lengths


class Lumina2Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    Lumina2NextDiT: Diffusion model with a Transformer backbone.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        in_channels (`int`, *optional*, defaults to 4):
            The number of input channels for the model. Typically, this matches the number of channels in the input
            images.
        hidden_size (`int`, *optional*, defaults to 4096):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        num_layers (`int`, *optional*, default to 32):
            The number of layers in the model. This defines the depth of the neural network.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in each attention layer. This parameter specifies how many separate attention
            mechanisms are used.
        num_kv_heads (`int`, *optional*, defaults to 8):
            The number of key-value heads in the attention mechanism, if different from the number of attention heads.
            If None, it defaults to num_attention_heads.
        multiple_of (`int`, *optional*, defaults to 256):
            A factor that the hidden size should be a multiple of. This can help optimize certain hardware
            configurations.
        ffn_dim_multiplier (`float`, *optional*):
            A multiplier for the dimensionality of the feed-forward network. If None, it uses a default value based on
            the model configuration.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            A small value added to the denominator for numerical stability in normalization layers.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            A scaling factor applied to certain parameters or layers in the model. This can be used for adjusting the
            overall scale of the model's operations.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Lumina2TransformerBlock"]
    _skip_layerwise_casting_patterns = ["x_embedder", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        scaling_factor: float = 1.0,
        axes_dim_rope: Tuple[int, int, int] = (32, 32, 32),
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        cap_feat_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or in_channels

        # 1. Positional, patch & conditional embeddings
        self.rope_embedder = Lumina2RotaryPosEmbed(
            theta=10000, axes_dim=axes_dim_rope, axes_lens=axes_lens, patch_size=patch_size
        )

        self.x_embedder = nn.Linear(in_features=patch_size * patch_size * in_channels, out_features=hidden_size)

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size, cap_feat_dim=cap_feat_dim, norm_eps=norm_eps
        )

        # 2. Noise and context refinement blocks
        self.noise_refiner = nn.ModuleList(
            [
                Lumina2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                Lumina2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # 3. Transformer blocks
        self.layers = nn.ModuleList(
            [
                Lumina2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
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

        # 1. Condition, positional & patch embedding
        batch_size, _, height, width = hidden_states.shape

        temb, encoder_hidden_states = self.time_caption_embed(hidden_states, timestep, encoder_hidden_states)

        (
            hidden_states,
            context_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.rope_embedder(hidden_states, encoder_attention_mask)

        hidden_states = self.x_embedder(hidden_states)

        # 2. Context & noise refinement
        for layer in self.context_refiner:
            encoder_hidden_states = layer(encoder_hidden_states, encoder_attention_mask, context_rotary_emb)

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, None, noise_rotary_emb, temb)

        # 3. Joint Transformer blocks
        max_seq_len = max(seq_lengths)
        use_mask = len(set(seq_lengths)) > 1

        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = encoder_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = hidden_states[i]

        hidden_states = joint_hidden_states

        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer, hidden_states, attention_mask if use_mask else None, rotary_emb, temb
                )
            else:
                hidden_states = layer(hidden_states, attention_mask if use_mask else None, rotary_emb, temb)

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)

        # 5. Unpatchify
        p = self.config.patch_size
        output = []
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            output.append(
                hidden_states[i][encoder_seq_len:seq_len]
                .view(height // p, width // p, p, p, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        output = torch.stack(output, dim=0)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
