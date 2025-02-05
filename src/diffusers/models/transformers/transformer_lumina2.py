# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...loaders import PeftAdapterMixin
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import LuminaFeedForward
from ..attention_processor import Attention
from ..embeddings import get_1d_rotary_pos_embed, apply_rotary_emb, Timesteps, TimestepEmbedding
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import LuminaLayerNormContinuous, LuminaRMSNormZero, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(self, hidden_size=4096, cap_feat_dim=2048, frequency_embedding_size=256, norm_eps=1e-5):
        super().__init__()
        
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )

        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024))

        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(
                cap_feat_dim,
                hidden_size,
                bias=True,
            ),
        )

    def forward(self, timestep, caption_feat):
        # timestep embedding:
        time_freq = self.time_proj(timestep)
        time_embed = self.timestep_embedder(time_freq.to(dtype=self.timestep_embedder.linear_1.weight.dtype))

        # caption condition embedding:
        caption_embed = self.caption_embedder(caption_feat)

        return time_embed, caption_embed
    
    
class Lumina2AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Lumina2Transformer2DModel model. It applies a s normalization layer and rotary embedding on query and key vector.
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

        input_ndim = hidden_states.ndim
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
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        attention_mask = attention_mask.expand(-1, attn.heads, sequence_length, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        
        return hidden_states
    
    
class Lumina2TransformerBlock(nn.Module):
    """
    A Lumina2TransformerBlock for Lumina2Transformer2DModel.

    Parameters:
        dim (`int`): Embedding dimension of the input features.
        num_attention_heads (`int`): Number of attention heads.
        num_kv_heads (`int`):
            Number of attention heads in key and value features (if using GQA), or set to None for the same as query.
        multiple_of (`int`): The number of multiple of ffn layer.
        ffn_dim_multiplier (`float`): The multipier factor of ffn layer dimension.
        norm_eps (`float`): The eps for norm layer.
        qk_norm (`bool`): normalization for query and key.
        cross_attention_dim (`int`): Cross attention embedding dimension of the input text prompt hidden_states.
        modulation (`bool`): Whether to use modulation.
    """

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

        # Self-attention
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
    ):
        """
        Perform a forward pass through the Lumina2TransformerBlock.

        Parameters:
            hidden_states (`torch.Tensor`): The input of hidden_states for Lumina2TransformerBlock.
            attention_mask (`torch.Tensor): The input of hidden_states corresponse attention mask.
            image_rotary_emb (`torch.Tensor`): Precomputed cosine and sine frequencies.
            encoder_hidden_states: (`torch.Tensor`): The hidden_states of text prompt are processed by Gemma encoder.
            encoder_mask (`torch.Tensor`): The hidden_states of text prompt attention mask.
            temb (`torch.Tensor`): Timestep embedding with text prompt embedding.
        """
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
        self.freqs_cis = self.precompute_freqs_cis(axes_dim, axes_lens, theta)
        
    def precompute_freqs_cis(self, axes_dim: List[int], axes_lens: List[int], theta: int) -> List[torch.Tensor]:
        freqs_cis = []
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(
                d,
                e,
                theta=self.theta,
                freqs_dtype=torch.float64,
            )
            freqs_cis.append(emb)
        return freqs_cis
    
    def get_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        result = []
        for i in range(len(self.axes_dim)):
            freqs = self.freqs_cis[i].to(ids.device)
            index = ids[:, :, i:i+1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
    ):
        bsz = len(hidden_states)
        pH = pW = self.patch_size
        device = hidden_states[0].device

        l_effective_cap_len = encoder_mask.sum(dim=1).tolist()
        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // pH) * (W // pW) for (H, W) in img_sizes]

        max_seq_len = max(
            (cap_len+img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len))
        )
        max_cap_len = max(l_effective_cap_len)
        max_img_len = max(l_effective_img_len)
        
        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)
        
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            H, W = img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW
            assert H_tokens * W_tokens == img_len

            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:cap_len+img_len, 0] = cap_len
            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            position_ids[i, cap_len:cap_len+img_len, 1] = row_ids
            position_ids[i, cap_len:cap_len+img_len, 2] = col_ids

        freqs_cis = self.get_freqs_cis(position_ids)
        
        cap_freqs_cis_shape = list(freqs_cis.shape)
        cap_freqs_cis_shape[1] = encoder_mask.shape[1]
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        img_freqs_cis_shape = list(freqs_cis.shape)
        img_freqs_cis_shape[1] = max_img_len
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)
        
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len:cap_len+img_len]
        
        flat_hidden_states = []
        for i in range(bsz):
            img = hidden_states[i]
            C, H, W = img.size()
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 2, 4, 0).flatten(2).flatten(0, 1)
            flat_hidden_states.append(img)
        hidden_states = flat_hidden_states
        padded_img_embed = torch.zeros(bsz, max_img_len, hidden_states[0].shape[-1], device=device, dtype=hidden_states[0].dtype)
        padded_img_mask = torch.zeros(bsz, max_img_len, dtype=torch.bool, device=device)
        for i in range(bsz):
            padded_img_embed[i, :l_effective_img_len[i]] = hidden_states[i]
            padded_img_mask[i, :l_effective_img_len[i]] = True
            
        return padded_img_embed, padded_img_mask, img_sizes, l_effective_cap_len, l_effective_img_len, freqs_cis, cap_freqs_cis, img_freqs_cis, max_seq_len


class Lumina2Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
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
    
    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: Optional[int] = 2,
        in_channels: Optional[int] = 16,
        out_channels: Optional[int] = None,
        hidden_size: Optional[int] = 2304,
        num_layers: Optional[int] = 26,
        num_refiner_layers: Optional[int] = 2,
        num_attention_heads: Optional[int] = 24,
        num_kv_heads: Optional[int] = 8,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: Optional[float] = 1e-5,
        scaling_factor: Optional[float] = 1.0,
        axes_dim_rope: Optional[tuple[int, int, int]] = (32, 32, 32),
        axes_lens: Optional[tuple[int, int, int]] = (300, 512, 512),
        cap_feat_dim: Optional[int] = 1024,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or in_channels

        self.rope_embedder = Lumina2RotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )
        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
            bias=True,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            cap_feat_dim=cap_feat_dim,
            norm_eps=norm_eps,
        )

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
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )
        
        self.gradient_checkpointing = False
    
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        return_dict=True,
    ) -> torch.Tensor:
        """
        Forward pass of LuminaNextDiT.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
        """
        bsz = hidden_states.size(0)
        device = hidden_states.device
        temb, encoder_hidden_states = self.time_caption_embed(timestep, encoder_hidden_states)
        hidden_states, hidden_mask, hidden_sizes, encoder_hidden_len, hidden_len, joint_rotary_emb, encoder_rotary_emb, hidden_rotary_emb, max_seq_len = self.rope_embedder(hidden_states, encoder_mask)
        
        hidden_states = self.x_embedder(hidden_states)
        for layer in self.context_refiner:
            encoder_hidden_states = layer(encoder_hidden_states, encoder_mask, encoder_rotary_emb)
            
        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, hidden_mask, hidden_rotary_emb, temb)
        
        mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool, device=device)
        padded_hidden_states = torch.zeros(bsz, max_seq_len, self.config.hidden_size, device=device, dtype=hidden_states.dtype)
        for i in range(bsz):
            cap_len = encoder_hidden_len[i]
            img_len = hidden_len[i]
            mask[i, :cap_len+img_len] = True
            padded_hidden_states[i, :cap_len] = encoder_hidden_states[i, :cap_len]
            padded_hidden_states[i, cap_len:cap_len+img_len] = hidden_states[i, :img_len]
        hidden_states = padded_hidden_states

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            
            for layer in self.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    mask,
                    joint_rotary_emb,
                    temb=temb,
                )
        else:
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    mask,
                    joint_rotary_emb,
                    temb=temb,
                )

        hidden_states = self.norm_out(hidden_states, temb)

        # uspatchify
        height_tokens = width_tokens = self.config.patch_size
        output = []
        for i in range(len(hidden_sizes)):
            height, width = hidden_sizes[i]
            begin = encoder_hidden_len[i]
            end = begin + (height // height_tokens) * (width // width_tokens)
            output.append(
                hidden_states[i][begin:end]
                .view(height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        output = torch.stack(output, dim=0)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)