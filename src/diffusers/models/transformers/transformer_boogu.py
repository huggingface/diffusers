"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.teacache_util import TeaCacheParams

from ..attention_processor_boogu import (
    BooguImageAttnProcessor,
    BooguImageDoubleStreamSelfAttnProcessor,
)
from .block_lumina2 import (
    Lumina2CombinedTimestepCaptionEmbedding,
    LuminaFeedForward,
    LuminaLayerNormContinuous,
    LuminaRMSNormZero,
)
from .rope_boogu import BooguImageDoubleStreamRotaryPosEmbed, BooguImagePromptTuningRotaryPosEmbed


logger = logging.get_logger(__name__)


class PromptEmbedding(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BooguImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["prompt_token_embedding", "norm"]

    @register_to_config
    def __init__(
        self,
        num_trainable_prompt_tokens: int = 32,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        num_layers: int = 2,
        theta: int = 10000,
    ):
        super().__init__()

        prompt_emb_head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.prompt_token_embedding = nn.Embedding(
            num_embeddings=self.config.num_trainable_prompt_tokens,
            embedding_dim=self.config.hidden_size,
        )

        # Rotary embedding for prompt tokens.
        self.prompt_rope_embedder = BooguImagePromptTuningRotaryPosEmbed(
            theta=self.config.theta,
            dim=prompt_emb_head_dim,
            num_trainable_prompt_tokens=self.config.num_trainable_prompt_tokens,
        )

        self.prompt_tuning_layers = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    dim=self.config.hidden_size,
                    num_attention_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_kv_heads,
                    multiple_of=self.config.multiple_of,
                    ffn_dim_multiplier=self.config.ffn_dim_multiplier,
                    norm_eps=self.config.norm_eps,
                    modulation=False,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.gradient_checkpointing = False

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Small std keeps prompt tuning stable at init.
        nn.init.normal_(self.prompt_token_embedding.weight, mean=0.0, std=0.02)

    def forward(self, idx=None, batch_size=1, device=None, use_causal_mask=True):
        if idx is None:
            prompt_embeddings = self.prompt_token_embedding.weight
        else:
            prompt_embeddings = self.prompt_token_embedding(idx)

        # Expand to [B, num_tokens, hidden_dim].
        hidden_states = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        rotary_emb, attention_mask = self.prompt_rope_embedder(batch_size, device, use_causal_mask)

        for i, layer in enumerate(self.prompt_tuning_layers):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_emb,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    rotary_emb,
                )
        return hidden_states


class BooguImageTransformerBlock(nn.Module):
    """
    Basic Boogu-Image transformer block: attention + MLP + RMSNorm.
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
        """Initialize the transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        # Initialize attention layer
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
            processor=BooguImageAttnProcessor(),
        )

        # Initialize feed-forward network
        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        # Initialize normalization layers
        if modulation:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize linear weights and modulation parameters."""
        nn.init.xavier_uniform_(self.attn.to_q.weight)
        nn.init.xavier_uniform_(self.attn.to_k.weight)
        nn.init.xavier_uniform_(self.attn.to_v.weight)
        nn.init.xavier_uniform_(self.attn.to_out[0].weight)

        nn.init.xavier_uniform_(self.feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_3.weight)

        if self.modulation:
            nn.init.zeros_(self.norm1.linear.weight)
            nn.init.zeros_(self.norm1.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            hidden_states: Input hidden states tensor
            attention_mask: Attention mask tensor
            image_rotary_emb: Rotary embeddings for image tokens
            temb: Optional timestep embedding tensor

        Returns:
            torch.Tensor: Output hidden states after transformer block processing
        """
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")
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


class BooguImageDoubleStreamTransformerBlock(nn.Module):
    """
    Boogu-Image double-stream block.
    Here "double-stream" is the same idea as a "dual-stream" layer:
    instruction tokens and image tokens are processed in parallel streams.
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
        """Initialize the double stream transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.modulation = modulation
        self.hidden_size = dim

        double_stream_processor = BooguImageDoubleStreamSelfAttnProcessor(
            head_dim=self.head_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=False,
        )

        # Image stream components.
        self.img_instruct_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=double_stream_processor,
        )

        self.img_self_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=BooguImageAttnProcessor(),
        )

        self.img_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Image modulation terms: cross-attn, MLP, self-attn.
            self.img_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
            self.img_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
            self.img_norm3 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.img_norm1 = RMSNorm(dim, eps=norm_eps)
            self.img_norm2 = RMSNorm(dim, eps=norm_eps)
            self.img_norm3 = RMSNorm(dim, eps=norm_eps)

        self.img_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.img_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        # Instruction stream components.
        self.instruct_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Instruction modulation terms: cross-attn, MLP.
            self.instruct_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
            self.instruct_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.instruct_norm1 = RMSNorm(dim, eps=norm_eps)
            self.instruct_norm2 = RMSNorm(dim, eps=norm_eps)

        self.instruct_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.initialize_weights()

        # double_stream_processor owns its own q/k/v projections.
        for param in self.img_instruct_attn.to_q.parameters():
            param.requires_grad = False
        for param in self.img_instruct_attn.to_k.parameters():
            param.requires_grad = False
        for param in self.img_instruct_attn.to_v.parameters():
            param.requires_grad = False

        del self.img_instruct_attn.to_k
        del self.img_instruct_attn.to_v
        del self.img_instruct_attn.to_q

    def initialize_weights(self) -> None:
        """Initialize linear weights and modulation parameters."""
        nn.init.xavier_uniform_(self.img_instruct_attn.to_out[0].weight)

        # Keep Xavier init consistent across Boogu-Image blocks.
        nn.init.xavier_uniform_(self.img_self_attn.to_q.weight)
        nn.init.xavier_uniform_(self.img_self_attn.to_k.weight)
        nn.init.xavier_uniform_(self.img_self_attn.to_v.weight)
        nn.init.xavier_uniform_(self.img_self_attn.to_out[0].weight)

        nn.init.xavier_uniform_(self.img_feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.img_feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.img_feed_forward.linear_3.weight)

        nn.init.xavier_uniform_(self.instruct_feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.instruct_feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.instruct_feed_forward.linear_3.weight)

        # Initialize modulation parameters
        if self.modulation:
            nn.init.zeros_(self.img_norm1.linear.weight)
            nn.init.zeros_(self.img_norm1.linear.bias)
            nn.init.zeros_(self.img_norm2.linear.weight)
            nn.init.zeros_(self.img_norm2.linear.bias)
            nn.init.zeros_(self.img_norm3.linear.weight)
            nn.init.zeros_(self.img_norm3.linear.bias)

            nn.init.zeros_(self.instruct_norm1.linear.weight)
            nn.init.zeros_(self.instruct_norm1.linear.bias)
            nn.init.zeros_(self.instruct_norm2.linear.weight)
            nn.init.zeros_(self.instruct_norm2.linear.bias)

    def forward(
        self,
        img_hidden_states: torch.Tensor,  # [B, L_img, D] - Image tokens (ref_img + noise_img)
        instruct_hidden_states: torch.Tensor,  # [B, L_instruct, D] - Instruction tokens
        img_attention_mask: torch.Tensor,  # [B, L_img] - Attention mask for [ref_img + noise_img]
        joint_attention_mask: torch.Tensor,  # [B, L_total] - Combined attention mask for [instruct + img]
        image_rotary_emb: torch.Tensor,  # [B, L_img, head_dim] - Rotary embeddings for [ref_img + noise_img]
        rotary_emb: torch.Tensor,  # [B, L_total, head_dim] - Rotary embeddings for [instruct + img]
        temb: Optional[torch.Tensor] = None,  # [B, 1024] - Timestep embeddings
        encoder_seq_lengths: List[int] = None,  # [B] - Instruction sequence lengths for each sample
        seq_lengths: List[int] = None,  # [B] - Total sequence lengths for each sample
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one dual-stream (double-stream) block step.
        Returns updated `(img_hidden_states, instruct_hidden_states)`.
        """
        if self.modulation and temb is None:
            raise ValueError("temb must be provided when modulation is enabled")

        # Extract dimensions
        batch_size = img_hidden_states.shape[0]
        L_instruct = instruct_hidden_states.shape[1]  # Instruction sequence length
        L_img = img_hidden_states.shape[1]  # Image sequence length (ref_img + noise_img)

        if self.modulation:
            # Step 1: modulation for both streams.
            img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img_hidden_states, temb)
            img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
            img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

            (
                instruct_norm1_out,
                instruct_gate_msa,
                instruct_scale_mlp,
                instruct_gate_mlp,
            ) = self.instruct_norm1(instruct_hidden_states, temb)
            instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)

            # Step 2: joint attention on [instruct + img].
            # Call processor directly because Attention.forward does not expose these dual-stream args.
            joint_attn_out = self.img_instruct_attn.processor(
                attn=self.img_instruct_attn,
                img_hidden_states=img_norm1_out,
                instruct_hidden_states=instruct_norm1_out,
                joint_attention_mask=joint_attention_mask,
                rotary_emb=rotary_emb,
                encoder_seq_lengths=encoder_seq_lengths,
                seq_lengths=seq_lengths,
            )

            # Split back into instruction/image segments.
            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            # Step 3: image self-attention.
            img_self_attn_out = self.img_self_attn(
                hidden_states=img_norm3_out,
                encoder_hidden_states=img_norm3_out,
                attention_mask=img_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

            # Step 4: residual updates.
            img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(1).tanh() * self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(1).tanh() * self.img_self_attn_norm(
                img_self_attn_out
            )

            img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
            img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(1).tanh() * self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + instruct_gate_msa.unsqueeze(
                1
            ).tanh() * self.instruct_attn_norm(instruct_attn_out)

            instruct_mlp_input = (
                1 + instruct_scale_mlp.unsqueeze(1)
            ) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
            instruct_hidden_states = instruct_hidden_states + instruct_gate_mlp.unsqueeze(
                1
            ).tanh() * self.instruct_ffn_norm2(instruct_mlp_out)

        else:
            # Non-modulated branch used by context-style blocks.
            img_norm1_out = self.img_norm1(img_hidden_states)
            img_norm3_out = self.img_norm3(img_hidden_states)
            instruct_norm1_out = self.instruct_norm1(instruct_hidden_states)

            # Same processor path as above.
            joint_attn_out = self.img_instruct_attn.processor(
                attn=self.img_instruct_attn,
                img_hidden_states=img_norm1_out,
                instruct_hidden_states=instruct_norm1_out,
                joint_attention_mask=joint_attention_mask,
                rotary_emb=rotary_emb,
                encoder_seq_lengths=encoder_seq_lengths,
                seq_lengths=seq_lengths,
            )

            instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct, self.hidden_size)
            img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

            img_self_attn_out = self.img_self_attn(
                hidden_states=img_norm3_out,
                encoder_hidden_states=img_norm3_out,
                attention_mask=img_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

            img_hidden_states = img_hidden_states + self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + self.img_self_attn_norm(img_self_attn_out)
            img_norm2_out = self.img_norm2(img_hidden_states)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_norm2_out))
            img_hidden_states = img_hidden_states + self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + self.instruct_attn_norm(instruct_attn_out)
            instruct_norm2_out = self.instruct_norm2(instruct_hidden_states)
            instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_norm2_out))
            instruct_hidden_states = instruct_hidden_states + self.instruct_ffn_norm2(instruct_mlp_out)

        return img_hidden_states, instruct_hidden_states


class BooguImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Boogu-Image transformer with mixed stream topology.
    Early layers use double-stream (aka dual-stream) processing, then switch
    to single-stream joint processing.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "BooguImageTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
        "PromptEmbedding",
    ]
    _repeated_blocks = [
        "BooguImageTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
    ]
    _skip_layerwise_casting_patterns = ["x_embedder", "norm", "embedding"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_double_stream_layers: int = 2,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: Tuple[int, int, int] = (40, 40, 40),
        axes_lens: Tuple[int, int, int] = (2048, 1664, 1664),
        instruction_feature_configs: Dict[str, Any] = {
            "instruction_feat_dim": 1024,
            "reduce_type": "mean",
            "num_instruction_feat_layers": 1,
        },
        prompt_tuning_configs: Dict[str, Any] = {"use_prompt_tuning": False},
        timestep_scale: float = 1.0,
    ) -> None:
        """Initialize the Boogu-Image mixed single-double stream transformer model."""
        super().__init__()

        # Validate configuration
        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )

        if num_double_stream_layers > num_layers:
            raise ValueError(
                f"num_double_stream_layers ({num_double_stream_layers}) cannot be greater than "
                f"num_layers ({num_layers})"
            )

        self.out_channels = out_channels or in_channels
        self.num_double_stream_layers = num_double_stream_layers
        self.num_single_stream_layers = num_layers - num_double_stream_layers
        self.instruction_feature_configs = instruction_feature_configs
        self.prompt_tuning_configs = prompt_tuning_configs
        self.preprocessed_instruction_feat_dim = self.cal_preprocessed_instruction_feat_dim(
            instruction_feature_configs
        )

        # Initialize embeddings
        self.rope_embedder = BooguImageDoubleStreamRotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            instruction_feat_dim=self.preprocessed_instruction_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        # Refiner layers.
        self.noise_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
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

        self.ref_image_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
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
                BooguImageTransformerBlock(
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

        # Mixed architecture: dual-stream first, then single-stream.
        # Here "double-stream" and "dual-stream" mean the same thing.
        self.double_stream_layers = nn.ModuleList(
            [
                BooguImageDoubleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_double_stream_layers)
            ]
        )

        # Single-stream layers process the fused sequence; they reuse BooguImageTransformerBlock.
        self.single_stream_layers = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(self.num_single_stream_layers)
            ]
        )

        # Output norm and projection.
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Distinguish multiple reference images.
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))  # support max 5 ref images

        self.gradient_checkpointing = False

        self.initialize_weights()

        # TeaCache settings
        self.enable_teacache = False
        self.teacache_rel_l1_thresh = 0.05
        self.teacache_params = TeaCacheParams()

        # Polynomial (highest-degree first) rescaling the relative L1 distance used by TeaCache.
        self.teacache_rescale_coefficients = [-5.48259225, 11.48772289, -4.47407401, 2.47730926, -0.03316487]

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the model.

        Uses Xavier uniform initialization for linear layers.
        """
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        nn.init.xavier_uniform_(self.ref_image_patch_embedder.weight)
        nn.init.constant_(self.ref_image_patch_embedder.bias, 0.0)

        nn.init.zeros_(self.norm_out.linear_1.weight)
        nn.init.zeros_(self.norm_out.linear_1.bias)
        nn.init.zeros_(self.norm_out.linear_2.weight)
        nn.init.zeros_(self.norm_out.linear_2.bias)

        nn.init.normal_(self.image_index_embedding, std=0.02)

    def img_patch_embed_and_refine(
        self,
        hidden_states,
        ref_image_hidden_states,
        padded_img_mask,
        padded_ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    ):
        """Embed image patches and run the refiner blocks."""
        batch_size = len(hidden_states)
        max_combined_img_len = max(
            [img_len + sum(ref_img_len) for img_len, ref_img_len in zip(l_effective_img_len, l_effective_ref_img_len)]
        )

        hidden_states = self.x_embedder(hidden_states)
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)

        for i in range(batch_size):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift : shift + ref_img_len, :] = (
                    ref_image_hidden_states[i, shift : shift + ref_img_len, :] + self.image_index_embedding[j]
                )
                shift += ref_img_len

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb)

        flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))
        num_ref_images = len(flat_l_effective_ref_img_len)
        max_ref_img_len = max(flat_l_effective_ref_img_len)

        batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images, max_ref_img_len, dtype=torch.bool)
        batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, self.config.hidden_size
        )
        batch_ref_img_rotary_emb = hidden_states.new_zeros(
            num_ref_images,
            max_ref_img_len,
            ref_img_rotary_emb.shape[-1],
            dtype=ref_img_rotary_emb.dtype,
        )
        batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)

        # Flatten reference images into a temporary batch.
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                batch_ref_img_mask[idx, :ref_img_len] = True
                batch_ref_image_hidden_states[idx, :ref_img_len] = ref_image_hidden_states[
                    i, shift : shift + ref_img_len
                ]
                batch_ref_img_rotary_emb[idx, :ref_img_len] = ref_img_rotary_emb[i, shift : shift + ref_img_len]
                batch_temb[idx] = temb[i]
                shift += ref_img_len
                idx += 1

        # Refine each reference-image sample.
        for layer in self.ref_image_refiner:
            batch_ref_image_hidden_states = layer(
                batch_ref_image_hidden_states,
                batch_ref_img_mask,
                batch_ref_img_rotary_emb,
                batch_temb,
            )

        # Restore reference-image sequence layout.
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                ref_image_hidden_states[i, shift : shift + ref_img_len] = batch_ref_image_hidden_states[
                    idx, :ref_img_len
                ]
                shift += ref_img_len
                idx += 1

        combined_img_hidden_states = hidden_states.new_zeros(batch_size, max_combined_img_len, self.config.hidden_size)
        for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
            combined_img_hidden_states[i, : sum(ref_img_len)] = ref_image_hidden_states[i, : sum(ref_img_len)]
            combined_img_hidden_states[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = hidden_states[i, :img_len]

        return combined_img_hidden_states

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        """Flatten patch tokens and pad to batched sequences."""
        batch_size = len(hidden_states)
        p = self.config.patch_size
        device = hidden_states[0].device

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None:
            ref_img_sizes = [
                [(img.size(1), img.size(2)) for img in imgs] if imgs is not None else None
                for imgs in ref_image_hidden_states
            ]
            l_effective_ref_img_len = [
                [(ref_img_size[0] // p) * (ref_img_size[1] // p) for ref_img_size in _ref_img_sizes]
                if _ref_img_sizes is not None
                else [0]
                for _ref_img_sizes in ref_img_sizes
            ]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # Reference-image patch embeddings.
        flat_ref_img_hidden_states = []
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                imgs = []
                for ref_img in ref_image_hidden_states[i]:
                    C, H, W = ref_img.size()
                    # "c (h p1) (w p2) -> (h w) (p1 p2 c)"
                    ref_img = ref_img.reshape(C, H // p, p, W // p, p)
                    ref_img = ref_img.permute(1, 3, 2, 4, 0)
                    ref_img = ref_img.reshape((H // p) * (W // p), p * p * C)
                    imgs.append(ref_img)

                img = torch.cat(imgs, dim=0)
                flat_ref_img_hidden_states.append(img)
            else:
                flat_ref_img_hidden_states.append(None)

        # Noise-image patch embeddings.
        flat_hidden_states = []
        for i in range(batch_size):
            img = hidden_states[i]
            C, H, W = img.size()

            # "c (h p1) (w p2) -> (h w) (p1 p2 c)"
            img = img.reshape(C, H // p, p, W // p, p)
            img = img.permute(1, 3, 2, 4, 0)
            img = img.reshape((H // p) * (W // p), p * p * C)
            flat_hidden_states.append(img)

        padded_ref_img_hidden_states = torch.zeros(
            batch_size,
            max_ref_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_ref_img_mask = torch.zeros(batch_size, max_ref_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                padded_ref_img_hidden_states[i, : sum(l_effective_ref_img_len[i])] = flat_ref_img_hidden_states[i]
                padded_ref_img_mask[i, : sum(l_effective_ref_img_len[i])] = True

        padded_hidden_states = torch.zeros(
            batch_size,
            max_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_img_mask = torch.zeros(batch_size, max_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            padded_hidden_states[i, : l_effective_img_len[i]] = flat_hidden_states[i]
            padded_img_mask[i, : l_effective_img_len[i]] = True

        return (
            padded_hidden_states,
            padded_ref_img_hidden_states,
            padded_img_mask,
            padded_ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        )

    def cal_preprocessed_instruction_feat_dim(self, instruction_feature_configs: Dict[str, Any]):
        num_instruction_feat_layers = max(instruction_feature_configs.get("num_instruction_feat_layers", 1), 1)
        instruction_feat_dim = instruction_feature_configs.get("instruction_feat_dim", 4096)
        reduce_type = instruction_feature_configs.get("reduce_type", "concat")
        if "cat" in reduce_type.lower():
            return num_instruction_feat_layers * instruction_feat_dim
        elif "mean" in reduce_type.lower():
            return instruction_feat_dim
        else:
            raise ValueError(f"Invalid reduce_type: {reduce_type}")

    def preprocess_instruction_hidden_states(
        self, raw_instruction_hidden_states, instruction_feature_configs: Dict[str, Any]
    ):
        num_instruction_feat_layers = max(instruction_feature_configs.get("num_instruction_feat_layers", 1), 1)
        reduce_type = instruction_feature_configs.get("reduce_type", "concat")

        instruction_hidden_states = None
        if isinstance(raw_instruction_hidden_states, torch.Tensor):
            instruction_hidden_states = raw_instruction_hidden_states
        elif isinstance(raw_instruction_hidden_states, (list, tuple)):
            assert len(raw_instruction_hidden_states) == num_instruction_feat_layers
            if "cat" in reduce_type.lower():
                instruction_hidden_states = torch.cat(raw_instruction_hidden_states, dim=-1)
            elif "mean" in reduce_type.lower():
                instruction_hidden_states = torch.mean(torch.stack(raw_instruction_hidden_states), dim=0)
            else:
                raise ValueError(f"Invalid reduce_type: {reduce_type}")
        else:
            raise ValueError(
                f"Invalid type of raw_instruction_hidden_states, expected torch.Tensor or list, but got {type(raw_instruction_hidden_states)}"
            )

        assert self.preprocessed_instruction_feat_dim == instruction_hidden_states.shape[-1]

        return instruction_hidden_states

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass:
        context/refiner -> dual-stream (double-stream) -> fusion -> single-stream -> projection.
        """
        instruction_hidden_states = self.preprocess_instruction_hidden_states(
            instruction_hidden_states, self.instruction_feature_configs
        )

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

        # === 1. Initial processing (same as original Boogu-Image) ===
        batch_size = len(hidden_states)
        is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

        if is_hidden_states_tensor:
            assert hidden_states.ndim == 4
            hidden_states = list(hidden_states)

        device = hidden_states[0].device

        # Timestep and instruction embedding.
        temb, instruction_hidden_states = self.time_caption_embed(
            timestep, instruction_hidden_states, hidden_states[0].dtype
        )

        # Flatten and pad token sequences.
        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        # Build rotary embeddings and sequence lengths.
        (
            context_rotary_emb,
            ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
            combined_img_rotary_emb,
            combined_img_seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            instruction_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        # Context refinement.
        for layer in self.context_refiner:
            instruction_hidden_states = layer(
                instruction_hidden_states,
                instruction_attention_mask,
                context_rotary_emb,
            )

        # Image patch embedding and refinement.
        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        # Dual-stream (double-stream) stage.
        instruct_hidden_states = instruction_hidden_states
        img_hidden_states = combined_img_hidden_states

        # Joint mask for [instruct + image].
        max_seq_len = max(seq_lengths)
        joint_attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            joint_attention_mask[i, :seq_len] = True

        # Run dual-stream blocks.
        if self.num_double_stream_layers > 0:
            # Image-only mask for [ref + noise].
            max_img_len = max(combined_img_seq_lengths)
            img_attention_mask = hidden_states.new_zeros(batch_size, max_img_len, dtype=torch.bool)
            for i, img_seq_len in enumerate(combined_img_seq_lengths):
                img_attention_mask[i, :img_seq_len] = True

            for layer in self.double_stream_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    img_hidden_states, instruct_hidden_states = self._gradient_checkpointing_func(
                        layer,
                        img_hidden_states,
                        instruct_hidden_states,
                        img_attention_mask,
                        joint_attention_mask,
                        combined_img_rotary_emb,
                        rotary_emb,
                        temb,
                        encoder_seq_lengths,
                        seq_lengths,
                    )
                else:
                    img_hidden_states, instruct_hidden_states = layer(
                        img_hidden_states,
                        instruct_hidden_states,
                        img_attention_mask,
                        joint_attention_mask,
                        combined_img_rotary_emb,
                        rotary_emb,
                        temb,
                        encoder_seq_lengths,
                        seq_lengths,
                    )

        # Fuse streams to joint sequence.
        joint_hidden_states = hidden_states.new_zeros(batch_size, max(seq_lengths), self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            joint_hidden_states[i, :encoder_seq_len] = instruct_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = img_hidden_states[i, : seq_len - encoder_seq_len]

        # Single-stream stage.
        hidden_states = joint_hidden_states

        # TeaCache optimization.
        if self.enable_teacache and len(self.single_stream_layers) > 0:
            teacache_hidden_states = hidden_states.clone()
            teacache_temb = temb.clone()
            modulated_inp, _, _, _ = self.single_stream_layers[0].norm1(teacache_hidden_states, teacache_temb)
            if self.teacache_params.is_first_or_last_step:
                should_calc = True
                self.teacache_params.accumulated_rel_l1_distance = 0
            else:
                rel_l1 = (
                    (
                        (modulated_inp - self.teacache_params.previous_modulated_inp).abs().mean()
                        / self.teacache_params.previous_modulated_inp.abs().mean()
                    )
                    .cpu()
                    .item()
                )
                rescaled = 0.0
                for coefficient in self.teacache_rescale_coefficients:
                    rescaled = rescaled * rel_l1 + coefficient
                self.teacache_params.accumulated_rel_l1_distance += rescaled
                if self.teacache_params.accumulated_rel_l1_distance < self.teacache_rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.teacache_params.accumulated_rel_l1_distance = 0
            self.teacache_params.previous_modulated_inp = modulated_inp
        else:
            should_calc = True

        if self.enable_teacache and not should_calc:
            hidden_states += self.teacache_params.previous_residual
        else:
            if self.enable_teacache:
                ori_hidden_states = hidden_states.clone()

            for layer in self.single_stream_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        layer, hidden_states, joint_attention_mask, rotary_emb, temb
                    )
                else:
                    hidden_states = layer(hidden_states, joint_attention_mask, rotary_emb, temb)

            if self.enable_teacache:
                self.teacache_params.previous_residual = hidden_states - ori_hidden_states

        # Output projection.
        hidden_states = self.norm_out(hidden_states, temb)

        # Reshape back to image format.
        p = self.config.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
            height, width = img_size
            img_tokens = hidden_states[i][seq_len - img_len : seq_len]
            # "(h w) (p1 p2 c) -> c (h p1) (w p2)"
            h, w = height // p, width // p
            c = img_tokens.shape[-1] // (p * p)
            img_output = img_tokens.reshape(h, w, p, p, c)
            img_output = img_output.permute(4, 0, 2, 1, 3)
            img_output = img_output.reshape(c, h * p, w * p)
            output.append(img_output)

        if is_hidden_states_tensor:
            output = torch.stack(output, dim=0)

        # Reset LoRA scaling.
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)
