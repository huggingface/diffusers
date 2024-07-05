# Copyright 2024 Stability AI, Lavender Flow, The HuggingFace Team. All rights reserved.
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


from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_torch_version, logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention_processor import Attention
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, AdaLayerNormZero, FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Taken from the original lavender flow inference code.
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# Lavender Flow patch embed doesn't use convs for projections.
# Additionally, it uses learned positional embeddings.
class LavenderFlowPatchEmbed(nn.Module):
    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        pos_embed_max_size=None,
    ):
        super().__init__()

        self.num_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pos_embed_max_size, embed_dim) * 0.1)

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size

    def forward(self, latent):
        batch_size, num_channels, height, width = latent.size()
        latent = latent.view(
            batch_size,
            num_channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        latent = latent.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        latent = self.proj(latent)
        return latent + self.pos_embed


# Taken from the original lavender flow inference code.
# Our feedforward only has GELU but lavender uses SiLU.
class LavenderFlowFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(dim, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(dim, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class LavenderFlowAttnProcessor2_0:
    """Attention processor used typically in processing Lavender Flow."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention") and is_torch_version("<", "2.1"):
            raise ImportError(
                "LavenderFlowAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to at least 2.1 or above as we use `scale` in `F.scaled_dot_product_attention()`. "
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        i=0,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # Reshape.
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Apply QK norm.
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Concatenate the projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_q(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Attention.
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, scale=attn.scale, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, encoder_hidden_states.shape[1] :],
                hidden_states[:, : encoder_hidden_states.shape[1]],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@maybe_allow_in_graph
class LavenderFlowDiTTransformerBlock(nn.Module):
    """Similar `LavenderFlowTransformerBlock with a single DiT instead of an MMDiT."""

    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, use_fp32_layer_norm=True)

        processor = LavenderFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm",
            use_fp32_layer_norm=True,
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = LavenderFlowFeedForward(dim, dim * 4)

    def forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor, i=9999):
        residual = hidden_states

        # Norm + Projection.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Attention.
        attn_output = self.attn(hidden_states=norm_hidden_states, i=i)

        # Process attention outputs for the `hidden_states`.
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(hidden_states)
        hidden_states = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = residual + hidden_states

        return hidden_states


@maybe_allow_in_graph
class LavenderFlowTransformerBlock(nn.Module):
    r"""
    Transformer block for Lavender Flow. Similar to SD3 MMDiT. Differences (non-exhaustive):

        * QK Norm in the attention blocks
        * No bias in the attention blocks
        * Most LayerNorms are in FP32

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        is_last (`bool`): Boolean to determine if this is the last block in the model.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, is_last=False):
        super().__init__()
        self.is_last = is_last
        context_norm_type = "ada_norm_continous" if is_last else "ada_norm_zero"

        self.norm1 = AdaLayerNormZero(dim, bias=False, use_fp32_layer_norm=True)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, bias=False, use_fp32_layer_norm=True
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim, bias=False, use_fp32_layer_norm=True)

        processor = LavenderFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm",
            added_qk_norm="layer_norm",
            use_fp32_layer_norm=True,
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
            context_pre_only=False,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = LavenderFlowFeedForward(dim, dim * 4)
        self.norm2_context = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        if not is_last:
            self.ff_context = LavenderFlowFeedForward(dim, dim * 4)
        else:
            self.ff_context = None

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor, i=0
    ):
        residual = hidden_states
        residual_context = encoder_hidden_states

        # Norm + Projection.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, i=i
        )

        # Process attention outputs for the `hidden_states`.
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = gate_mlp.unsqueeze(1) * self.ff(hidden_states)
        hidden_states = residual + hidden_states

        # Process attention outputs for the `encoder_hidden_states`.
        encoder_hidden_states = self.norm2_context(residual_context + c_gate_msa.unsqueeze(1) * context_attn_output)
        encoder_hidden_states = encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = c_gate_mlp.unsqueeze(1) * self.ff_context(encoder_hidden_states)
        encoder_hidden_states = residual_context + encoder_hidden_states

        return encoder_hidden_states, hidden_states


class LavenderFlowTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 4,
        num_mmdit_layers: int = 4,
        num_single_dit_layers: int = 32,
        attention_head_dim: int = 256,
        num_attention_heads: int = 12,
        joint_attention_dim: int = 2048,
        caption_projection_dim: int = 3072,
        out_channels: int = 4,
        pos_embed_max_size: int = 1024,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = LavenderFlowPatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.context_embedder = nn.Linear(
            self.config.joint_attention_dim, self.config.caption_projection_dim, bias=False
        )
        self.time_step_embed = Timesteps(num_channels=256, downscale_freq_shift=0, scale=1000, flip_sin_to_cos=True)
        self.time_step_proj = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        self.joint_transformer_blocks = nn.ModuleList(
            [
                LavenderFlowTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_mmdit_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                LavenderFlowDiTTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for _ in range(self.config.num_single_dit_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, norm_type="no_norm", bias=False)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        # https://arxiv.org/abs/2309.16588
        # prevents artifacts in the attention maps
        self.register_tokens = nn.Parameter(torch.randn(1, 8, self.inner_dim) * 0.02)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        height, width = hidden_states.shape[-2:]

        # Apply patch embedding, timestep embedding, and project the caption embeddings.
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_step_embed(timestep).to(dtype=next(self.parameters()).dtype)
        temb = self.time_step_proj(temb)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [self.register_tokens.repeat(encoder_hidden_states.size(0), 1, 1), encoder_hidden_states], dim=1
        )

        # MMDiT blocks.
        for index_block, block in enumerate(self.joint_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, i=index_block
                )

        # Single DiT blocks that combine the `hidden_states` (image) and `encoder_hidden_states` (text)
        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    combined_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        combined_hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )

                else:
                    combined_hidden_states = block(hidden_states=combined_hidden_states, temb=temb)

            hidden_states = combined_hidden_states[:, encoder_seq_len:]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        out_channels = self.config.out_channels
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], out_channels, height * patch_size, width * patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
