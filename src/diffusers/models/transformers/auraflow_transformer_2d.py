# Copyright 2024 AuraFlow Authors, The HuggingFace Team. All rights reserved.
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


from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention_processor import (
    Attention,
    AttentionProcessor,
    AuraFlowAttnProcessor2_0,
    FusedAuraFlowAttnProcessor2_0,
)
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormZero, FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Taken from the original aura flow inference code.
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# Aura Flow patch embed doesn't use convs for projections.
# Additionally, it uses learned positional embeddings.
class AuraFlowPatchEmbed(nn.Module):
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

    def pe_selection_index_based_on_dim(self, h, w):
        # select subset of positional embedding based on H, W, where H, W is size of latent
        # PE will be viewed as 2d-grid, and H/p x W/p of the PE will be selected
        # because original input are in flattened format, we have to flatten this 2d grid as well.
        h_p, w_p = h // self.patch_size, w // self.patch_size
        original_pe_indexes = torch.arange(self.pos_embed.shape[1])
        h_max, w_max = int(self.pos_embed_max_size**0.5), int(self.pos_embed_max_size**0.5)
        original_pe_indexes = original_pe_indexes.view(h_max, w_max)
        starth = h_max // 2 - h_p // 2
        endh = starth + h_p
        startw = w_max // 2 - w_p // 2
        endw = startw + w_p
        original_pe_indexes = original_pe_indexes[starth:endh, startw:endw]
        return original_pe_indexes.flatten()

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
        pe_index = self.pe_selection_index_based_on_dim(height, width)
        return latent + self.pos_embed[:, pe_index]


# Taken from the original Aura flow inference code.
# Our feedforward only has GELU but Aura uses SiLU.
class AuraFlowFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        final_hidden_dim = int(2 * hidden_dim / 3)
        final_hidden_dim = find_multiple(final_hidden_dim, 256)

        self.linear_1 = nn.Linear(dim, final_hidden_dim, bias=False)
        self.linear_2 = nn.Linear(dim, final_hidden_dim, bias=False)
        self.out_projection = nn.Linear(final_hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.linear_1(x)) * self.linear_2(x)
        x = self.out_projection(x)
        return x


class AuraFlowPreFinalBlock(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


@maybe_allow_in_graph
class AuraFlowSingleTransformerBlock(nn.Module):
    """Similar to `AuraFlowJointTransformerBlock` with a single DiT instead of an MMDiT."""

    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")

        processor = AuraFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="fp32_layer_norm",
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = AuraFlowFeedForward(dim, dim * 4)

    def forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor):
        residual = hidden_states

        # Norm + Projection.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Attention.
        attn_output = self.attn(hidden_states=norm_hidden_states)

        # Process attention outputs for the `hidden_states`.
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(hidden_states)
        hidden_states = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = residual + hidden_states

        return hidden_states


@maybe_allow_in_graph
class AuraFlowJointTransformerBlock(nn.Module):
    r"""
    Transformer block for Aura Flow. Similar to SD3 MMDiT. Differences (non-exhaustive):

        * QK Norm in the attention blocks
        * No bias in the attention blocks
        * Most LayerNorms are in FP32

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        is_last (`bool`): Boolean to determine if this is the last block in the model.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")
        self.norm1_context = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")

        processor = AuraFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            added_proj_bias=False,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="fp32_layer_norm",
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
            context_pre_only=False,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = AuraFlowFeedForward(dim, dim * 4)
        self.norm2_context = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff_context = AuraFlowFeedForward(dim, dim * 4)

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
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
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
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


class AuraFlowTransformer2DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A 2D Transformer model as introduced in AuraFlow (https://blog.fal.ai/auraflow/).

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_mmdit_layers (`int`, *optional*, defaults to 4): The number of layers of MMDiT Transformer blocks to use.
        num_single_dit_layers (`int`, *optional*, defaults to 4):
            The number of layers of Transformer blocks to use. These blocks use concatenated image and text
            representations.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        out_channels (`int`, defaults to 16): Number of output channels.
        pos_embed_max_size (`int`, defaults to 4096): Maximum positions to embed from the image latents.
    """

    _no_split_modules = ["AuraFlowJointTransformerBlock", "AuraFlowSingleTransformerBlock", "AuraFlowPatchEmbed"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
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

        self.pos_embed = AuraFlowPatchEmbed(
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
                AuraFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_mmdit_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                AuraFlowSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for _ in range(self.config.num_single_dit_layers)
            ]
        )

        self.norm_out = AuraFlowPreFinalBlock(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        # https://arxiv.org/abs/2309.16588
        # prevents artifacts in the attention maps
        self.register_tokens = nn.Parameter(torch.randn(1, 8, self.inner_dim) * 0.02)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedAuraFlowAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAuraFlowAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

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
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

        # Single DiT blocks that combine the `hidden_states` (image) and `encoder_hidden_states` (text)
        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    combined_hidden_states = self._gradient_checkpointing_func(
                        block,
                        combined_hidden_states,
                        temb,
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
