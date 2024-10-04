# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import FeedForward
from ...models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous
from ...utils import is_torch_version, logging
from ..embeddings import CogView3PlusPatchEmbed, CombinedTimestepTextProjEmbeddings
from ..modeling_outputs import Transformer2DModelOutput
from ..normalization import CogView3PlusAdaLayerNormZeroTextImage


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogView3PlusTransformerBlock(nn.Module):
    """
    Updated CogView3 Transformer Block to align with AdalnAttentionMixin style, simplified with qk_ln always True.
    """

    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
    ):
        super().__init__()

        # attn_mixin.adaln_modules + layer[i].input_layernorm
        self.norm1 = CogView3PlusAdaLayerNormZeroTextImage(embedding_dim=time_embed_dim, dim=dim)

        self.attn = Attention(
            query_dim=self.dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=self.dim,
            bias=True,
            qk_norm="layer_norm",
            layrnorm_elementwise_affine=False,
            eps=1e-6,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        text_length: int,
    ) -> torch.Tensor:
        encoder_hidden_states, hidden_states = hidden_states[:, :text_length], hidden_states[:, text_length:]

        # norm1
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, emb)

        # Attention
        attn_input = torch.cat((norm_encoder_hidden_states, norm_hidden_states), dim=1)
        attn_output = self.attn(hidden_states=attn_input)
        context_attn_output, attn_output = attn_output[:, :text_length], attn_output[:, text_length:]

        # Apply gate to attention output
        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        norm_hidden_states = torch.cat((norm_encoder_hidden_states, norm_hidden_states), dim=1)

        ff_output = self.ff(norm_hidden_states)

        # Apply gate to MLP output
        context_ff_output, ff_output = ff_output[:, :text_length], ff_output[:, text_length:]

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        hidden_states = torch.cat((encoder_hidden_states, hidden_states), dim=1)

        return hidden_states


class CogView3PlusTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in CogView3.

    Reference: https://arxiv.org/abs/2403.05121
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        out_channels: int = 16,
        encoder_hidden_states_dim: int = 4096,
        pooled_projection_dim: int = 1536,
        pos_embed_max_size: int = 128,
        time_embed_dim: int = 512,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = CogView3PlusPatchEmbed(
            in_channels=self.config.in_channels,
            hidden_size=self.inner_dim,
            patch_size=self.config.patch_size,
            text_hidden_size=self.config.encoder_hidden_states_dim,
            pos_embed_max_size=self.config.pos_embed_max_size,
        )

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.config.time_embed_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
            timesteps_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogView3PlusTransformerBlock(
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.config.time_embed_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

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

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedJointAttnProcessor2_0
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

        self.set_attn_processor(FusedJointAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`CogView3PlusTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor`): Input `hidden_states`.
            timestep (`torch.LongTensor`): Indicates denoising step.
            y (`torch.LongTensor`, *optional*): 标签输入，用于获取标签嵌入。
            block_controlnet_hidden_states: (`list` of `torch.Tensor`): A list of tensors for residuals.
            joint_attention_kwargs (`dict`, *optional*): Additional kwargs for the attention processor.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a `Transformer2DModelOutput`.

        Returns:
            Output tensor or `Transformer2DModelOutput`.
        """

        height, width = hidden_states.shape[-2:]
        text_length = encoder_hidden_states.shape[1]

        hidden_states = self.pos_embed(
            hidden_states, encoder_hidden_states
        )  # takes care of adding positional embeddings too.
        emb = self.time_text_embed(timestep, pooled_projections)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    emb,
                    text_length,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    emb=emb,
                    text_length=text_length,
                )

        hidden_states = hidden_states[:, text_length:]
        hidden_states = self.norm_out(hidden_states, emb)
        hidden_states = self.proj_out(hidden_states)  # (batch_size, height*width, patch_size*patch_size*out_channels)
        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, self.out_channels, patch_size, patch_size)
        )
        hidden_states = torch.einsum("nhwcpq->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
