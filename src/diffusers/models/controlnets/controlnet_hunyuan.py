# Copyright 2024 HunyuanDiT Authors, Qixun Wang and The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, logging
from ..attention_processor import AttentionProcessor
from ..embeddings import (
    HunyuanCombinedTimestepTextSizeStyleEmbedding,
    PatchEmbed,
    PixArtAlphaTextProjection,
)
from ..modeling_utils import ModelMixin
from ..transformers.hunyuan_transformer_2d import HunyuanDiTBlock
from .controlnet import Tuple, zero_module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class HunyuanControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]


class HunyuanDiT2DControlNetModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        sample_size=32,
        hidden_size=1152,
        transformer_num_layers: int = 40,
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 1024,
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
        use_style_cond_and_image_meta_size: bool = True,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim

        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim_t5,
            hidden_size=cross_attention_dim_t5 * 4,
            out_features=cross_attention_dim,
            act_fn="silu_fp32",
        )

        self.text_embedding_padding = nn.Parameter(
            torch.randn(text_len + text_len_t5, cross_attention_dim, dtype=torch.float32)
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
        )

        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            seq_len=text_len_t5,
            cross_attention_dim=cross_attention_dim_t5,
            use_style_cond_and_image_meta_size=use_style_cond_and_image_meta_size,
        )

        # controlnet_blocks
        self.controlnet_blocks = nn.ModuleList([])

        # HunyuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunyuanDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    activation_fn=activation_fn,
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    skip=False,  # always False as it is the first half of the model
                )
                for layer in range(transformer_num_layers // 2 - 1)
            ]
        )
        self.input_block = zero_module(nn.Linear(hidden_size, hidden_size))
        for _ in range(len(self.blocks)):
            controlnet_block = nn.Linear(hidden_size, hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

    @property
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
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers. If `processor` is a dict, the key needs to define the path to the
                corresponding cross attention processor. This is strongly recommended when setting trainable attention
                processors.
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

    @classmethod
    def from_transformer(
        cls, transformer, conditioning_channels=3, transformer_num_layers=None, load_weights_from_transformer=True
    ):
        config = transformer.config
        activation_fn = config.activation_fn
        attention_head_dim = config.attention_head_dim
        cross_attention_dim = config.cross_attention_dim
        cross_attention_dim_t5 = config.cross_attention_dim_t5
        hidden_size = config.hidden_size
        in_channels = config.in_channels
        mlp_ratio = config.mlp_ratio
        num_attention_heads = config.num_attention_heads
        patch_size = config.patch_size
        sample_size = config.sample_size
        text_len = config.text_len
        text_len_t5 = config.text_len_t5

        conditioning_channels = conditioning_channels
        transformer_num_layers = transformer_num_layers or config.transformer_num_layers

        controlnet = cls(
            conditioning_channels=conditioning_channels,
            transformer_num_layers=transformer_num_layers,
            activation_fn=activation_fn,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            cross_attention_dim_t5=cross_attention_dim_t5,
            hidden_size=hidden_size,
            in_channels=in_channels,
            mlp_ratio=mlp_ratio,
            num_attention_heads=num_attention_heads,
            patch_size=patch_size,
            sample_size=sample_size,
            text_len=text_len,
            text_len_t5=text_len_t5,
        )
        if load_weights_from_transformer:
            key = controlnet.load_state_dict(transformer.state_dict(), strict=False)
            logger.warning(f"controlnet load from Hunyuan-DiT. missing_keys: {key[0]}")
        return controlnet

    def forward(
        self,
        hidden_states,
        timestep,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        return_dict=True,
    ):
        """
        The [`HunyuanDiT2DControlNetModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        controlnet_cond ( `torch.Tensor` ):
            The conditioning input to ControlNet.
        conditioning_scale ( `float` ):
            Indicate the conditioning scale.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of `BertModel`.
        text_embedding_mask: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of `BertModel`.
        encoder_hidden_states_t5 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of T5 Text Encoder.
        text_embedding_mask_t5: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of T5 Text Encoder.
        image_meta_size (torch.Tensor):
            Conditional embedding indicate the image sizes
        style: torch.Tensor:
            Conditional embedding indicate the style
        image_rotary_emb (`torch.Tensor`):
            The image rotary embeddings to apply on query and key tensors during attention calculation.
        return_dict: bool
            Whether to return a dictionary.
        """

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # b,c,H,W -> b, N, C

        # 2. pre-process
        hidden_states = hidden_states + self.input_block(self.pos_embed(controlnet_cond))

        temb = self.time_extra_emb(
            timestep, encoder_hidden_states_t5, image_meta_size, style, hidden_dtype=timestep.dtype
        )  # [B, D]

        # text projection
        batch_size, sequence_length, _ = encoder_hidden_states_t5.shape
        encoder_hidden_states_t5 = self.text_embedder(
            encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
        )
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, sequence_length, -1)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1)
        text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
        text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()

        encoder_hidden_states = torch.where(text_embedding_mask, encoder_hidden_states, self.text_embedding_padding)

        block_res_samples = ()
        for layer, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )  # (N, L, D)

            block_res_samples = block_res_samples + (hidden_states,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        # 6. scaling
        controlnet_block_res_samples = [sample * conditioning_scale for sample in controlnet_block_res_samples]

        if not return_dict:
            return (controlnet_block_res_samples,)

        return HunyuanControlNetOutput(controlnet_block_samples=controlnet_block_res_samples)


class HunyuanDiT2DMultiControlNetModel(ModelMixin):
    r"""
    `HunyuanDiT2DMultiControlNetModel` wrapper class for Multi-HunyuanDiT2DControlNetModel

    This module is a wrapper for multiple instances of the `HunyuanDiT2DControlNetModel`. The `forward()` API is
    designed to be compatible with `HunyuanDiT2DControlNetModel`.

    Args:
        controlnets (`List[HunyuanDiT2DControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `HunyuanDiT2DControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states,
        timestep,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        return_dict=True,
    ):
        """
        The [`HunyuanDiT2DControlNetModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        controlnet_cond ( `torch.Tensor` ):
            The conditioning input to ControlNet.
        conditioning_scale ( `float` ):
            Indicate the conditioning scale.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of `BertModel`.
        text_embedding_mask: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of `BertModel`.
        encoder_hidden_states_t5 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of T5 Text Encoder.
        text_embedding_mask_t5: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of T5 Text Encoder.
        image_meta_size (torch.Tensor):
            Conditional embedding indicate the image sizes
        style: torch.Tensor:
            Conditional embedding indicate the style
        image_rotary_emb (`torch.Tensor`):
            The image rotary embeddings to apply on query and key tensors during attention calculation.
        return_dict: bool
            Whether to return a dictionary.
        """
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            block_samples = controlnet(
                hidden_states=hidden_states,
                timestep=timestep,
                controlnet_cond=image,
                conditioning_scale=scale,
                encoder_hidden_states=encoder_hidden_states,
                text_embedding_mask=text_embedding_mask,
                encoder_hidden_states_t5=encoder_hidden_states_t5,
                text_embedding_mask_t5=text_embedding_mask_t5,
                image_meta_size=image_meta_size,
                style=style,
                image_rotary_emb=image_rotary_emb,
                return_dict=return_dict,
            )

            # merge samples
            if i == 0:
                control_block_samples = block_samples
            else:
                control_block_samples = [
                    control_block_sample + block_sample
                    for control_block_sample, block_sample in zip(control_block_samples[0], block_samples[0])
                ]
                control_block_samples = (control_block_samples,)

        return control_block_samples
