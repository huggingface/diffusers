# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ..attention import BasicTransformerBlock, SkipFFTransformerBlock
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ..embeddings import TimestepEmbedding, get_timestep_embedding
from ..modeling_utils import ModelMixin
from ..normalization import GlobalResponseNorm, RMSNorm
from ..resnet import Downsample2D, Upsample2D


class UVit2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # global config
        hidden_size: int = 1024,
        use_bias: bool = False,
        hidden_dropout: float = 0.0,
        # conditioning dimensions
        cond_embed_dim: int = 768,
        micro_cond_encode_dim: int = 256,
        micro_cond_embed_dim: int = 1280,
        encoder_hidden_size: int = 768,
        # num tokens
        vocab_size: int = 8256,  # codebook_size + 1 (for the mask token) rounded
        codebook_size: int = 8192,
        # `UVit2DConvEmbed`
        in_channels: int = 768,
        block_out_channels: int = 768,
        num_res_blocks: int = 3,
        downsample: bool = False,
        upsample: bool = False,
        block_num_heads: int = 12,
        # `TransformerLayer`
        num_hidden_layers: int = 22,
        num_attention_heads: int = 16,
        # `Attention`
        attention_dropout: float = 0.0,
        # `FeedForward`
        intermediate_size: int = 2816,
        # `Norm`
        layer_norm_eps: float = 1e-6,
        ln_elementwise_affine: bool = True,
        sample_size: int = 64,
    ):
        super().__init__()

        self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=use_bias)
        self.encoder_proj_layer_norm = RMSNorm(hidden_size, layer_norm_eps, ln_elementwise_affine)

        self.embed = UVit2DConvEmbed(
            in_channels, block_out_channels, vocab_size, ln_elementwise_affine, layer_norm_eps, use_bias
        )

        self.cond_embed = TimestepEmbedding(
            micro_cond_embed_dim + cond_embed_dim, hidden_size, sample_proj_bias=use_bias
        )

        self.down_block = UVitBlock(
            block_out_channels,
            num_res_blocks,
            hidden_size,
            hidden_dropout,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            block_num_heads,
            attention_dropout,
            downsample,
            False,
        )

        self.project_to_hidden_norm = RMSNorm(block_out_channels, layer_norm_eps, ln_elementwise_affine)
        self.project_to_hidden = nn.Linear(block_out_channels, hidden_size, bias=use_bias)

        self.transformer_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=hidden_size // num_attention_heads,
                    dropout=hidden_dropout,
                    cross_attention_dim=hidden_size,
                    attention_bias=use_bias,
                    norm_type="ada_norm_continuous",
                    ada_norm_continous_conditioning_embedding_dim=hidden_size,
                    norm_elementwise_affine=ln_elementwise_affine,
                    norm_eps=layer_norm_eps,
                    ada_norm_bias=use_bias,
                    ff_inner_dim=intermediate_size,
                    ff_bias=use_bias,
                    attention_out_bias=use_bias,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.project_from_hidden_norm = RMSNorm(hidden_size, layer_norm_eps, ln_elementwise_affine)
        self.project_from_hidden = nn.Linear(hidden_size, block_out_channels, bias=use_bias)

        self.up_block = UVitBlock(
            block_out_channels,
            num_res_blocks,
            hidden_size,
            hidden_dropout,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            block_num_heads,
            attention_dropout,
            downsample=False,
            upsample=upsample,
        )

        self.mlm_layer = ConvMlmLayer(
            block_out_channels, in_channels, use_bias, ln_elementwise_affine, layer_norm_eps, codebook_size
        )

        self.gradient_checkpointing = False

    def forward(self, input_ids, encoder_hidden_states, pooled_text_emb, micro_conds, cross_attention_kwargs=None):
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        micro_cond_embeds = get_timestep_embedding(
            micro_conds.flatten(), self.config.micro_cond_encode_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )

        micro_cond_embeds = micro_cond_embeds.reshape((input_ids.shape[0], -1))

        pooled_text_emb = torch.cat([pooled_text_emb, micro_cond_embeds], dim=1)
        pooled_text_emb = pooled_text_emb.to(dtype=self.dtype)
        pooled_text_emb = self.cond_embed(pooled_text_emb).to(encoder_hidden_states.dtype)

        hidden_states = self.embed(input_ids)

        hidden_states = self.down_block(
            hidden_states,
            pooled_text_emb=pooled_text_emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        hidden_states = self.project_to_hidden_norm(hidden_states)
        hidden_states = self.project_to_hidden(hidden_states)

        for layer in self.transformer_layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def layer_(*args):
                    return checkpoint(layer, *args)

            else:
                layer_ = layer

            hidden_states = layer_(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs={"pooled_text_emb": pooled_text_emb},
            )

        hidden_states = self.project_from_hidden_norm(hidden_states)
        hidden_states = self.project_from_hidden(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        hidden_states = self.up_block(
            hidden_states,
            pooled_text_emb=pooled_text_emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        logits = self.mlm_layer(hidden_states)

        return logits

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

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)


class UVit2DConvEmbed(nn.Module):
    def __init__(self, in_channels, block_out_channels, vocab_size, elementwise_affine, eps, bias):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, in_channels)
        self.layer_norm = RMSNorm(in_channels, eps, elementwise_affine)
        self.conv = nn.Conv2d(in_channels, block_out_channels, kernel_size=1, bias=bias)

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = self.conv(embeddings)
        return embeddings


class UVitBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_res_blocks: int,
        hidden_size,
        hidden_dropout,
        ln_elementwise_affine,
        layer_norm_eps,
        use_bias,
        block_num_heads,
        attention_dropout,
        downsample: bool,
        upsample: bool,
    ):
        super().__init__()

        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
            )
        else:
            self.downsample = None

        self.res_blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    channels,
                    layer_norm_eps,
                    ln_elementwise_affine,
                    use_bias,
                    hidden_dropout,
                    hidden_size,
                )
                for i in range(num_res_blocks)
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                SkipFFTransformerBlock(
                    channels,
                    block_num_heads,
                    channels // block_num_heads,
                    hidden_size,
                    use_bias,
                    attention_dropout,
                    channels,
                    attention_bias=use_bias,
                    attention_out_bias=use_bias,
                )
                for _ in range(num_res_blocks)
            ]
        )

        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
            )
        else:
            self.upsample = None

    def forward(self, x, pooled_text_emb, encoder_hidden_states, cross_attention_kwargs):
        if self.downsample is not None:
            x = self.downsample(x)

        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks):
            x = res_block(x, pooled_text_emb)

            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
            x = attention_block(
                x, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs
            )
            x = x.permute(0, 2, 1).view(batch_size, channels, height, width)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class ConvNextBlock(nn.Module):
    def __init__(
        self, channels, layer_norm_eps, ln_elementwise_affine, use_bias, hidden_dropout, hidden_size, res_ffn_factor=4
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=use_bias,
        )
        self.norm = RMSNorm(channels, layer_norm_eps, ln_elementwise_affine)
        self.channelwise_linear_1 = nn.Linear(channels, int(channels * res_ffn_factor), bias=use_bias)
        self.channelwise_act = nn.GELU()
        self.channelwise_norm = GlobalResponseNorm(int(channels * res_ffn_factor))
        self.channelwise_linear_2 = nn.Linear(int(channels * res_ffn_factor), channels, bias=use_bias)
        self.channelwise_dropout = nn.Dropout(hidden_dropout)
        self.cond_embeds_mapper = nn.Linear(hidden_size, channels * 2, use_bias)

    def forward(self, x, cond_embeds):
        x_res = x

        x = self.depthwise(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.channelwise_linear_1(x)
        x = self.channelwise_act(x)
        x = self.channelwise_norm(x)
        x = self.channelwise_linear_2(x)
        x = self.channelwise_dropout(x)

        x = x.permute(0, 3, 1, 2)

        x = x + x_res

        scale, shift = self.cond_embeds_mapper(F.silu(cond_embeds)).chunk(2, dim=1)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        return x


class ConvMlmLayer(nn.Module):
    def __init__(
        self,
        block_out_channels: int,
        in_channels: int,
        use_bias: bool,
        ln_elementwise_affine: bool,
        layer_norm_eps: float,
        codebook_size: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(block_out_channels, in_channels, kernel_size=1, bias=use_bias)
        self.layer_norm = RMSNorm(in_channels, layer_norm_eps, ln_elementwise_affine)
        self.conv2 = nn.Conv2d(in_channels, codebook_size, kernel_size=1, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        logits = self.conv2(hidden_states)
        return logits
