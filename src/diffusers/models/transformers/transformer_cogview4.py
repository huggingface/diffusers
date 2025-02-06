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


from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.attention import FeedForward
from ...models.attention_processor import (
    Attention,
    AttentionProcessor,
    CogView4AttnProcessor,
)
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous
from ...utils import is_torch_version, logging
from ..embeddings import CogView3CombinedTimestepSizeEmbeddings, CogView4PatchEmbed
from ..modeling_outputs import Transformer2DModelOutput
from ..normalization import CogView3PlusAdaLayerNormZeroTextImage


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogView4TransformerBlock(nn.Module):
    r"""
    Transformer block used in [CogView](https://github.com/THUDM/CogView3) model.

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
    """

    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
    ):
        super().__init__()

        self.norm1 = CogView3PlusAdaLayerNormZeroTextImage(embedding_dim=time_embed_dim, dim=dim)
        self.adaln = self.norm1.linear
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-5,
            processor=CogView4AttnProcessor(),
        )

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def multi_modulate(self, hidden_states, encoder_hidden_states, factors) -> torch.Tensor:
        n_sample, n_type, h = factors[0].shape
        shift_factor, scale_factor = factors[0].view(-1, h), factors[1].view(-1, h)

        shift_factor_hidden_states, shift_factor_encoder_hidden_states = shift_factor.chunk(2, dim=0)
        scale_factor_hidden_states, scale_factor_encoder_hidden_states = scale_factor.chunk(2, dim=0)

        hidden_states = torch.addcmul(shift_factor_hidden_states, hidden_states, (1 + scale_factor_hidden_states))
        encoder_hidden_states = torch.addcmul(
            shift_factor_encoder_hidden_states, encoder_hidden_states, (1 + scale_factor_encoder_hidden_states)
        )

        return hidden_states, encoder_hidden_states

    def multi_gate(self, hidden_states, encoder_hidden_states, factor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        gate_factor = factor.view(-1, hidden_dim)
        gate_factor_hidden_states, gate_factor_encoder_hidden_states = gate_factor.chunk(2, dim=0)
        hidden_states = gate_factor_hidden_states * hidden_states
        encoder_hidden_states = gate_factor_encoder_hidden_states * encoder_hidden_states
        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        time_embedding: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, encoder_hidden_states_len, hidden_dim = encoder_hidden_states.shape
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states

        # time_embedding embedding, [n_sample, h]
        assert time_embedding is not None

        layernorm_factor = (
            self.adaln(time_embedding)
            .view(
                time_embedding.shape[0],
                6,
                2,
                hidden_states.shape[-1],
            )
            .permute(1, 2, 0, 3)
            .contiguous()
        )

        ##############################################################
        # Optional Input Layer norm
        hidden_states = self.layernorm(hidden_states)
        hidden_states, encoder_hidden_states = self.multi_modulate(
            hidden_states=hidden_states[:, encoder_hidden_states_len:],
            encoder_hidden_states=hidden_states[:, :encoder_hidden_states_len],
            factors=(layernorm_factor[0], layernorm_factor[1]),
        )
        hidden_states, encoder_hidden_states = self.attn1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states, encoder_hidden_states = self.multi_gate(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            factor=layernorm_factor[2],
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states += residual

        residual = hidden_states
        ##############################################################
        hidden_states = self.layernorm(hidden_states)
        hidden_states, encoder_hidden_states = self.multi_modulate(
            hidden_states=hidden_states[:, encoder_hidden_states_len:],
            encoder_hidden_states=hidden_states[:, :encoder_hidden_states_len],
            factors=(layernorm_factor[3], layernorm_factor[4]),
        )
        hidden_states = self.ff(hidden_states)
        encoder_hidden_states = self.ff(encoder_hidden_states)
        hidden_states, encoder_hidden_states = self.multi_gate(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            factor=layernorm_factor[5],
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states += residual

        ##############################################################
        hidden_states, encoder_hidden_states = (
            hidden_states[:, encoder_hidden_states_len:],
            hidden_states[:, :encoder_hidden_states_len],
        )
        return hidden_states, encoder_hidden_states


def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


class CogView4Transformer2DModel(ModelMixin, ConfigMixin):
    r"""
    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, defaults to `40`):
            The number of channels in each head.
        num_attention_heads (`int`, defaults to `64`):
            The number of heads to use for multi-head attention.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        condition_dim (`int`, defaults to `256`):
            The embedding dimension of the input SDXL-style resolution conditions (original_size, target_size,
            crop_coords).
        pos_embed_max_size (`int`, defaults to `128`):
            The maximum resolution of the positional embeddings, from which slices of shape `H x W` are taken and added
            to input patched latents, where `H` and `W` are the latent height and width respectively. A value of 128
            means that the maximum supported height and width for image generation is `128 * vae_scale_factor *
            patch_size => 128 * 8 * 2 => 2048`.
        sample_size (`int`, defaults to `128`):
            The base resolution of input latents. If height/width is not provided during generation, this value is used
            to determine the resolution as `sample_size * vae_scale_factor => 128 * 8 => 1024`
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogView4TransformerBlock", "CogView4PatchEmbed", "CogView4PatchEmbed"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        out_channels: int = 16,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        pos_embed_max_size: int = 128,
        sample_size: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # CogView3 uses 3 additional SDXL-like conditions - original_size, target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        self.pooled_projection_dim = 3 * 2 * condition_dim

        self.max_h = 256
        self.max_w = 256
        self.rope = self.prepare_rope(
            embed_dim=self.config.attention_head_dim, max_h=self.max_h, max_w=self.max_w, rotary_base=10000
        )

        self.layernorm = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-5)

        self.patch_embed = CogView4PatchEmbed(
            in_channels=in_channels,
            hidden_size=self.inner_dim,
            patch_size=patch_size,
            text_hidden_size=text_embed_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.time_condition_embed = CogView3CombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=self.pooled_projection_dim,
            timesteps_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogView4TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                )
                for _ in range(num_layers)
            ]
        )

        ######################################
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=time_embed_dim,
            elementwise_affine=False,
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

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @staticmethod
    def prepare_rope(embed_dim, max_h, max_w, rotary_base):
        dim_h = embed_dim // 2
        dim_w = embed_dim // 2
        h_inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)].float() / dim_h)
        )
        w_inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim_w, 2, dtype=torch.float32)[: (dim_w // 2)].float() / dim_w)
        )
        h_seq = torch.arange(max_h, dtype=h_inv_freq.dtype)
        w_seq = torch.arange(max_w, dtype=w_inv_freq.dtype)
        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)
        return (freqs_h, freqs_w)

    def get_rope_embedding(self, height, width, target_h, target_w, device):
        # Get pre-computed frequencies
        freqs_h, freqs_w = self.rope

        h_idx = torch.arange(height)
        w_idx = torch.arange(width)
        inner_h_idx = (h_idx * self.max_h) // target_h
        inner_w_idx = (w_idx * self.max_w) // target_w

        freqs_h = freqs_h[inner_h_idx].to(device)
        freqs_w = freqs_w[inner_w_idx].to(device)

        # Create position matrices for height and width
        # [height, 1, dim//4] and [1, width, dim//4]
        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)
        # Broadcast freqs_h and freqs_w to [height, width, dim//4]
        freqs_h = freqs_h.expand(height, width, -1)
        freqs_w = freqs_w.expand(height, width, -1)

        # Concatenate along last dimension to get [height, width, dim//2]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)

        freqs = torch.cat([freqs, freqs], dim=-1)  # [height, width, dim]
        freqs = freqs.reshape(height * width, -1)

        return freqs
        # return freqs.cos(), freqs.sin()

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        timestep: torch.LongTensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`CogView3PlusTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor`):
                Input `hidden_states` of shape `(batch size, channel, height, width)`.
            encoder_hidden_states (`torch.Tensor`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) of shape
                `(batch_size, sequence_len, text_embed_dim)`
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            original_size (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for original image size as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for target image size as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crop_coords (`torch.Tensor`):
                CogView3 uses SDXL-like micro-conditioning for crop coordinates as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            `torch.Tensor` or [`~models.transformer_2d.Transformer2DModelOutput`]:
                The denoised latents using provided inputs as conditioning.
        """
        batch_size, channel, height, width = hidden_states.shape
        patch_height, patch_width = height // self.config.patch_size, width // self.config.patch_size
        do_cfg = negative_prompt_embeds is not None

        if do_cfg:
            assert batch_size == prompt_embeds.shape[0] + negative_prompt_embeds.shape[0], (
                "batch size mismatch in CFG mode"
            )
        else:
            assert batch_size == prompt_embeds.shape[0], "batch size mismatch in non-CFG mode"

        # 1. RoPE
        image_rotary_emb = self.get_rope_embedding(
            patch_height, patch_width, target_h=patch_height, target_w=patch_width, device=hidden_states.device
        )

        # 2. Conditional embeddings
        temb = self.time_condition_embed(timestep, original_size, target_size, crop_coords, hidden_states.dtype)
        temb = F.silu(temb)
        temb_cond, temb_uncond = temb.chunk(2)
        hidden_states, prompt_embeds, negative_prompt_embeds = self.patch_embed(
            hidden_states, prompt_embeds, negative_prompt_embeds
        )
        hidden_states_cond, hidden_states_uncond = hidden_states.chunk(2)

        encoder_hidden_states_cond = prompt_embeds
        encoder_hidden_states_uncond = negative_prompt_embeds

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # TODO 微调使用
                ...
            else:
                hidden_states_cond, encoder_hidden_states_cond = block(
                    hidden_states=hidden_states_cond,
                    encoder_hidden_states=encoder_hidden_states_cond,
                    time_embedding=temb_cond,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states_uncond, encoder_hidden_states_uncond = block(
                    hidden_states=hidden_states_uncond,
                    encoder_hidden_states=encoder_hidden_states_uncond,
                    time_embedding=temb_uncond,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states_cond, encoder_hidden_states_cond = (
            self.norm_out(hidden_states_cond, temb_cond),
            self.norm_out(encoder_hidden_states_cond, temb_cond),
        )
        hidden_states_uncond, encoder_hidden_states_uncond = (
            self.norm_out(hidden_states_uncond, temb_uncond),
            self.norm_out(encoder_hidden_states_uncond, temb_uncond),
        )

        hidden_states_cond = self.proj_out(hidden_states_cond)
        hidden_states_uncond = self.proj_out(hidden_states_uncond)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states_cond = hidden_states_cond.reshape(
            shape=(hidden_states_cond.shape[0], height, width, self.out_channels, patch_size, patch_size)
        )
        hidden_states_cond = torch.einsum("nhwcpq->nchpwq", hidden_states_cond)
        output_cond = hidden_states_cond.reshape(
            shape=(hidden_states_cond.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        hidden_states_uncond = hidden_states_uncond.reshape(
            shape=(hidden_states_uncond.shape[0], height, width, self.out_channels, patch_size, patch_size)
        )
        hidden_states_uncond = torch.einsum("nhwcpq->nchpwq", hidden_states_uncond)
        output_uncond = hidden_states_uncond.reshape(
            shape=(hidden_states_uncond.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        if not return_dict:
            return (output_cond, output_uncond)
        return Transformer2DModelOutput(sample=output_cond), Transformer2DModelOutput(sample=output_uncond)
