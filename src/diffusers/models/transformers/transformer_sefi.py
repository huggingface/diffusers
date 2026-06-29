# Copyright 2026 SeFi-Image Authors and The HuggingFace Team. All rights reserved.
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
from typing import Any

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput, apply_lora_scale
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from .transformer_flux2 import Flux2Transformer2DModel


@dataclass
class SeFiTransformer2DModelOutput(BaseOutput):
    """
    Output of [`SeFiTransformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, image_sequence_length, out_channels)`):
            Predicted velocity for packed semantic and texture latents.
    """

    sample: torch.Tensor


class SeFiDualTimestepEmbeddings(nn.Module):
    """Dual semantic/texture timestep embedding used by SeFi-Image."""

    def __init__(self, in_channels: int, embedding_dim: int, bias: bool = False):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"`embedding_dim` must be even for dual timestep embeddings, got {embedding_dim}.")

        half_dim = embedding_dim // 2
        self.time_proj = Timesteps(
            num_channels=int(in_channels),
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.semantic_embedder = TimestepEmbedding(
            in_channels=int(in_channels),
            time_embed_dim=half_dim,
            sample_proj_bias=bias,
        )
        self.texture_embedder = TimestepEmbedding(
            in_channels=int(in_channels),
            time_embed_dim=half_dim,
            sample_proj_bias=bias,
        )

    def forward(self, timestep_sem: torch.Tensor, timestep_tex: torch.Tensor) -> torch.Tensor:
        sem_proj = self.time_proj(timestep_sem)
        tex_proj = self.time_proj(timestep_tex)
        sem_emb = self.semantic_embedder(sem_proj.to(timestep_sem.dtype))
        tex_emb = self.texture_embedder(tex_proj.to(timestep_tex.dtype))
        return torch.cat([sem_emb, tex_emb], dim=-1)


class SeFiTransformer2DModel(ModelMixin, ConfigMixin):
    """
    SeFi-Image transformer with explicit semantic and texture timestep conditioning.

    SeFi-Image reuses a Flux2-style MMDiT backbone, but replaces the single timestep embedding with a dual embedding:
    one timestep for the semantic latent stream and one timestep for the texture latent stream.

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size of the Flux2 backbone.
        in_channels (`int`, defaults to `128`):
            Number of packed latent channels. This is `semantic_channels + texture_channels`.
        out_channels (`int`, *optional*):
            Number of output packed latent channels. Defaults to `in_channels`.
        num_layers (`int`, defaults to `4`):
            Number of double-stream transformer layers.
        num_single_layers (`int`, defaults to `12`):
            Number of single-stream transformer layers.
        attention_head_dim (`int`, defaults to `128`):
            Dimension per attention head.
        num_attention_heads (`int`, defaults to `16`):
            Number of attention heads.
        joint_attention_dim (`int`, defaults to `6144`):
            Dimension of the concatenated Qwen3-VL hidden states.
        timestep_guidance_channels (`int`, defaults to `256`):
            Number of channels for sinusoidal timestep projection.
        mlp_ratio (`float`, defaults to `3.0`):
            MLP expansion ratio in transformer blocks.
        axes_dims_rope (`tuple[int, ...]`, defaults to `(32, 32, 32, 32)`):
            RoPE dimensions for Flux2 positional embeddings.
        rope_theta (`int`, defaults to `2000`):
            RoPE theta.
        eps (`float`, defaults to `1e-6`):
            Normalization epsilon.
        text_input_dim (`int`, *optional*):
            Expected text embedding dimension. Defaults to `joint_attention_dim`.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 4,
        num_single_layers: int = 12,
        attention_head_dim: int = 128,
        num_attention_heads: int = 16,
        joint_attention_dim: int = 6144,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        text_input_dim: int | None = None,
    ):
        super().__init__()

        text_input_dim = joint_attention_dim if text_input_dim is None else text_input_dim
        if int(text_input_dim) != int(joint_attention_dim):
            raise ValueError(
                f"`text_input_dim` must match `joint_attention_dim`, got {text_input_dim} and {joint_attention_dim}."
            )

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.backbone = Flux2Transformer2DModel(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            guidance_embeds=False,
        )
        # The reference SeFi transformer deletes Flux2's timestep/guidance embedder and stores only the dual embedder.
        self.backbone.time_guidance_embed = nn.Identity()
        self.dual_time_embed = SeFiDualTimestepEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
        )
        self.gradient_checkpointing = False

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_sem: torch.Tensor,
        timestep_tex: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | SeFiTransformer2DModelOutput:
        """
        The [`SeFiTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor`):
                Packed semantic and texture latents of shape `(batch_size, image_sequence_length, in_channels)`.
            timestep_sem (`torch.Tensor`):
                Semantic stream timesteps, normalized to the Diffusers convention where `1.0` corresponds to `1000`.
            timestep_tex (`torch.Tensor`):
                Texture stream timesteps, normalized to the Diffusers convention where `1.0` corresponds to `1000`.
            encoder_hidden_states (`torch.Tensor`):
                Text conditioning embeddings.
            txt_ids (`torch.Tensor`):
                Text token position ids.
            img_ids (`torch.Tensor`):
                Image token position ids.
            joint_attention_kwargs (`dict`, *optional*):
                Keyword arguments forwarded to attention processors.
            return_dict (`bool`, defaults to `True`):
                Whether to return [`SeFiTransformer2DModelOutput`] or a tuple.

        Returns:
            [`SeFiTransformer2DModelOutput`] or `tuple`:
                Predicted semantic and texture latent velocities.
        """

        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep_sem = timestep_sem.to(hidden_states.dtype) * 1000
        timestep_tex = timestep_tex.to(hidden_states.dtype) * 1000
        temb = self.dual_time_embed(timestep_sem, timestep_tex)

        double_stream_mod_img = self.backbone.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.backbone.double_stream_modulation_txt(temb)
        single_stream_mod = self.backbone.single_stream_modulation(temb)

        hidden_states = self.backbone.x_embedder(hidden_states)
        encoder_hidden_states = self.backbone.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.backbone.pos_embed(img_ids)
        text_rotary_emb = self.backbone.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        for block in self.backbone.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.backbone.single_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.backbone.norm_out(hidden_states, temb)
        output = self.backbone.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return SeFiTransformer2DModelOutput(sample=output)
