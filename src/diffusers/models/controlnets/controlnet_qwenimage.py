# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import (
    BaseOutput,
    apply_lora_scale,
    deprecate,
    logging,
)
from ..attention import AttentionMixin
from ..cache_utils import CacheMixin
from ..controlnets.controlnet import zero_module
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
    RMSNorm,
    compute_text_seq_len_from_mask,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class QwenImageControlNetOutput(BaseOutput):
    controlnet_block_samples: tuple[torch.Tensor]


class QwenImageControlNetModel(
    ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        extra_condition_channels: int = 0,  # for controlnet-inpainting
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # controlnet_blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))
        self.controlnet_x_embedder = zero_module(
            torch.nn.Linear(in_channels + extra_condition_channels, self.inner_dim)
        )

        self.gradient_checkpointing = False

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 5,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
        extra_condition_channels: int = 0,
    ):
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads
        config["extra_condition_channels"] = extra_condition_channels

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.img_in.load_state_dict(transformer.img_in.state_dict())
            controlnet.txt_in.load_state_dict(transformer.txt_in.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.FloatTensor | Transformer2DModelOutput:
        """
        The [`QwenImageControlNetModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask for the encoder hidden states. Expected to have 1.0 for valid tokens and 0.0 for padding tokens.
                Used in the attention processor to prevent attending to padding tokens. The mask can have any pattern
                (not just contiguous valid tokens followed by padding) since it's applied element-wise in attention.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_shapes (`list[tuple[int, int, int]]`, *optional*):
                Image shapes for RoPE computation.
            txt_seq_lens (`list[int]`, *optional*):
                **Deprecated**. Not needed anymore, we use `encoder_hidden_states` instead to infer text sequence
                length.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            If `return_dict` is True, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a `tuple` where
            the first element is the controlnet block samples.
        """
        # Handle deprecated txt_seq_lens parameter
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` to `QwenImageControlNetModel.forward()` is deprecated and will be removed in "
                "version 0.39.0. The text sequence length is now automatically inferred from `encoder_hidden_states` "
                "and `encoder_hidden_states_mask`.",
                standard_warn=False,
            )

        hidden_states = self.img_in(hidden_states)

        # add
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        temb = self.time_text_embed(timestep, hidden_states)

        # Use the encoder_hidden_states sequence length for RoPE computation and normalize mask
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Construct joint attention mask once to avoid reconstructing in every block
        block_attention_kwargs = joint_attention_kwargs.copy() if joint_attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            # Build joint mask: [text_mask, all_ones_for_image]
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        block_samples = ()
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,  # Don't pass encoder_hidden_states_mask (using attention_mask instead)
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead)
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                )
            block_samples = block_samples + (hidden_states,)

        # controlnet block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples

        if not return_dict:
            return controlnet_block_samples

        return QwenImageControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
        )


class QwenImageMultiControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    `QwenImageMultiControlNetModel` wrapper class for Multi-QwenImageControlNetModel

    This module is a wrapper for multiple instances of the `QwenImageControlNetModel`. The `forward()` API is designed
    to be compatible with `QwenImageControlNetModel`.

    Args:
        controlnets (`list[QwenImageControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `QwenImageControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: list[torch.tensor],
        conditioning_scale: list[float],
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> QwenImageControlNetOutput | tuple:
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` to `QwenImageMultiControlNetModel.forward()` is deprecated and will be "
                "removed in version 0.39.0. The text sequence length is now automatically inferred from "
                "`encoder_hidden_states` and `encoder_hidden_states_mask`.",
                standard_warn=False,
            )
        # ControlNet-Union with multiple conditions
        # only load one ControlNet for saving memories
        if len(self.nets) == 1:
            controlnet = self.nets[0]

            for i, (image, scale) in enumerate(zip(controlnet_cond, conditioning_scale)):
                block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    conditioning_scale=scale,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                else:
                    if block_samples is not None and control_block_samples is not None:
                        control_block_samples = [
                            control_block_sample + block_sample
                            for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                        ]
        else:
            raise ValueError("QwenImageMultiControlNetModel only supports a single controlnet-union now.")

        return control_block_samples
