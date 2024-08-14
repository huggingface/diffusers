# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional

from torch import nn

from ..modeling_outputs import Transformer2DModelOutput
from .transformer_2d import Transformer2DModel


class DualTransformer2DModel(nn.Module):
    """
    Dual transformer wrapper that combines two `Transformer2DModel`s for mixed inference.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
    ):
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    sample_size=sample_size,
                    num_vector_embeds=num_vector_embeds,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                )
                for _ in range(2)
            ]
        )

        # Variables that can be set by a pipeline:

        # The ratio of transformer1 to transformer2's output states to be combined during inference
        self.mix_ratio = 0.5

        # The shape of `encoder_hidden_states` is expected to be
        # `(batch_size, condition_lengths[0]+condition_lengths[1], num_features)`
        self.condition_lengths = [77, 257]

        # Which transformer to use to encode which condition.
        # E.g. `(1, 0)` means that we'll use `transformers[1](conditions[0])` and `transformers[0](conditions[1])`
        self.transformer_index_for_condition = [1, 0]

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.Tensor` of shape `(batch size, channel, height, width)`): Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            attention_mask (`torch.Tensor`, *optional*):
                Optional attention mask to be applied in Attention.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformers.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformers.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        input_states = hidden_states

        encoded_states = []
        tokens_start = 0
        # attention_mask is not used yet
        for i in range(2):
            # for each of the two transformers, pass the corresponding condition tokens
            condition_state = encoder_hidden_states[:, tokens_start : tokens_start + self.condition_lengths[i]]
            transformer_index = self.transformer_index_for_condition[i]
            encoded_state = self.transformers[transformer_index](
                input_states,
                encoder_hidden_states=condition_state,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            encoded_states.append(encoded_state - input_states)
            tokens_start += self.condition_lengths[i]

        output_states = encoded_states[0] * self.mix_ratio + encoded_states[1] * (1 - self.mix_ratio)
        output_states = output_states + input_states

        if not return_dict:
            return (output_states,)

        return Transformer2DModelOutput(sample=output_states)
