from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
)

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


def add_special_tokens(hidden_states, attention_mask, sos_token, eos_token):
    batch_size = hidden_states.shape[0]

    if attention_mask is not None:
        # Add two more steps to attn mask
        new_attn_mask_step = attention_mask.new_ones((batch_size, 1))
        attention_mask = torch.concat([new_attn_mask_step, attention_mask, new_attn_mask_step], dim=-1)

    # Add the SOS / EOS tokens at the start / end of the sequence respectively
    sos_token = sos_token.expand(batch_size, 1, -1)
    eos_token = eos_token.expand(batch_size, 1, -1)
    hidden_states = torch.concat([sos_token, hidden_states, eos_token], dim=1)
    return hidden_states, attention_mask


@dataclass
class AudioLDM2ProjectionModelOutput(BaseOutput):
    """
    Args:
    Class for AudioLDM2 projection layer's outputs.
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states obtained by linearly projecting the hidden-states for each of the text
             encoders and subsequently concatenating them together.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices, formed by concatenating the attention masks
             for the two text encoders together. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    """

    hidden_states: torch.FloatTensor
    attention_mask: Optional[torch.LongTensor] = None


class AudioLDM2PreTrainedModel(PreTrainedModel):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


class AudioLDM2ProjectionModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, text_encoder_1_dim, text_encoder_2_dim, langauge_model_dim):
        super().__init__()
        # additional projection layers
        self.clap_projection = nn.Linear(text_encoder_1_dim, langauge_model_dim)
        self.t5_projection = nn.Linear(text_encoder_2_dim, langauge_model_dim)

        # learnable SOS / EOS token embeddings
        self.clap_sos_embed = nn.Parameter(torch.ones(langauge_model_dim))
        self.clap_eos_embed = nn.Parameter(torch.ones(langauge_model_dim))

        self.t5_sos_embed = nn.Parameter(torch.ones(langauge_model_dim))
        self.t5_eos_embed = nn.Parameter(torch.ones(langauge_model_dim))

    def forward(
        self,
        clap_hidden_states: Optional[torch.FloatTensor] = None,
        t5_hidden_states: Optional[torch.FloatTensor] = None,
        clap_attention_mask: Optional[torch.LongTensor] = None,
        t5_attention_mask: Optional[torch.LongTensor] = None,
    ):
        clap_hidden_states = self.clap_projection(clap_hidden_states)
        clap_hidden_states, clap_attention_mask = add_special_tokens(
            clap_hidden_states, clap_attention_mask, sos_token=self.clap_sos_embed, eos_token=self.clap_eos_embed
        )

        t5_hidden_states = self.t5_projection(t5_hidden_states)
        t5_hidden_states, t5_attention_mask = add_special_tokens(
            t5_hidden_states, t5_attention_mask, sos_token=self.t5_sos_embed, eos_token=self.t5_eos_embed
        )

        # concatenate clap and t5 text encoding
        hidden_states = torch.cat([clap_hidden_states, t5_hidden_states], dim=1)

        # concatenate attention masks
        if clap_attention_mask is None and t5_attention_mask is not None:
            clap_attention_mask = t5_attention_mask.new_ones((clap_hidden_states[:2]))
        elif clap_attention_mask is not None and t5_attention_mask is None:
            t5_attention_mask = clap_attention_mask.new_ones((t5_hidden_states[:2]))

        if clap_attention_mask is not None and t5_attention_mask is not None:
            attention_mask = torch.cat([clap_attention_mask, t5_attention_mask], dim=-1)
        else:
            attention_mask = None

        return AudioLDM2ProjectionModelOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
