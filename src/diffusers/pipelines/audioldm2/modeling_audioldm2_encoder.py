from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    ClapTextConfig,
    ClapTextModelWithProjection,
    GPT2Config,
    GPT2Model,
    GPT2PreTrainedModel,
    T5Config,
    T5EncoderModel,
)


def add_special_tokens(hidden_states, attention_mask, sos_token, eos_token):
    batch_size = hidden_states.shape[0]

    if attention_mask is not None:
        # Add two more steps to attn mask
        new_attn_mask_step = attention_mask.new_ones((batch_size, 1))
        attention_mask = torch.cat([new_attn_mask_step, attention_mask, new_attn_mask_step], dim=-1)

    # Add the SOS / EOS tokens at the start / end of the sequence respectively
    sos_token = sos_token.expand(batch_size, 1, -1)
    eos_token = eos_token.expand(batch_size, 1, -1)
    hidden_states = torch.cat([sos_token, hidden_states, eos_token], dim=1)
    return hidden_states, attention_mask


class AudioLDM2TextEncoder(GPT2PreTrainedModel):
    def __init__(
        self, clap_config: ClapTextConfig, t5_config: T5Config, gpt2_config: GPT2Config, max_new_tokens: int = 8
    ):
        super().__init__(gpt2_config)
        # base models
        self.clap_encoder = ClapTextModelWithProjection(clap_config)
        self.t5_encoder = T5EncoderModel(t5_config)
        self.model = GPT2Model(gpt2_config)

        # additional projection layers
        hidden_size = gpt2_config.n_embd
        self.clap_projection = nn.Linear(clap_config.projection_dim, hidden_size)
        self.t5_projection = nn.Linear(t5_config.d_model, hidden_size)

        # learnable SOS / EOS token embeddings
        self.clap_sos_embed = nn.Parameter(torch.ones(hidden_size))
        self.clap_eos_embed = nn.Parameter(torch.ones(hidden_size))

        self.t5_sos_embed = nn.Parameter(torch.ones(hidden_size))
        self.t5_eos_embed = nn.Parameter(torch.ones(hidden_size))

        self.max_new_tokens = max_new_tokens

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        clap_input_ids: Optional[torch.LongTensor] = None,
        t5_input_ids: Optional[torch.LongTensor] = None,
        clap_attention_mask: Optional[torch.LongTensor] = None,
        t5_attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None:
            if clap_input_ids is None or t5_input_ids is None:
                raise ValueError(
                    "Both `clap_input_ids` and `t5_input_ids` have to be defined when `inputs_embeds` is None."
                )
            clap_hidden_states = self.clap_encoder(clap_input_ids, attention_mask=clap_attention_mask).text_embeds
            # additional L_2 normalization over each hidden-state
            clap_hidden_states = nn.functional.normalize(clap_hidden_states, dim=-1)
            clap_hidden_states = self.clap_projection(clap_hidden_states)

            clap_hidden_states, clap_attention_mask = add_special_tokens(
                clap_hidden_states, clap_attention_mask, sos_token=self.clap_sos_embed, eos_token=self.clap_eos_embed
            )

            t5_hidden_states = self.t5_encoder(t5_input_ids, attention_mask=t5_attention_mask).text_embeds
            t5_hidden_states = self.t5_projection(t5_hidden_states).last_hidden_states

            t5_hidden_states, t5_attention_mask = self.add_special_tokens(
                t5_hidden_states, t5_attention_mask, sos_token=self.t5_sos_embed, eos_token=self.t5_eos_embed
            )

            # concatenate clap and t5 text encoding
            inputs_embeds = torch.cat([clap_hidden_states, t5_hidden_states], dim=-1)

            # concatenate attention masks
            if clap_attention_mask is None and t5_attention_mask is not None:
                clap_attention_mask = t5_attention_mask.new_ones((clap_hidden_states[:2]))
            elif clap_attention_mask is not None and t5_attention_mask is None:
                t5_attention_mask = clap_attention_mask.new_ones((t5_hidden_states[:2]))

            if clap_attention_mask is not None and t5_attention_mask is not None:
                attention_mask = torch.cat([clap_attention_mask, t5_attention_mask], dim=-1)

        transformer_outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        return transformer_outputs

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **model_kwargs,
    ):
        """

        Generates a sequence of hidden-states for the GPT2 model.

        Parameters:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`, *optional*):
                Number of new tokens to generate. If un-specified, defaults to the model attribute
                `max_new_tokens`.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the
                `forward` function of the model.

        Return:
            `inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        steps = 3 * max_new_tokens // 4

        for _ in range(steps):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(inputs_embeds, **model_kwargs)

            # forward pass to get next hidden states
            output = self.model(**model_inputs, return_dict=True)

            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = self._update_model_kwargs_for_generation(output, model_kwargs)

        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        inputs_embeds,
        clap_input_ids=None,
        t5_input_ids=None,
        clap_attention_mask=None,
        t5_attention_mask=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            # no need for clap / t5 input ids since we already have our inputs embeds for generation
            clap_input_ids = None
            t5_input_ids = None

        if past_key_values is not None:
            # only last token for inputs_embeds if past is defined in kwargs
            inputs_embeds = inputs_embeds[:, -1:]

        return {
            "clap_input_ids": clap_input_ids,
            "t5_input_ids": t5_input_ids,
            "clap_attention_mask": clap_attention_mask,
            "t5_attention_mask": t5_attention_mask,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }
