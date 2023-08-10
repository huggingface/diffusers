from transformers import GPT2Model, GPT2PreTrainedModel, GPT2Config
import torch.nn as nn
import torch

from typing import List


class Sequence2AudioMAE(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config, sequence_input_embed_dim: List[int]):
        super().__init__(config)
        self.transformer = GPT2Model(config)

        self.sos_token_embeds = nn.Embedding(2, config.n_embd)
        self.eos_token_embeds = nn.Embedding(2, config.n_embd)

        for dim in sequence_input_embed_dim:
            self.input_sequence_embed_linear.append(nn.Linear(dim, config.n_embd))

        # Initialize weights and apply final processing
        self.post_init()

    def add_sos_bos_tokens(self, hidden_states, attention_mask, id):
        batch_size = hidden_states.shape[0]

        new_attn_mask_step = torch.ones((batch_size, 1)).to(hidden_states.device)
        key_id = torch.tensor([id]).to(hidden_states.device)

        # Add two more steps to attn mask
        new_attn_mask = torch.cat(
            [new_attn_mask_step, attention_mask, new_attn_mask_step], dim=1
        )

        # Project hidden-states using linear layer
        hidden_states = self.input_sequence_embed_linear[id](hidden_states)

        # Add the SOS / EOS tokens at the start / end of the sequence respectively
        sos_token = self.start_of_sequence_tokens(key_id).expand(batch_size, 1, -1)
        eos_token = self.end_of_sequence_tokens(key_id).expand(batch_size, 1, -1)
        hidden_states = torch.cat([sos_token, hidden_states, eos_token], dim=1)
        return hidden_states, new_attn_mask

    @torch.no_grad()
    def generate(self, inputs_embeds, **model_kwargs):
        """

        Generates a sequence of hidden-states for the GPT2 model.

        Parameters:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the
                `forward` function of the model.

        Return:
            `inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        steps = self.mae_token_num

        for _ in range(3 * steps // 4):
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

    def prepare_inputs_for_generation(self, inputs_embeds, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            # only last token for inputs_embeds if past is defined in kwargs
            inputs_embeds = inputs_embeds[:, -1:]

        return {
            "input_ids": None,  # always use inputs_embeds
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }