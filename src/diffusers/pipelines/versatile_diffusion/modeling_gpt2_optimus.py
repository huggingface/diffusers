import torch
from torch import nn

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2PreTrainedModel


class GPT2OptimusModel(GPT2PreTrainedModel):
    def __init__(self, config, latent_as_gpt_emb, latent_as_gpt_memory, latent_size):
        super().__init__(config)
        self.latent_as_gpt_emb = latent_as_gpt_emb
        self.latent_as_gpt_memory = latent_as_gpt_memory
        self.latent_size = latent_size
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.linear = nn.Linear(
            self.latent_size, config.hidden_size * config.n_layer, bias=False
        )  # different latent vector for each layer
        self.linear_emb = nn.Linear(
            self.latent_size, config.hidden_size, bias=False
        )  # share the same latent vector as the embeddings

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            if self.latent_as_gpt_emb:
                past_emb = self.linear_emb(past)  # used as embeddings to add on other three embeddings

            if self.latent_as_gpt_memory:
                past = self.linear(past)
                share_latent = False
                if share_latent:
                    # the same latent vector shared by all layers
                    past = [past.unsqueeze(-2), past.unsqueeze(-2)]  # query, key
                    past = [past] * len(self.h)
                    past_length = past[0][0].size(-2)
                else:
                    # different latent vectors for each layer
                    past_split = torch.split(past.unsqueeze(1), self.config.hidden_size, dim=2)
                    past = list(zip(past_split, past_split))
                    past_length = 1  # past[0][0].size(-2)
            else:
                past_length = 0
                past = [None] * len(self.h)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        if self.latent_as_gpt_emb:
            hidden_states = hidden_states + past_emb.unsqueeze(1)

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
            )

            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        return outputs  # last hidden state, presents, (all hidden_states), (attentions)


class GPT2OptimusForLatentConnector(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.latent_as_gpt_emb = True
        self.latent_as_gpt_memory = True
        self.latent_size = getattr(config, "latent_size", 32)
        self.transformer = GPT2OptimusModel(
            config,
            latent_as_gpt_emb=self.latent_as_gpt_emb,
            latent_as_gpt_memory=self.latent_as_gpt_memory,
            latent_size=self.latent_size,
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.tie_weights()

        # Initialize weights and apply final processing
        self.post_init()
        self.tie_weights()

    def _tie_or_clone_weights(self, first_module, second_module):
        """Tie or clone module weights depending of weither we are using TorchScript or not"""
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

        if hasattr(first_module, "bias") and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                "constant",
                0,
            )

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        return {
            "input_ids": input_ids,
            "past_key_values": past,
        }
