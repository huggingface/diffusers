import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end. It is possible to use
    torch.nn.MultiheadAttention here but I am including an explicit implementation here to show that there is nothing
    too scary here.
    """

    def __init__(self, hidden_size, num_heads, attention_dropout=0.0, residual_dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # regularization
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)
        # output projection
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.n_head = num_heads

    def forward(self, hidden_states, layer_past=None):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        keys = (
            self.key(hidden_states)
            .view(batch_size, seq_length, self.n_head, hidden_size // self.n_head)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, hs)
        queries = (
            self.query(hidden_states)
            .view(batch_size, seq_length, self.n_head, hidden_size // self.n_head)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, hs)
        values = (
            self.value(hidden_states)
            .view(batch_size, seq_length, self.n_head, hidden_size // self.n_head)
            .transpose(1, 2)
        )  # (batch_size, num_heads, seq_length, hs)

        present = torch.stack((keys, values))

        # self-attention;
        # (batch_size, num_heads, seq_length, hs) x (batch_size, num_heads, hs, seq_length) -> (batch_size, num_heads, seq_length, seq_length)
        attention_logits = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))

        attention_scores = F.softmax(attention_logits, dim=-1)
        attention_scores = self.attention_dropout(attention_scores)
        hidden_states = (
            attention_scores @ values
        )  # (batch_size, num_heads, seq_length, seq_length) x (batch_size, num_heads, seq_length, hs) -> (batch_size, num_heads, seq_length, hs)
        # re-assemble all head outputs side by side
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        # output projection
        hidden_states = self.residual_dropout(self.projection(hidden_states))
        return hidden_states, present


class Block(nn.Module):
    """An unassuming Transformer block"""

    def __init__(self, hidden_size, num_heads, attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(
            hidden_size,
            num_heads,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(residual_dropout),
        )

    def forward(self, hidden_states, layer_past=None, return_present=False):
        attention_output, present = self.attention(self.layernorm1(hidden_states), layer_past)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.layernorm2(hidden_states))

        if layer_past is not None or return_present:
            return hidden_states, present
        return hidden_states


class Transformer(ModelMixin, ConfigMixin):
    """The full BERT/GPT language model, with a context size of `max_position_embeddings`"""

    @register_to_config
    def __init__(
        self,
        vocab_size=1024,
        hidden_size=512,
        num_layers=24,
        max_position_embeddings=256,
        num_heads=8,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        residual_dropout=0.0,
    ):
        super().__init__()

        # we add one for the mask token
        self.vocab_size = vocab_size + 1
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers

        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.max_position_embeddings, self.hidden_size))
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    embedding_dropout=embedding_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        # decoder head
        self.final_layernorm = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.vocab_size - 1, bias=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids):
        # 1. get token embeddings
        token_embeddings = self.token_embeddings(input_ids)

        # 2. add position embeddings
        seq_length = token_embeddings.shape[1]
        if seq_length > self.max_position_embeddings:
            raise ValueError("Cannot forward, model sequence length is exhausted.")

        position_embeddings = self.position_embeddings[:, :seq_length, :]
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # 3. forward through blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # 4. decoder head
        hidden_states = self.final_layernorm(hidden_states)
        logits = self.head(hidden_states)

        return logits
