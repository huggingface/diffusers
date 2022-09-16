from dataclasses import dataclass

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .attention import Block


@dataclass
class TransformerOutput(BaseOutput):
    """
    Class for Transformer outputs.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    logits: torch.FloatTensor = None


class Transformer(ModelMixin, ConfigMixin):
    """The full BERT-like language model, with a context size of `max_position_embeddings`"""

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

    def forward(self, input_ids: torch.LongTensor, return_dict: bool = True):
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

        if not return_dict:
            return (logits,)

        return TransformerOutput(logits=logits)
