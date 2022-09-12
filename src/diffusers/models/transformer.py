import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end. It is possible to use
    torch.nn.MultiheadAttention here but I am including an explicit implementation here to show that there is nothing
    too scary here.
    """

    def __init__(self, hidden_size, num_heads, causal=False, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.n_head = num_heads
        self.causal = causal
        # TODO define this
        # if self.causal:
        #     block_size = np.prod(H.latent_shape)
        #     mask = torch.tril(torch.ones(block_size, block_size))
        #     self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if self.causal and layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal and layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, hidden_size, num_heads, causal=False, attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size, num_heads, causal=causal, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),  # nice
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        attn, present = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present
        return x


class Transformer(ModelMixin, ConfigMixin):
    """the full GPT language model, with a context size of block_size"""

    @register_to_config
    def __init__(
        self,
        vocab_size=1024,
        hidden_size=512,
        num_layers=24,
        max_position_embeddings=256,
        num_heads=8,
        causal=False,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    ):
        super().__init__()

        # we add one for the mask token
        self.vocab_size = vocab_size + 1
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.causal = causal
        if self.causal:
            self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_position_embeddings, self.hidden_size))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    causal=causal,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    embd_pdrop=embd_pdrop,
                )
                for _ in range(self.num_layers)
            ]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.vocab_size - 1, bias=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, t=None):
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)

        if self.causal:
            token_embeddings = torch.cat(
                (self.start_tok.repeat(token_embeddings.size(0), 1, 1), token_embeddings), dim=1
            )

        t = token_embeddings.shape[1]
        assert t <= self.max_position_embeddings, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
