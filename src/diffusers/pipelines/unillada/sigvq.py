"""SigVQ: Semantic token embedding extractor for the image decoder."""

import torch
import torch.nn as nn


class _LinearWrapper(nn.Module):
    """Wraps nn.Linear inside a .proj attribute to match diffusers checkpoint key format."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.proj(x)


class _FeedForward(nn.Module):
    """SiLU feed-forward matching diffusers key layout: net.0.proj / net.1 / net.2"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            _LinearWrapper(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SigVQ(nn.Module):
    """
    Lightweight semantic token extractor.
    Maps discrete VQ token IDs to continuous feature vectors via embedding + projection.

    Args:
        vocab_size: VQ codebook size (default: 16384).
        inner_dim: Feature dimension (default: 4096).
    """

    def __init__(self, vocab_size: int = 16384, inner_dim: int = 4096):
        super().__init__()
        self.prior_token_embedding = nn.Embedding(vocab_size, inner_dim)
        self.prior_projector = _FeedForward(dim=inner_dim, hidden_dim=inner_dim)
        self.requires_grad_(False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) discrete token indices.
        Returns:
            (batch, seq_len, inner_dim) projected feature vectors.
        """
        return self.prior_projector(self.prior_token_embedding(token_ids))
