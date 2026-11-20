"""Regression coverage for Klein prompt feature ordering across hidden layers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from diffusers import Flux2KleinPipeline


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prompt_hidden_layer_feature_order(batch_size, dtype):
    class Encoder(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.dtype = dtype
            self.hidden_states = tuple(torch.randn(batch_size, 4, 8) for _ in range(4))

        def forward(self, **kwargs):
            return SimpleNamespace(hidden_states=self.hidden_states)

    encoder = Encoder()
    tokenizer = MagicMock(
        return_value={
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
    )
    layers = (3, 1, 2)
    actual = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        encoder,
        tokenizer,
        ["prompt"] * batch_size,
        max_sequence_length=4,
        hidden_states_layers=layers,
    )
    expected = torch.stack([encoder.hidden_states[k] for k in layers], dim=1)
    expected = expected.to(dtype).permute(0, 2, 1, 3).reshape(batch_size, 4, 24)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
