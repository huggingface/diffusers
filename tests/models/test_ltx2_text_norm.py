import torch

from diffusers.models.condition_embedders.condition_embedder_ltx2 import (
    per_layer_masked_mean_norm,
    per_token_rms_norm,
)


def test_per_token_rms_norm_fp16_does_not_overflow():
    torch.manual_seed(0)
    hidden = torch.randn(1, 4, 8, 2)
    hidden[0, 1, 0, 0] = 300.0  # 300**2 does not fit in float16
    reference = per_token_rms_norm(hidden)
    output = per_token_rms_norm(hidden.half())

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)


def test_per_layer_masked_mean_norm_fp16_does_not_overflow():
    # Sum of these values is 131072, which is inf in float16.
    hidden = torch.full((1, 2, 256, 1), 256.0)
    lengths = torch.tensor([2])
    reference = per_layer_masked_mean_norm(hidden, lengths, device="cpu")
    output = per_layer_masked_mean_norm(hidden.half(), lengths, device="cpu")

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)
