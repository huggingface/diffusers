import torch

from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _HUB_KERNELS_REGISTRY,
    _flash_attention_3_varlen_hub,
)


def test_flash_attention_3_varlen_hub_handles_tensor_return(monkeypatch):
    def flash_attention_3_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    ):
        return q + 1000

    monkeypatch.setattr(
        _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_VARLEN_HUB],
        "kernel_fn",
        flash_attention_3_varlen_func,
    )

    batch_size = 2
    seq_len = 4
    heads = 2
    dim = 5
    query = torch.arange(batch_size * seq_len * heads * dim, dtype=torch.float32).reshape(
        batch_size, seq_len, heads, dim
    )
    key = query.clone()
    value = query.clone()

    out = _flash_attention_3_varlen_hub(query, key, value)

    assert out.shape == query.shape
    assert torch.equal(out, query + 1000)
