import pytest
import torch

from diffusers.models.attention_dispatch import _AttentionBackendRegistry, dispatch_attention_fn

from ..testing_utils import require_torch_gpu, torch_device


VARLEN_BACKENDS = []
for name, fn in _AttentionBackendRegistry._backends.items():
    # Handle both enum and string keys
    key = name.value if hasattr(name, "value") else name
    if key in {"flash_varlen", "flash_varlen_3", "sage_varlen"} and callable(fn):
        VARLEN_BACKENDS.append(name)

if not VARLEN_BACKENDS:
    pytest.skip("No varlen backends available", allow_module_level=True)


@require_torch_gpu
@pytest.mark.parametrize("backend", VARLEN_BACKENDS)
def test_varlen_backend_accepts_seq_lens(backend):
    if torch_device != "cuda":
        pytest.skip("CUDA required for varlen backends")
    backend_fn = _AttentionBackendRegistry._backends.get(backend)
    if not callable(backend_fn):
        pytest.skip(f"{backend} backend not available")

    key = backend.value if hasattr(backend, "value") else backend
    dtype = torch.float16
    seq_lens = torch.tensor([3, 5], dtype=torch.int32, device=torch_device)

    q = torch.randn(2, 5, 2, 8, device=torch_device, dtype=dtype)
    k = torch.randn(2, 5, 2, 8, device=torch_device, dtype=dtype)
    v = torch.randn(2, 5, 2, 8, device=torch_device, dtype=dtype)

    out = dispatch_attention_fn(q, k, v, backend=key, seq_lens=seq_lens)
    assert out.shape == q.shape
