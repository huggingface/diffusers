# Reporting bugs and performance claims

Use this guide when writing bug reports or performance claims in issues and pull requests.

## Bug reports

A reproduction is the smallest version of the real workflow that still fails. Start with what you were doing when the failure occurred, then remove anything unrelated. Someone should be able to run it and see the same failure without first accepting your theory about its cause. A script that tests your theory is not a reproduction. It may pass even when the real workflow still fails.

- Keep the real model, settings, and dtype. A slower repro is better than a faster one that changes the behavior.
- If you truly can't produce one (gated model, 8 GPUs), say so at the top and give the closest thing you have. Don't silently substitute something smaller.

✓ A reproduction — the failing call itself, trimmed (from [#14518](https://github.com/huggingface/diffusers/issues/14518): Krea-2 OOMs a 16GB GPU under 4-bit quantization):

```python
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained(...)
pipe.to("cuda")

image = pipe(...).images[0]

print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")  # 19.94 GiB — OOM on 16GB
```

✗ Not a reproduction — a synthetic benchmark built to demonstrate the suspected cause (paraphrased from an earlier draft of the same report):

```python
# "enable_gqa + attn_mask falls back to the math backend and materializes an S x S score matrix"
q = torch.randn(1, 24, 8100, 128, device="cuda", dtype=torch.bfloat16)
k = v = torch.randn(1, 6, 8100, 128, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, 1, 8100, 8100, dtype=torch.bool, device="cuda")
F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
print(torch.cuda.max_memory_allocated())
```

The first script exposes the real pipeline call. The second only tests the proposed mechanism, so it may pass even when the pipeline still fails. Keep the hypothesis separate from the reproduction.

For structure, follow the [bug report template](../.github/ISSUE_TEMPLATE/bug-report.yml).

## Performance claims

Lead with the end-to-end result: measure the full pipeline call before and after the change on the same hardware, dtype, and seed. Only include lower-level measurements when requested.

Attach the measurement script and keep it as simple as possible. See `benchmarks/benchmarking_utils.py` for the repository's benchmarking helper.

```python
import torch.utils.benchmark as benchmark

def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}, num_threads=1)
    return f"{t0.blocked_autorange().mean:.3f}s"

pipe(**call_kwargs)  # warmup
print(benchmark_fn(pipe, **call_kwargs))
print(f"peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
```

State the setup (GPU, dtype, torch/diffusers versions, attention backend) and the exact call (model id, resolution/duration, steps, batch size). One-shot timings are noisy.
