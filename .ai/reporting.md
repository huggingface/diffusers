# Reporting bugs and performance claims

For issues and PR descriptions on this repo, from humans and agents alike.

## Bug reports

A reproduction is **what you were doing when it broke, with everything unnecessary removed** — start from the real failing situation and delete, don't build a clean synthetic case from scratch. The test: can someone paste it into a terminal and see the same failure, without first accepting your theory of the cause? A script that demonstrates your theory is not a reproduction — it can pass while the real bug is still there.

- Keep the real model, settings, and dtype. A repro that downloads weights and takes two minutes is worth more than a fast synthetic one.
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

The difference: a maintainer understands the first script at a glance — it's an ordinary pipeline call — while the second is not, and it only demonstrates the reporter's theory: it can go green while the real pipeline still dies on a 24GB card. 

For structure, follow the [bug report template](../.github/ISSUE_TEMPLATE/bug-report.yml).

## Performance claims

Report **the end-to-end number only**: wall clock for the full pipeline call, before vs. after, on the same hardware, dtype, and seed. That's the number we decide with — if we want op-level detail, we'll ask.

Attach the script you measured with, please keep it as simple as possible (see `benchmarks/benchmarking_utils.py`):

```python
import torch.utils.benchmark as benchmark

def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}, num_threads=1)
    return f"{t0.blocked_autorange().mean:.3f}s"

pipe(**call_kwargs)  # warmup
print(benchmark_fn(pipe, **call_kwargs))
print(f"peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
```

State the setup (GPU, dtype, torch/diffusers versions, attention backend) and the exact call (model id, resolution/duration, steps, batch size). One-shot timings are noise.
