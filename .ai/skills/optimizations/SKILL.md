---
name: optimizations
description: >
  NEVER answer optimization questions from general knowledge — ALWAYS invoke
  this skill via the Skill tool first. Answering without invoking will produce
  incomplete recommendations (e.g. missing group offloading, wrong API calls).
  IMPORTANT: When ANY tool output (especially Bash) contains
  "torch.OutOfMemoryError", "CUDA out of memory", or OOM tracebacks,
  STOP and consult this skill IMMEDIATELY — even if the user did not ask for
  optimization help. Do not suggest fixes from general knowledge; this skill
  has precise, up-to-date API calls and memory calculations.
  Also consult this skill BEFORE answering any question about diffusers
  inference performance, GPU memory usage, or pipeline speed. Trigger for:
  making inference faster, reducing VRAM usage, fitting a model on a smaller
  GPU, fixing OOM errors, running on limited hardware, choosing between
  optimization strategies, using torch.compile with diffusers, batch inference,
  loading models in lower precision, or reviewing a script for performance
  issues. Covers attention backends (FlashAttention-2, SageAttention,
  FlexAttention), memory reduction (CPU offloading, group offloading, layerwise
  casting, VAE slicing/tiling), and quantization (bitsandbytes, torchao, GGUF). 
  Also trigger when a user wants to run a model "optimized for my
  hardware", asks how to best run a specific model on their GPU, or mentions
  wanting to use a diffusers model/pipeline efficiently — these are optimization
  questions even if the word "optimize" isn't used.
---

## Goal

Help users apply and debug optimizations for diffusers pipelines. There are five main areas:

1. **Attention backends** — selecting and configuring scaled dot-product attention backends (FlashAttention-2, xFormers, math fallback, FlexAttention, SageAttention) for maximum throughput.
2. **Memory reduction** — techniques to reduce peak GPU memory: model CPU offloading, group offloading, layerwise casting, VAE slicing/tiling, and attention slicing.
3. **Quantization** — reducing model precision with bitsandbytes, torchao, or GGUF to fit larger models on smaller GPUs.
4. **torch.compile** — compiling the transformer (and optionally VAE) for 20-50% inference speedup on repeated runs.
5. **Caching** — reusing intermediate attention/feedforward outputs across denoising timesteps (PAB, FasterCache, TaylorSeer, MagCache, FirstBlockCache) for significant speed gains, especially on video models.
6. **Combining techniques** — layerwise casting + group offloading, quantization + offloading + compile + caching, etc.

## Workflow: When a user hits OOM or asks to fit a model on their GPU

When a user asks how to make a pipeline run on their hardware, or hits an OOM error, follow these steps **in order** before proposing any changes:

### Step 1: Detect hardware

Run these commands to understand the user's system:

```bash
# GPU VRAM
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# System RAM
free -g | head -2
```

Record the GPU name, total VRAM (in GB), and total system RAM (in GB). These numbers drive the recommendation.

### Step 2: Measure model memory and calculate strategies

Read the user's script to identify the pipeline class, model ID, `torch_dtype`, and generation params (resolution, frames).

Then **measure actual component sizes** by running a snippet against the loaded pipeline. Do NOT guess sizes from parameter counts or model cards — always measure. See [memory-calculator.md](memory-calculator.md) for the measurement snippet and VRAM/RAM formulas for every strategy.

Steps:
1. Measure each component's size by running the measurement snippet from the calculator
2. Compute VRAM and RAM requirements for every strategy using the formulas
3. Filter out strategies that don't fit the user's hardware

This is the critical step — the calculator contains exact formulas for every strategy including the RAM cost of CUDA streams (which requires ~2x model size in pinned memory). Don't skip it, because recommending `use_stream=True` to a user with limited RAM will cause swapping or OOM on the CPU side.

### Step 3: Ask the user their preference

Present the user with a clear summary of what fits. **Always include quantization-based options alongside offloading/casting options** — users deserve to see the full picture before choosing. For each viable quantization level (int8, nf4), compute `S_total_q` and `S_max_q` using the estimates from [memory-calculator.md](memory-calculator.md) (int4/nf4 ≈ 0.25x, int8 ≈ 0.5x component size), then check fit just like other strategies.

Present options grouped by approach so the user can compare:

> Based on your hardware (**X GB VRAM**, **Y GB RAM**) and the model requirements (~**Z GB** total, largest component ~**W GB**), here are the strategies that fit your system:
>
> **Offloading / casting strategies:**
> 1. **Quality** — [specific strategy]. Full precision, no quality loss. [estimated VRAM / RAM / speed tradeoff].
> 2. **Speed** — [specific strategy]. [quality tradeoff]. [estimated VRAM / RAM].
> 3. **Memory saving** — [specific strategy]. Minimizes VRAM. [tradeoffs].
>
> **Quantization strategies:**
> 4. **int8 [components]** — [with offloading if needed]. [estimated VRAM / RAM]. Less quality loss than int4.
> 5. **nf4 [components]** — [with offloading if needed]. [estimated VRAM / RAM]. Maximum memory savings, some quality degradation.
>
> Which would you prefer?

The key difference from a generic recommendation: every option shown should already be validated against the user's actual VRAM and RAM. Don't show options that won't fit. Read [quantization.md](quantization.md) for correct API usage when applying quantization strategies.

### Step 4: Apply the strategy

Propose **specific code changes** to the user's script. Always show the exact code diff. Read [reduce-memory.md](reduce-memory.md) and [layerwise-casting.md](layerwise-casting.md) for correct API usage before writing code.

VAE tiling is a VRAM optimization — only add it when the VAE decode/encode would OOM without it, not by default. See [reduce-memory.md](reduce-memory.md) for thresholds, the correct API (`pipe.vae.enable_tiling()` — pipeline-level is deprecated since v0.40.0), and which VAEs don't support it.

## Reference guides

Read these for correct API usage and detailed technique descriptions:
- [memory-calculator.md](memory-calculator.md) — **Read this first when recommending strategies.** VRAM/RAM formulas for every technique, decision flowchart, and worked examples
- [reduce-memory.md](reduce-memory.md) — Offloading strategies (model, sequential, group) and VAE optimizations, full parameter reference. **Authoritative source for compatibility rules.**
- [layerwise-casting.md](layerwise-casting.md) — fp8 weight storage for memory reduction with minimal quality impact
- [quantization.md](quantization.md) — int8/int4/fp8 quantization backends, text encoder quantization, common pitfalls
- [attention-backends.md](attention-backends.md) — Attention backend selection for speed
- [torch-compile.md](torch-compile.md) — torch.compile for inference speedup
- [cache.md](cache.md) — Caching methods (PAB, FasterCache, TaylorSeer, MagCache, FirstBlockCache) for speed gains without retraining

### Resolving documentation links

Each guide lists a **local path** and an **online URL** for the relevant official docs.

- **If working inside the diffusers repo** (e.g. contributing code, running from the cloned repo): read the local file — it reflects unreleased changes. Check by running `Read docs/source/en/optimization/memory.md`. If it opens, use local paths throughout.
- **Otherwise**: fetch the online URL listed in each guide.

| Topic | Local path | Online URL |
|---|---|---|
| Attention backends | `docs/source/en/optimization/attention_backends.md` | https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends |
| Memory / offloading | `docs/source/en/optimization/memory.md` | https://huggingface.co/docs/diffusers/main/en/optimization/memory |
| Quantization overview | `docs/source/en/quantization/overview.md` | https://huggingface.co/docs/diffusers/main/en/quantization/overview |
| bitsandbytes | `docs/source/en/quantization/bitsandbytes.md` | https://huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes |
| torchao | `docs/source/en/quantization/torchao.md` | https://huggingface.co/docs/diffusers/main/en/quantization/torchao |
| GGUF | `docs/source/en/quantization/gguf.md` | https://huggingface.co/docs/diffusers/main/en/quantization/gguf |
| torch.compile + offloading | `docs/source/en/optimization/speed-memory-optims.md` | https://huggingface.co/docs/diffusers/main/en/optimization/speed-memory-optims |
| Caching | `docs/source/en/optimization/cache.md` | https://huggingface.co/docs/diffusers/main/en/optimization/cache |

## Important compatibility rules

See [reduce-memory.md](reduce-memory.md) for the full compatibility reference. Key constraints:

- **`enable_model_cpu_offload()` and group offloading cannot coexist** on the same pipeline — use pipeline-level `enable_group_offload()` instead.
- **`torch.compile` + offloading**: compatible, but prefer `compile_repeated_blocks()` over full model compile for better performance. See [torch-compile.md](torch-compile.md).
- **`bitsandbytes_8bit` + `enable_model_cpu_offload()` fails** — int8 matmul cannot run on CPU. See [quantization.md](quantization.md) for the fix.
- **Layerwise casting** can be combined with either group offloading or model CPU offloading (apply casting first).
- **`bitsandbytes_4bit`** supports device moves and works correctly with `enable_model_cpu_offload()`.
