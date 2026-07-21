# Plan: lazy tensor materialization for Diffusers model loading

Port the loading philosophy behind transformers' dynamic weight loading
(https://huggingface.co/docs/transformers/main/en/weightconverter#fast-and-efficient-model-loading)
to Diffusers — **without** porting the full `WeightConverter` API. The core idea being adopted:

> The loader scans the checkpoint *once* to discover what each tensor is for, keeps tensors as
> **lazy safetensors slices** (no bytes read), and schedules their materialization (read + dtype
> cast + device copy) **asynchronously on a small thread pool**. The main thread consumes one
> parameter at a time, sets it on the meta model, and frees it immediately.

## 1. Where Diffusers loses time and memory today

Current flow (`ModelMixin.from_pretrained` → `_load_pretrained_model` → `_load_shard_file` →
`load_state_dict` + `load_model_dict_into_meta`):

| Problem | Where | Effect |
|---|---|---|
| Whole shard is eagerly turned into a state dict before any param is set | `load_state_dict` (`model_loading_utils.py:155`) uses `safetensors.torch.load_file` | For the mmap path this creates all tensor views up front; for `disable_mmap`, DDUF and `.bin` paths the **entire shard is read into RAM** before loading begins |
| Non-sharded checkpoints are loaded eagerly even earlier | `modeling_utils.py:1352` (`state_dict = load_state_dict(...)` inside `from_pretrained`) | A 23 GB Flux transformer is fully "opened" before `_load_pretrained_model` even starts |
| Per-param work is fully serial on the main thread | `load_model_dict_into_meta` (`model_loading_utils.py:213`) | Page-in from mmap, dtype cast (`param.to(dtype)`) and H2D copy happen one param at a time; disk I/O never overlaps with compute or PCIe transfers |
| Parallelism is per *shard file* only | `_load_shard_files_with_threadpool` (`model_loading_utils.py:391`), gated by `HF_ENABLE_PARALLEL_LOADING` | Single-file checkpoints (the common Diffusers case) get zero parallelism; per-shard workers each build a full eager state dict, multiplying peak RAM |
| Shape/mismatch checks require materialized tensors | `_find_mismatched_keys` (`model_loading_utils.py:695`) reads `.shape` off real tensors | With lazy slices, shapes are available from safetensors metadata for free |

The wins from lazy materialization are largest exactly where Diffusers lives: huge single-file
transformers (Flux, Wan, HunyuanVideo, ...), a dtype cast on almost every load
(fp32-serialized → bf16), and `device_map` GPU placement.

## 2. Goals / non-goals

**Goals**

1. Never build a full eager state dict for safetensors checkpoints (mmap'ed or DDUF).
2. Overlap file reads, dtype casts, and H2D copies with a small `ThreadPoolExecutor`
   (transformers found `min(4, cpu_count)` optimal; more threads *hurt*).
3. One flat "key → lazy slice" view across all shards; one scheduling pass over keys.
4. Peak host memory bounded by (in-flight tensors + consumed-but-unset tensors), not by shard size,
   on the memory-constrained paths (sync mode).
5. Keep the public API and all loading features working unchanged: `device_map` (incl. disk
   offload), quantizers (bnb / gguf / torchao / quanto), `keep_in_fp32_modules`,
   `ignore_mismatched_sizes`, `variant`, DDUF, `low_cpu_mem_usage=False` legacy path, flashpack.

**Non-goals (for now)**

- The full `WeightConverter` / `ConversionOps` / reversible-mapping API. We only structure the new
  loop so that API can slot in later (see §7).
- Tensor-parallel sharding-during-load (no `tp_plan` in Diffusers models yet; `enable_parallelism`
  operates post-load).
- Touching pipeline-level loading; it inherits the benefit through each component's
  `from_pretrained`.

## 3. Design

### 3.1 Lazy checkpoint views (new code in `model_loading_utils.py`)

```python
def _load_lazy_state_dict(checkpoint_files, dduf_entries=None, disable_mmap=False):
    """Returns (merged_dict, open_file_handles).

    merged_dict maps checkpoint key -> PySafeSlice (lazy, no bytes read) for safetensors,
    or an eager tensor for formats without lazy support (.bin via torch.load mmap, gguf).
    """
```

- safetensors on disk: `safetensors.safe_open(file, framework="pt")` + `f.get_slice(k)` per key —
  identical to transformers (`modeling_utils.py:4327-4330` there). Handles stay open until loading
  finishes, then are closed.
- safetensors with `disable_mmap=True`: read bytes once, keep today's eager behavior (this flag
  exists precisely to avoid mmap on network filesystems).
- DDUF: entries already expose `as_mmap()`; start with today's eager `safetensors.torch.load`
  per shard and mark as a follow-up (the zip layout makes `safe_open` unavailable; laziness can
  come later via offset-based slicing into the mmap).
- `.bin`: `torch.load(mmap=True)` result is used as-is (tensors are lazily paged by the OS; treat
  them as "already materialized" in the scheduler).
- GGUF: unchanged eager `load_gguf_checkpoint` (the reader decompresses anyway).

Shape metadata for every key is available without reading data (`slice.get_shape()`,
`slice.get_dtype()`), which lets us hoist all bookkeeping ahead of any I/O:

- `_find_mismatched_keys` / `ignore_mismatched_sizes`
- the quantizer shape check for flattened bnb / gguf params
- unexpected-key filtering (skip scheduling those tensors entirely — today they are loaded and
  discarded)

### 3.2 One scheduling pass + async materialization

Replace the per-shard loop (`_load_shard_file` per file) with a single function, modeled on
transformers' `convert_and_load_state_dict_in_model` but much smaller (no converters, no TP):

```python
def _load_state_dict_into_meta_model_async(model, lazy_state_dict, *, dtype, device_map,
                                           keep_in_fp32_modules, hf_quantizer, ...):
    thread_pool = ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) if use_async else None

    jobs = {}  # param_name -> Future | Callable
    for param_name, lazy_tensor in lazy_state_dict.items():   # phase 1: schedule (no I/O)
        # skip unexpected keys / mismatched sizes using slice metadata only
        target_device = _determine_param_device(param_name, device_map)
        target_dtype  = _resolve_target_dtype(param_name, dtype, keep_in_fp32_modules,
                                              hf_quantizer, model)  # today's rules, hoisted
        jobs[param_name] = _spawn_materialize(thread_pool, lazy_tensor, target_device, target_dtype)

    for param_name, job in jobs.items():                      # phase 2: consume, one at a time
        param = job.result() if isinstance(job, Future) else job()
        # disk offload / cpu offload / quantizer create_quantized_param / set_module_tensor_to_device
        ...                                                    # same branch structure as
        del param                                              # load_model_dict_into_meta today
```

with the materialization job doing exactly what transformers' `_materialize_copy` does
(`core_model_loading.py:936`): `tensor = slice[...]` then a single fused
`tensor.to(device=..., dtype=...)` — the read, the cast, and the device copy in one worker task.

**Sync fallback** (jobs stay as callables, executed lazily one by one — bounded memory), matching
transformers' conditions:

- `HF_DEACTIVATE_ASYNC_LOAD=1` (reuse the same env var name for ecosystem consistency)
- disk offload in the `device_map` (memory-constrained by definition)
- on-the-fly quantization (workers must not race ahead of the quantize step on the main thread)

**What moves out of the hot loop.** All the per-param policy in today's
`load_model_dict_into_meta` (`keep_in_fp32_modules` check, float8 quantizer exception, "cast to
old_param.dtype when `dtype=None`", contiguity) is resolved at *schedule* time so the worker does a
single `.to()`. Quantized-param creation (`hf_quantizer.create_quantized_param`) stays on the main
thread at *consume* time, exactly as today — quantizers are not audited for thread safety.

**Ordering.** Sort keys with a natural dot-key sort (transformers' `dot_natural_key`) so
consumption follows layer order — this keeps mmap reads roughly sequential on disk and keeps the
"scheduled but not yet consumed" window small.

### 3.3 Wiring into `modeling_utils.py`

- `from_pretrained`: stop eagerly loading non-sharded checkpoints at `modeling_utils.py:1352`.
  Instead always pass `resolved_model_file` down and build the lazy view inside
  `_load_pretrained_model`. `loaded_keys` for the non-sharded case comes from the lazy view's keys
  (no data read), sharded case keeps using `sharded_metadata["all_checkpoint_keys"]`.
- `_fix_state_dict_keys_on_load` (`modeling_utils.py:2022`) only renames dict keys via
  `dict.pop`/insert — it works on the lazy dict unchanged.
- `hf_quantizer.maybe_update_state_dict` / `maybe_update_loaded_keys`: audit each quantizer.
  Key-only rewrites keep working on the lazy dict; any hook that inspects tensor *values* forces
  sync/eager mode for that quantizer (flag on the quantizer, default eager-compatible).
- `_caching_allocator_warmup` stays as-is and pairs well with async H2D copies.
- After the consume loop: close all safetensors handles, then the existing `empty_device_cache()`;
  add a `torch.cuda.synchronize()` (per used device) before returning when `non_blocking` copies
  were used, so callers never observe in-flight copies.
- `low_cpu_mem_usage=False` legacy path (`_load_state_dict_into_model` with
  `module._load_from_state_dict`): materialize eagerly per shard as today. Not worth optimizing;
  it exists for `ignore_mismatched_sizes`-style edge cases and init-preserving loads.
- `HF_ENABLE_PARALLEL_LOADING` / `_load_shard_files_with_threadpool`: superseded by per-tensor
  scheduling (which parallelizes single-file loads too). Keep the env var as a no-op alias that
  logs a notice, then remove after a deprecation cycle.

### 3.4 What explicitly does not change

- Public signatures of `from_pretrained` / `_load_pretrained_model` outputs
  (`missing/unexpected/mismatched keys`, `offload_index`, `error_msgs`, warning texts).
- `save_pretrained`, flashpack path, `Transformer2DModel` class remapping, meta-device init.
- Numerics: same cast order (checkpoint dtype → target dtype in one `.to()`), same
  `keep_in_fp32_modules` and float8 rules.

## 4. File-by-file change list

| File | Change |
|---|---|
| `src/diffusers/models/model_loading_utils.py` | Add `_load_lazy_state_dict`, `_spawn_materialize`, `_materialize_copy`, `_resolve_target_dtype`, natural-sort key; add `_load_state_dict_into_meta_model_async` (evolves out of `load_model_dict_into_meta`); shape checks move to metadata. Remove `_load_shard_file`/`_load_shard_files_with_threadpool` once the new path is default (and drop the duplicate `_find_mismatched_keys` definition at `:455` — `:695` shadows it today). |
| `src/diffusers/models/modeling_utils.py` | `from_pretrained`: defer non-sharded eager load; `_load_pretrained_model`: call the new loader once over all files instead of per-shard. |
| `src/diffusers/quantizers/*` | Audit `maybe_update_state_dict` / `check_if_quantized_param` / `create_quantized_param` for lazy-dict compatibility; add a `supports_lazy_loading` capability flag (default `False` → eager fallback) and flip it per-backend as validated: bnb and torchao first, gguf stays eager. |
| `tests/models/test_modeling_common.py` | New tests (see §5). |
| `benchmarks/benchmarking_model_loading.py` | New loading benchmark on the existing harness (see §6). |
| `docs/source/en/using-diffusers/loading.md` (or similar) | Document `HF_DEACTIVATE_ASYNC_LOAD` and the deprecation of `HF_ENABLE_PARALLEL_LOADING`. |

## 5. Correctness validation

Must pass before flipping the default:

- Loaded `state_dict()` bit-identical to the old path for: single-file safetensors, sharded,
  `variant="fp16"`, DDUF, `.bin`, gguf, bnb-4bit pre-quantized, torchao, `device_map="auto"` with
  and without disk offload, `keep_in_fp32_modules` models (e.g. Wan with fp32 norms),
  `ignore_mismatched_sizes=True`, deprecated-attention-block checkpoints
  (`_fix_state_dict_keys_on_load`), `dtype=torch.float8_e4m3fn` post-cast path.
- Loading-info parity: identical missing/unexpected/mismatched keys on doctored checkpoints.
- Interrupt safety: pool shutdown with `cancel_futures=True` on exception (no hung threads,
  handles closed).

## 6. Benchmarking plan

This is the selling point of the whole effort, and it is a *system-level* optimization: results
depend on storage bandwidth, page cache state, PCIe generation, CPU contention, and dtype-cast
compute. A single "it's faster on my box" number is not acceptable evidence — the claim must
survive a matrix of environments, and every headline number must come with a script anyone can
re-run. Benchmark code lives in `benchmarks/benchmarking_model_loading.py`, reusing the existing
harness (`benchmarking_utils.py`, `push_results.py`) so results flow into the same dataset as the
inference benchmarks.

### 6.1 Metrics — and why the obvious ones lie

| Metric | How measured | Pitfall it avoids |
|---|---|---|
| End-to-end load wall-clock | `from_pretrained` on a pre-downloaded snapshot (`local_files_only=True`) | Never include download time; hub latency noise would swamp everything |
| Phase breakdown | Timers around: lazy-view build, scheduling pass, consume loop, `dispatch_model` | Attributes wins/regressions to the right phase instead of one opaque number |
| Time-to-first-forward | Load + one tiny forward pass, synchronized | Catches costs hidden by `non_blocking=True` copies that a pure load timer misses |
| Peak host memory | **PSS/USS** (sampled via `psutil`/`/proc/self/smaps_rollup` at ~50 ms) *and* RSS | RSS counts resident mmap'ed file pages, so it overstates "real" usage of the old mmap path and understates the win; report both, explain the difference once |
| Host memory timeline | The same sampler, plotted over time; `memray` for allocation-level drill-down | Peaks are windows, not points — shows the in-flight-futures window stays small (risk §8) |
| Peak VRAM | `torch.cuda.max_memory_allocated()`/`max_memory_reserved()` + NVML `nvmlDeviceGetMemoryInfo` | Allocator-level vs driver-level peaks can diverge with the warmup allocation |
| Disk throughput | `iostat`/`/proc/diskstats` delta during the load | Proves (or disproves) "we now saturate the NVMe"; distinguishes I/O-bound from CPU-bound cells |
| Worker CPU utilization | `psutil.Process.cpu_percent` per thread | Detects GIL-bound casts vs releasing-the-GIL copies |

### 6.2 Environment matrix

System-level means the answer changes with the system; each axis below has flipped the verdict on
similar optimizations before:

- **Page cache**: cold (`sync; echo 3 > /proc/sys/vm/drop_caches` before each run — needs root;
  document the fallback of reading a >RAM-size dummy file to evict) vs warm (immediate re-run).
  Cold is the honest first-load number; warm is the "second load in a notebook" number. Both matter
  and they will differ a lot.
- **Storage**: local NVMe; network filesystem (NFS/Lustre — the common cluster case, where mmap
  page-in latency is brutal and the async pool should shine); the `disable_mmap` path.
- **GPU link** — three concrete machines, chosen so the H2D axis is fully covered:
  1. **RTX 4090** (PCIe Gen4 x16, discrete VRAM) — consumer baseline, copies are expensive;
  2. **DGX Spark (GB10, Grace-Blackwell)** — unified CPU/GPU memory, H2D copies are nearly free,
     so this isolates the pure I/O + cast benefit;
  3. **H100 SXM** — datacenter case, highest H2D bandwidth with discrete HBM.
- **CPU contention**: idle machine vs a synthetic background load (e.g. `stress-ng --cpu N/2`).
  Transformers observed 16 workers being *2× slower* than 4 in contended settings; we must show
  our default degrades gracefully, since real users load models while dataloaders are running.

### 6.3 Workload matrix

Models × dtype × placement, chosen so every code path in §3 has at least one cell:

- **Models**: Flux.1-dev transformer (23 GB, single file — the headline case); SDXL UNet (5 GB,
  many small tensors); Wan 2.2 or HunyuanVideo transformer (sharded + `keep_in_fp32_modules`);
  a **tiny model** (~50 MB VAE) — per-tensor scheduling adds fixed overhead and must not regress
  small loads by more than noise; **full pipeline** `FluxPipeline.from_pretrained` — components
  load sequentially, so pipeline-level wall-clock is what end users actually feel.
- **dtype**: no-cast (bf16 checkpoint → bf16, pure I/O); cast (fp32 → bf16, the common case,
  exercises worker-side compute); fp8 post-cast path.
- **Placement**: CPU; single CUDA device; `device_map="auto"` across 2 GPUs; disk offload
  (sync fallback — must be ≈ old path, this is a no-regression cell); bnb-4bit pre-quantized;
  on-the-fly bnb quantization (sync fallback cell).

### 6.4 Baselines and sweeps

- Baselines: current `main`; `main` + `HF_ENABLE_PARALLEL_LOADING=1` (the sharded cells — the new
  path must beat the old parallel mode, not just the serial default); transformers ≥ 5.x loading a
  comparable-size LLM on the same box as an external sanity reference.
- Sweeps on the headline cell: worker count 1/2/4/8/16 (validate that transformers' 4-worker
  default holds for our tensor-size distribution — diffusion transformers have far fewer, larger
  tensors than LLMs, so the optimum may differ); `HF_DEACTIVATE_ASYNC_LOAD=1` (sync-lazy) to
  isolate "lazy" wins from "async" wins.

### 6.5 Methodology rules

- ≥ 5 runs per cell; report **median and IQR**, not mean — I/O timings have heavy tails.
- One process per measurement (fresh interpreter), fixed CPU governor (`performance`), no
  frequency-scaling surprises; record kernel, torch, safetensors versions and storage model in the
  results metadata.
- Snapshot pre-downloaded; `local_files_only=True`; verify with a hub-offline env var that no
  network call sneaks into the timed region.
- Memory sampler runs in a subprocess (not a thread) so it cannot be starved by the GIL during
  worker-heavy phases.

### 6.6 Evidence beyond numbers

For the PR description and docs, two artifacts that make the mechanism visible:

- A **py-spy/pyinstrument flamegraph** before/after on the headline cell: the "after" graph should
  show the main thread blocked in `Future.result()` while workers sit in `slice[...]` and `.to()`.
- A **torch profiler trace** showing H2D copy streams overlapping file reads — the picture that
  explains *why* it is faster, so reviewers aren't taking a table on faith.
- The host-memory timeline plot for old vs new on the `disable_mmap`/DDUF cell, showing the
  shark-fin (full state dict) collapsing into a sawtooth (bounded in-flight window).

### 6.7 Success criteria and guardrails

- Headline (Flux bf16 → single CUDA, cold cache, NVMe): target ≥ 1.5× wall-clock; report honestly
  if lower — even 1.2× with a large RAM reduction on eager paths justifies landing the PR.
- **No cell regresses > 5%** (median): notably tiny models, disk offload, on-the-fly quant, warm
  CPU mmap loads. These no-regression cells are as important as the headline.
- Add a loading benchmark to the periodic benchmark workflow via `run_all.py`/`push_results.py`
  so future changes to the loader show up in the tracked dataset; no hard CI gate (shared runners
  make load timings too noisy for pass/fail), but the trend line is reviewed like the inference
  numbers.

Expected shape of results, from transformers' experience: biggest speedups where dtype-cast and
GPU placement combine; near-neutral for warm-cache CPU mmap loads (laziness mostly moves *when*
pages fault in); large RAM wins limited to the paths that were eager before (`disable_mmap`,
DDUF, per-shard parallel mode).

### 6.8 Runbook

The harness is `benchmarks/benchmarking_model_loading.py` (already landed alongside this plan).
It spawns one fresh subprocess per measurement, samples PSS/USS/RSS from a sidecar *process*,
evicts checkpoint pages with `posix_fadvise(DONTNEED)` for cold runs (no root needed), and
verifies coldness by reporting actual bytes read from `/proc/diskstats`. Per machine:

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev --include "transformer/*"  # once
cd benchmarks && python benchmarking_model_loading.py --label <machine>          # full matrix
```

Default matrix: Flux.1-dev transformer (23.8 GB, 3 shards, bf16) × {cuda-bf16, cuda-fp32-cast,
cpu-bf16} × {cold, warm} plus the `HF_ENABLE_PARALLEL_LOADING=1` baseline cell; 5 runs each;
results land in `loading_benchmark_<label>.json` (raw runs + median/IQR summary + machine
metadata). Order of execution: benchmarks run **once the feature branch is complete** — on each
machine, run the matrix twice back-to-back (`main` checkout, then the feature branch) so both
sides see identical hardware, kernel, and storage state; other models from §6.3 are added after
the Flux matrix is green on all machines.

## 7. Future work this unlocks (out of scope here)

The scheduling loop is deliberately shaped as *match key → resolve placement → collect job*, i.e.
the degenerate "every key maps to itself" case of transformers' mapping walk. The main
`from_pretrained` path gets the full lazy-slice benefit from this plan as-is; what does *not* are
the conversion paths that bypass it — most notably `from_single_file`, which still materializes
the whole original checkpoint eagerly and converts it dict-to-dict before loading. A later
`WeightConverter`-style API would slot in at the "match" step, at which point those conversion
paths inherit laziness too: rename-only mappings stay zero-cost, and tensor-op conversions
materialize only the tensors each op consumes. The natural consumers already in the codebase:

- `_fix_state_dict_keys_on_load`'s deprecated attention-block renames → `WeightRenaming` entries.
- The 36 `convert_*` functions in `loaders/single_file_utils.py` (`from_single_file`), which today
  materialize the entire original checkpoint and rebuild a full converted dict in RAM — the
  single biggest long-term beneficiary of per-tensor lazy conversion.
- Pre-quantized deserialization (gguf/torchao) expressed as conversion ops instead of quantizer
  special cases.

## 8. Risks and mitigations

- **Futures outpacing consumption → RAM spike.** All jobs are submitted up front; if the pool runs
  far ahead of the main thread the theoretical peak approaches full model size — i.e. never worse
  than today's eager dict, and the 4-worker pool + layer-ordered consumption keeps the practical
  window at a few tensors. If benchmarks show spikes, cap in-flight results with a bounded
  semaphore released at consume time.
- **Thread-unsafe tensor sources.** A `PySafeSlice` read (`slice[...]`) is a pure read on an
  mmap'ed region and is safe across threads; `.bin`/gguf paths never enter the pool.
- **Quantizer hooks that mutate tensors at state-dict level.** Capability flag + eager fallback
  (§3.3) keeps every backend on its current, tested path until explicitly migrated.
- **File-handle lifetime.** Handles owned by the loader, closed in a `finally`; slices never
  escape `_load_pretrained_model`.
- **Windows/mmap quirks and network filesystems.** `disable_mmap=True` keeps today's fully eager
  behavior as the escape hatch; `HF_DEACTIVATE_ASYNC_LOAD=1` disables threading independently.

## 9. Rollout

Everything ships as **one PR**, but built in three stages that stay separate commits so the diff
reviews incrementally and any stage can be bisected later:

1. **Stage 1 — lazy views, sync loop.** `_load_lazy_state_dict` + single scheduling pass, thread
   pool disabled (callables only). Pure refactor: byte-identical results, all formats. The parity
   test suite lands in this stage.
2. **Stage 2 — async pool.** Enable the 4-worker pool with the sync-fallback conditions; add
   `HF_DEACTIVATE_ASYNC_LOAD`.
3. **Stage 3 — cleanup.** Deprecate `HF_ENABLE_PARALLEL_LOADING`, delete the per-shard loaders,
   migrate quantizer capability flags (bnb, torchao) to lazy mode.

Benchmarks run **after the feature branch is complete** (see §6.8): the full §6 matrix on all
three machines, feature branch vs `main`, with the flamegraph/trace/memory-timeline artifacts
attached to the PR description.
