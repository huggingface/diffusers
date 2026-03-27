# Profiling Plan: Diffusers Pipeline Profiling with torch.profiler

Education materials to strategically profile pipelines to potentially improve their
runtime with `torch.compile`. To set these pipelines up for success with `torch.compile`,
we often have to get rid of DtoH syncs, CPU overheads, kernel launch delays, and
graph breaks. In this context, profiling serves that purpose for us.

Thanks to Claude Code for paircoding! We acknowledge the [Claude of OSS](https://claude.com/contact-sales/claude-for-oss) support provided to us.

## Table of contents

* [Context](#context)
* [Target pipelines](#target-pipelines)
* [Approach taken](#approach)
* [Verification](#verification)
* [Interpretation](#interpreting-traces-in-perfetto-ui)
* [Taking profiling-guided steps for improvements](#afterwards)

Jump to the "Verification" section to get started right away.

## Context

We want to uncover CPU overhead, CPU-GPU sync points, and other bottlenecks in popular diffusers pipelines — especially issues that become non-trivial under `torch.compile`. The approach is inspired by [flux-fast's run_benchmark.py](https://github.com/huggingface/flux-fast/blob/0a1dcc91658f0df14cd7fce862a5c8842784c6da/run_benchmark.py#L66-L85) which uses `torch.profiler` with method-level annotations, and motivated by issues like [diffusers#11696](https://github.com/huggingface/diffusers/pull/11696) (DtoH sync from scheduler `.item()` call).

## Target Pipelines

| Pipeline | Type | Checkpoint | Steps |
|----------|------|-----------|-------|
| `FluxPipeline` | text-to-image | `black-forest-labs/FLUX.1-dev` | 2 |
| `Flux2KleinPipeline` | text-to-image | `black-forest-labs/FLUX.2-klein-base-9B` | 2 |
| `WanPipeline` | text-to-video | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 2 |
| `LTX2Pipeline` | text-to-video | `Lightricks/LTX-2` | 2 |
| `QwenImagePipeline` | text-to-image | `Qwen/Qwen-Image` | 2 |

> [!NOTE]
> We use realistic inference call hyperparameters that mimic how these pipelines will be actually used. This
> include using classifier-free guidance (where applicable), reasonable dimensions such 1024x1024, etc.
> But we keep the overall running time to a bare minimum (hence 2 `num_inference_steps`). 

## Approach

Follow the flux-fast pattern: **annotate key pipeline methods** with `torch.profiler.record_function` wrappers, then run the pipeline under `torch.profiler.profile` and export a Chrome trace.

### New Files

```
profiling/
  profiling_utils.py       # Annotation helper + profiler setup
  profiling_pipelines.py   # CLI entry point with pipeline configs
```

### Step 1: `profiling_utils.py` — Annotation and Profiler Infrastructure

**A) `annotate(func, name)` helper** (same pattern as flux-fast):

```python
def annotate(func, name):
    """Wrap a function with torch.profiler.record_function for trace annotation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.profiler.record_function(name):
            return func(*args, **kwargs)
    return wrapper
```

**B) `annotate_pipeline(pipe)` function** — applies annotations to key methods on any pipeline:

- `pipe.transformer.forward` → `"transformer_forward"`
- `pipe.vae.decode` → `"vae_decode"` (if present)
- `pipe.vae.encode` → `"vae_encode"` (if present)
- `pipe.scheduler.step` → `"scheduler_step"`
- `pipe.encode_prompt` → `"encode_prompt"` (if present, for full-pipeline profiling)

This is non-invasive — it monkey-patches bound methods without modifying source.

**C) `PipelineProfiler` class:**

- `__init__(pipeline_config, output_dir, mode="eager"|"compile")`
- `setup_pipeline()` → loads from pretrained, optionally compiles transformer, calls `annotate_pipeline()`
- `run()`:
  1. Warm up with 1 unannotated run
  2. Profile 1 run with `torch.profiler.profile`:
     - `activities=[CPU, CUDA]`
     - `record_shapes=True`
     - `profile_memory=True`
     - `with_stack=True`
  3. Export Chrome trace JSON
  4. Print `key_averages()` summary table (sorted by CUDA time) to stdout

### Step 2: `profiling_pipelines.py` — CLI with Pipeline Configs

**Pipeline config registry** — each entry specifies:

- `pipeline_cls`, `pretrained_model_name_or_path`, `torch_dtype`
- `call_kwargs` with pipeline-specific defaults:

| Pipeline | Resolution | Frames | Steps | Extra |
|----------|-----------|--------|-------|-------|
| Flux | 1024x1024 | — | 2 | `guidance_scale=3.5` |
| Flux2Klein | 1024x1024 | — | 2 | `guidance_scale=3.5` |
| Wan | 480x832 | 81 | 2 | — |
| LTX2 | 768x512 | 121 | 2 | `guidance_scale=4.0` |
| QwenImage | 1024x1024 | — | 2 | `true_cfg_scale=4.0` |

All configs use `output_type="latent"` by default (skip VAE decode for cleaner denoising-loop traces).

**CLI flags:**

- `--pipeline flux|flux2|wan|ltx2|qwenimage|all`
- `--mode eager|compile|both`
- `--output_dir profiling_results/`
- `--num_steps N` (override, default 4)
- `--full_decode` (switch output_type from `"latent"` to `"pil"` to include VAE)
- `--compile_mode default|reduce-overhead|max-autotune`
- `--compile_fullgraph` flag

**Output:** `{output_dir}/{pipeline}_{mode}.json` Chrome trace + stdout summary.

### Step 3: Known Sync Issues to Validate

The profiling should surface these known/suspected issues:

1. **Scheduler DtoH sync via `nonzero().item()`** — For Flux, this was fixed by adding `scheduler.set_begin_index(0)` before the denoising loop ([diffusers#11696](https://github.com/huggingface/diffusers/pull/11696)). Profiling should reveal whether similar sync points exist in other pipelines.

2. **`modulate_index` tensor rebuilt every forward in `transformer_qwenimage.py`** (line 901-905) — Python list comprehension + `torch.tensor()` each step. Minor but visible in trace.

3. **Any other `.item()`, `.cpu()`, `.numpy()` calls** in the denoising loop hot path — the profiler's `with_stack=True` will surface these as CPU stalls with Python stack traces.

## Verification

1. Run: `python profiling/profiling_pipelines.py --pipeline flux --mode eager --num_steps 2`
2. Verify `profiling_results/flux_eager.json` is produced
3. Open trace in [Perfetto UI](https://ui.perfetto.dev/) — confirm:
   - `transformer_forward` and `scheduler_step` annotations visible
   - CPU and CUDA timelines present
   - Stack traces visible on CPU events
4. Run with `--mode compile` and compare trace for fewer/fused CUDA kernels

You can also use the `run_profiling.sh` script to bulk launch runs for different pipelines.

## Interpreting Traces in Perfetto UI

Open the exported `.json` trace at [ui.perfetto.dev](https://ui.perfetto.dev/). The trace has two main rows: **CPU** (top) and **CUDA** (bottom). In Perfetto, the CPU row is typically labeled with the process/thread name (e.g., `python (PID)` or `MainThread`) and appears at the top. The CUDA row is labeled `GPU 0` (or similar) and appears below the CPU rows.

**Navigation:** Use `W` to zoom in, `S` to zoom out, and `A`/`D` to pan left/right. You can also scroll to zoom and click-drag to pan. Use `Shift+scroll` to scroll vertically through rows.

### What to look for

**1. Gaps between CUDA kernels**

Zoom into the CUDA row during the denoising loop. Ideally, GPU kernels should be back-to-back with no gaps. Gaps mean the GPU is idle waiting for the CPU to launch the next kernel. Common causes:
- Python overhead between ops (visible as CPU slices in the CPU row during the gap)
- DtoH sync (`.item()`, `.cpu()`) forcing the GPU to drain before the CPU can proceed

**2. CPU stalls (DtoH syncs)**

These appear on the **CPU row** (not the CUDA row) — they are CPU-side blocking calls that wait for the GPU to finish. Look for long slices labeled `cudaStreamSynchronize` or `cudaDeviceSynchronize`. To find them: zoom into the CPU row during a denoising step and look for unusually wide slices, or use Perfetto's search bar (press `/`) and type `cudaStreamSynchronize` to jump directly to matching events. Click on a slice — if `with_stack=True` was enabled, the bottom panel ("Current Selection") shows the Python stack trace pointing to the exact line causing the sync (e.g., a `.item()` call in the scheduler).

**3. Annotated regions**

Our `record_function` annotations (`transformer_forward`, `scheduler_step`, etc.) appear as labeled spans on the CPU row. This lets you quickly:
- Measure how long each phase takes (click a span to see duration)
- See if `scheduler_step` is disproportionately expensive relative to `transformer_forward` (it should be negligible)
- Spot unexpected CPU work between annotated regions

**4. Eager vs compile comparison**

Open both traces side by side (two Perfetto tabs). Key differences to look for:
- **Fewer, wider CUDA kernels** in compile mode (fused ops) vs many small kernels in eager
- **Smaller CPU gaps** between kernels in compile mode (less Python dispatch overhead)
- **CUDA kernel count per step**: to compare, zoom into a single `transformer_forward` span on the CUDA row and count the distinct kernel slices within it. In eager mode you'll typically see many narrow slices (one per op); in compile mode these fuse into fewer, wider slices. A quick way to estimate: select a time range covering one denoising step on the CUDA row — Perfetto shows the number of slices in the selection summary at the bottom. If compile mode shows a similar kernel count to eager, fusion isn't happening effectively (likely due to graph breaks).
- **Graph breaks**: if compile mode still shows many small kernels in a section, that section likely has a graph break — check `TORCH_LOGS="+dynamo"` output for details

**5. Memory timeline**

In Perfetto, look for the memory counter track (if `profile_memory=True`). Spikes during the denoising loop suggest unexpected allocations per step. Steady-state memory during denoising is expected — growing memory is not.

**6. Kernel launch latency**

Each CUDA kernel is launched from the CPU. The CPU-side launch calls (`cudaLaunchKernel`) appear as small slices on the **CPU row** — zoom in closely to a denoising step to see them. The corresponding GPU-side kernel executions appear on the **CUDA row** directly below. You can also use Perfetto's search bar (`/`) and type `cudaLaunchKernel` to find them. The time between the CPU dispatch and the GPU kernel starting should be minimal (single-digit microseconds). If you see consistent delays > 10-20us between launch and execution:
- The launch queue may be starved because of excessive Python work between ops
- There may be implicit syncs forcing serialization
- `torch.compile` should help here by batching launches — compare eager vs compile to confirm

To inspect this: zoom into a single denoising step, select a CUDA kernel on the GPU row, and look at the corresponding CPU-side launch slice directly above it. The horizontal offset between them is the launch latency. In a healthy trace, CPU launch slices should be well ahead of GPU execution (the CPU is "feeding" the GPU faster than it can consume).

### Quick checklist per pipeline

| Question | Where to look | Healthy | Unhealthy |
|----------|--------------|---------|-----------|
| GPU staying busy? | CUDA row gaps | Back-to-back kernels | Frequent gaps > 100us |
| CPU blocking on GPU? | `cudaStreamSynchronize` slices | Rare/absent during denoise | Present every step |
| Scheduler overhead? | `scheduler_step` span duration | < 1% of step time | > 5% of step time |
| Compile effective? | CUDA kernel count per step | Fewer large kernels | Same as eager |
| Kernel launch latency? | CPU launch → GPU kernel offset | < 10us, CPU ahead of GPU | > 20us or CPU trailing GPU |
| Memory stable? | Memory counter track | Flat during denoise loop | Growing per step |

## Afterwards

To keep the profiling iterations fast, we always used [regional compilation](https://pytorch.org/tutorials/recipes/regional_compilation.html). As one would expect the trace with compilation should show
fewer kernel launches than its eager counterpart:

TODO: show traces

_(The traces above were obtained with Flux2.)_

### Spotting gaps between launches

Then a reasonable next step is to spot frequent gaps between kernel executions. In the compiled
case, we don't spot any on the surface. But if we zone in, some become apparent.

TODO: show gaps in a compile trace

So, we provided the profile trace (with compilation) to Claude, asked it to find the instances of
"cudaStreamSynchronize" and "cudaDeviceSynchronize", and to come up with some potential fixes.
Claude came back pretty strong:

```
Issue 1 — Gap between transformer forwards:
- Root cause: tqdm progress bar update() calls between steps add CPU overhead (I/O, time calculations)
- Fix: profiling/profiling_utils.py — added pipe.set_progress_bar_config(disable=True) during profiling setup.
This eliminates the tqdm overhead from the trace. (The remaining gap from scheduler step + Python dispatch is
inherent to eager-mode execution and should shrink significantly under torch.compile.)

Issue 2 — cudaStreamSynchronize during last transformer forward:
- Root cause: _unpack_latents_with_ids() (called right after the denoising loop) computes h = torch.max(h_ids) +
1 and w = torch.max(w_ids) + 1 on GPU tensors, then uses them as shape args for torch.zeros((h * w, ch), ...).
This triggers an implicit .item() DtoH sync, blocking the CPU while the GPU is still finishing the last
transformer forward's kernels.
- Fix: Added height/width parameters to _unpack_latents_with_ids(), pre-computed from the known pixel dimensions
at the call site.
```

It still didn't eliminate the gaps as expected so, we fed that back to Claude and it spotted
something more crucial. TODO: caching context fix.

With the fix applied, the improvements were visible:

TODO: show before and after trace

Before:  

- `_set_context` total: 21.6ms (8 calls)                                                                          
- cache_context total: 21.7ms                                                                                   
- CPU gaps: 5,523us / 8,007us / 5,508us                                                                         
                                                                                                            
After:                                                                                                          
- `_set_context` total: 0.0ms (8 calls)                                                                           
- cache_context total: 0.1ms                                                                                    
- CPU gaps: 158us / 2,777us / 136us  

We also profiled the Wan model and uncovered problems related to CPU DtoH syncs. Below is an
overview.

TODO: provide trace outputs and numbers

> [!NOTE]
> The above-mentioned fixes are available in [this PR](TODO:link).

### Notes

* As mentioned above, we profiled with regional compilation so it's possible that
there are still some gaps outside the compiled regions. A full compilation
will likely mitigate it. In case it doesn't, the above observations could
be useful to mitigate that.