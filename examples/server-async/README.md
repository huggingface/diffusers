# Asynchronous server and parallel execution of models

> Example/demo server that keeps a single model in memory while safely running parallel inference requests by creating per-request lightweight views and cloning only small, stateful components (schedulers, RNG state, small mutable attrs). Works with StableDiffusion3/Flux pipelines and a custom `diffusers` fork.

## ⚠️ IMPORTANT

* This example uses a custom Diffusers fork: `https://github.com/F4k3r22/diffusers-async`.
* The server and inference harness live in this repo: `https://github.com/F4k3r22/DiffusersServer`.
  The example demonstrates how to run pipelines like `StableDiffusion3-3.5` and `Flux.1` concurrently while keeping a single copy of the heavy model parameters on GPU.

## Necessary components

All the components needed to create the inference server are in `DiffusersServer/`

```
DiffusersServer/                 # the example server package
├── __init__.py                   
├── create_server.py             # helper script to build/run the app programmatically
├── Pipelines.py                 # pipeline loader classes (SD3, Flux, legacy SD, video)
├── serverasync.py               # FastAPI app factory (create_app_fastapi)
├── superpipeline.py             # optional custom pipeline glue code
├── uvicorn_diffu.py             # convenience script to start uvicorn with recommended flags
```


## What `diffusers-async` adds / Why we needed it

Core problem: a naive server that calls `pipe.__call__` concurrently can hit **race conditions** (e.g., `scheduler.set_timesteps` mutates shared state) or explode memory by deep-copying the whole pipeline per-request.

`diffusers-async` / this example addresses that by:

* **Request-scoped views**: `RequestScopedPipeline` creates a shallow copy of the pipeline per request so heavy weights (UNet, VAE, text encoder) remain shared and *are not duplicated*.
* **Per-request mutable state**: stateful small objects (scheduler, RNG state, small lists/dicts, callbacks) are cloned per request. Where available we call `scheduler.clone_for_request(...)`, otherwise we fallback to safe `deepcopy` or other heuristics.
* **`retrieve_timesteps(..., return_scheduler=True)`**: retro-compatible helper that returns `(timesteps, num_inference_steps, scheduler)` without mutating the shared scheduler. This is the safe path for getting a scheduler configured per-request.
* **Robust attribute handling**: wrapper avoids writing to read-only properties (e.g., `components`) and auto-detects small mutable attributes to clone while avoiding duplication of large tensors.

## How the server works (high-level flow)

1. **Single model instance** is loaded into memory (GPU/MPS) when the server starts.
2. On each HTTP inference request:

   * The server uses `RequestScopedPipeline.generate(...)` which:

     * obtains a *local scheduler* (via `clone_for_request()` or `deepcopy`),
     * does `local_pipe = copy.copy(base_pipe)` (shallow copy),
     * sets `local_pipe.scheduler = local_scheduler` (if possible),
     * clones only small mutable attributes (callbacks, rng, small latents),
     * optionally enters a `model_cpu_offload_context()` for memory offload hooks,
     * calls the pipeline on the local view (`local_pipe(...)`).
3. **Result**: inference completes, images are moved to CPU & saved (if requested), internal buffers freed (GC + `torch.cuda.empty_cache()`).
4. Multiple requests can run in parallel while sharing heavy weights and isolating mutable state.


## How to set up and run the server

### 1) Install dependencies

Recommended: create a virtualenv / conda environment.

If using the `diffusers` fork via git, either:

**A) Preinstall the fork first:**

```bash
pip install "git+https://github.com/F4k3r22/diffusers-async.git@main"
pip install -r requirements.txt
```

### 2) Start the server

Using the `server.py` file that already has everything you need:

```bash
python server.py
```

### 3) Example request

`POST /api/diffusers/inference` with JSON body:

```json
{
  "prompt": "A futuristic cityscape, vibrant colors",
  "num_inference_steps": 30,
  "num_images_per_prompt": 1
}
```

Response example:

```json
{
  "response": ["http://localhost:8500/images/img123.png"]
}
```

## Troubleshooting (quick)

* `Already borrowed` — tokenizers (Rust) error when used concurrently.

  * Workarounds:

    * Acquire a `Lock` around tokenization or around the pipeline call (serializes that part).
    * Use the slow tokenizer (`converter_to_slow`) for concurrency tests.
    * Patch only the tokenization method to use a lock instead of serializing entire forward.
* `can't set attribute 'components'` — pipeline exposes read-only `components`.

  * The RequestScopedPipeline now detects read-only properties and skips setting them.
* Scheduler issues:

  * If the scheduler doesn't implement `clone_for_request` and `deepcopy` fails, we log and fallback — but prefer `retrieve_timesteps(..., return_scheduler=True)` to avoid mutating the shared scheduler.

