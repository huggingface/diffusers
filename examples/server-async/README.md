# Asynchronous server and parallel execution of models

> Example/demo server that keeps a single model in memory while safely running parallel inference requests by creating per-request lightweight views and cloning only small, stateful components (schedulers, RNG state, small mutable attrs). Works with StableDiffusion3 pipelines.
> We recommend running 10 to 50 inferences in parallel for optimal performance, averaging between 25 and 30 seconds to 1 minute and 1 minute and 30 seconds. (This is only recommended if you have a GPU with 35GB of VRAM or more; otherwise, keep it to one or two inferences in parallel to avoid decoding or saving errors due to memory shortages.)

## ⚠️ IMPORTANT

* The example demonstrates how to run pipelines like `StableDiffusion3-3.5` concurrently while keeping a single copy of the heavy model parameters on GPU.

## Necessary components

All the components needed to create the inference server are in the current directory:

```
server-async/
├── utils/
├─────── __init__.py
├─────── scheduler.py              # BaseAsyncScheduler wrapper and async_retrieve_timesteps for secure inferences
├─────── requestscopedpipeline.py  # RequestScoped Pipeline for inference with a single in-memory model
├─────── utils.py                  # Image/video saving utilities and service configuration
├── Pipelines.py                   # pipeline loader classes (SD3)
├── serverasync.py                 # FastAPI app with lifespan management and async inference endpoints
├── test.py                        # Client test script for inference requests
├── requirements.txt               # Dependencies
└── README.md                      # This documentation
```

## What `diffusers-async` adds / Why we needed it

Core problem: a naive server that calls `pipe.__call__` concurrently can hit **race conditions** (e.g., `scheduler.set_timesteps` mutates shared state) or explode memory by deep-copying the whole pipeline per-request.

`diffusers-async` / this example addresses that by:

* **Request-scoped views**: `RequestScopedPipeline` creates a shallow copy of the pipeline per request so heavy weights (UNet, VAE, text encoder) remain shared and *are not duplicated*.
* **Per-request mutable state**: stateful small objects (scheduler, RNG state, small lists/dicts, callbacks) are cloned per request. The system uses `BaseAsyncScheduler.clone_for_request(...)` for scheduler cloning, with fallback to safe `deepcopy` or other heuristics.
* **Tokenizer concurrency safety**: `RequestScopedPipeline` now manages an internal tokenizer lock with automatic tokenizer detection and wrapping. This ensures that Rust tokenizers are safe to use under concurrency — race condition errors like `Already borrowed` no longer occur.
* **`async_retrieve_timesteps(..., return_scheduler=True)`**: fully retro-compatible helper that returns `(timesteps, num_inference_steps, scheduler)` without mutating the shared scheduler. For users not using `return_scheduler=True`, the behavior is identical to the original API.
* **Robust attribute handling**: wrapper avoids writing to read-only properties (e.g., `components`) and auto-detects small mutable attributes to clone while avoiding duplication of large tensors. Configurable tensor size threshold prevents cloning of large tensors.
* **Enhanced scheduler wrapping**: `BaseAsyncScheduler` automatically wraps schedulers with improved `__getattr__`, `__setattr__`, and debugging methods (`__repr__`, `__str__`).

## How the server works (high-level flow)

1. **Single model instance** is loaded into memory (GPU/MPS) when the server starts.
2. On each HTTP inference request:

   * The server uses `RequestScopedPipeline.generate(...)` which:

     * automatically wraps the base scheduler in `BaseAsyncScheduler` (if not already wrapped),
     * obtains a *local scheduler* (via `clone_for_request()` or `deepcopy`),
     * does `local_pipe = copy.copy(base_pipe)` (shallow copy),
     * sets `local_pipe.scheduler = local_scheduler` (if possible),
     * clones only small mutable attributes (callbacks, rng, small latents) with auto-detection,
     * wraps tokenizers with thread-safe locks to prevent race conditions,
     * optionally enters a `model_cpu_offload_context()` for memory offload hooks,
     * calls the pipeline on the local view (`local_pipe(...)`).
3. **Result**: inference completes, images are moved to CPU & saved (if requested), internal buffers freed (GC + `torch.cuda.empty_cache()`).
4. Multiple requests can run in parallel while sharing heavy weights and isolating mutable state.

## How to set up and run the server

### 1) Install dependencies

Recommended: create a virtualenv / conda environment.

```bash
pip install diffusers
pip install -r requirements.txt
```

### 2) Start the server

Using the `serverasync.py` file that already has everything you need:

```bash
python serverasync.py
```

The server will start on `http://localhost:8500` by default with the following features:
- FastAPI application with async lifespan management
- Automatic model loading and pipeline initialization
- Request counting and active inference tracking
- Memory cleanup after each inference
- CORS middleware for cross-origin requests

### 3) Test the server

Use the included test script:

```bash
python test.py
```

Or send a manual request:

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

### 4) Server endpoints

- `GET /` - Welcome message
- `POST /api/diffusers/inference` - Main inference endpoint
- `GET /images/{filename}` - Serve generated images
- `GET /api/status` - Server status and memory info

## Advanced Configuration

### RequestScopedPipeline Parameters

```python
RequestScopedPipeline(
    pipeline,                        # Base pipeline to wrap
    mutable_attrs=None,             # Custom list of attributes to clone
    auto_detect_mutables=True,      # Enable automatic detection of mutable attributes
    tensor_numel_threshold=1_000_000, # Tensor size threshold for cloning
    tokenizer_lock=None,            # Custom threading lock for tokenizers
    wrap_scheduler=True             # Auto-wrap scheduler in BaseAsyncScheduler
)
```

### BaseAsyncScheduler Features

* Transparent proxy to the original scheduler with `__getattr__` and `__setattr__`
* `clone_for_request()` method for safe per-request scheduler cloning
* Enhanced debugging with `__repr__` and `__str__` methods
* Full compatibility with existing scheduler APIs

### Server Configuration

The server configuration can be modified in `serverasync.py` through the `ServerConfigModels` dataclass:

```python
@dataclass
class ServerConfigModels:
    model: str = 'stabilityai/stable-diffusion-3.5-medium'  
    type_models: str = 't2im'  
    host: str = '0.0.0.0' 
    port: int = 8500
```

## Troubleshooting (quick)

* `Already borrowed` — previously a Rust tokenizer concurrency error.
  ✅ This is now fixed: `RequestScopedPipeline` automatically detects and wraps tokenizers with thread locks, so race conditions no longer happen.

* `can't set attribute 'components'` — pipeline exposes read-only `components`.
  ✅ The RequestScopedPipeline now detects read-only properties and skips setting them automatically.

* Scheduler issues:
  * If the scheduler doesn't implement `clone_for_request` and `deepcopy` fails, we log and fallback — but prefer `async_retrieve_timesteps(..., return_scheduler=True)` to avoid mutating the shared scheduler.
  ✅ Note: `async_retrieve_timesteps` is fully retro-compatible — if you don't pass `return_scheduler=True`, the behavior is unchanged.

* Memory issues with large tensors:
  ✅ The system now has configurable `tensor_numel_threshold` to prevent cloning of large tensors while still cloning small mutable ones.

* Automatic tokenizer detection:
  ✅ The system automatically identifies tokenizer components by checking for tokenizer methods, class names, and attributes, then applies thread-safe wrappers.