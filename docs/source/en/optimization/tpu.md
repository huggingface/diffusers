<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# TorchTPU

[TorchTPU](https://github.com/google-pytorch/torch_tpu/) is a PyTorch backend for Google's Tensor Processing Units (TPUs), which lets you run Diffusers pipelines on Cloud TPUs (v6e, v5p, etc.) with minimal code changes.

Two execution modes are available:

| Mode | Constant | How to activate | Notes |
|---|---|---|---|
| Strict eager (default) | `EagerMode.DEFER_NEVER` | `import torch_tpu` | Operations dispatched one at a time, asynchronous |
| Compile | — | `torch.compile(module, backend="tpu")` | AOT compilation with `TpuBackend` |

Follow the [TorchTPU installation guide](https://github.com/google-pytorch/torch_tpu/). After installation,
`import torch_tpu` registers the `"tpu"` device automatically.

## Eager mode

FLUX.1-schnell doesn't fit on a single v6e chip all at once, so use [`~DiffusionPipeline.enable_model_cpu_offload`] to
move each model to the TPU only while it runs. It detects the `"tpu"` device automatically.

```python
import torch
import torch_tpu  # noqa: F401

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]

image.save("output.png")
```

If a model is too large for a single chip, or you have several chips and want lower latency, shard the models
across chips instead. See the [Tensor parallelism](#tensor-parallelism) section.

## Compiled mode

`import torch_tpu` registers `"tpu"` as a `torch.compile` backend name (`TpuBackend` under the hood), so
components compile like any other `torch.compile` target — no diffusers-specific method needed. The first
call (warmup) is slow because it compiles; later calls with the same shapes reuse the compiled graph.

> [!IMPORTANT]
> TorchTPU requires **static shapes** — pass `dynamic=False`. Every time `height`, `width`, or
> `num_inference_steps` changes, the graph is recompiled from scratch. Keep these values constant
> across all calls after warmup, or run another warmup pass before changing them.

As in eager mode, [`~DiffusionPipeline.enable_model_cpu_offload`] keeps every model, text encoders included, on the
TPU while it runs. The offload hooks can't be traced by `torch.compile`, so compile the transformer's repeated blocks
with [`~ModelMixin.compile_repeated_blocks`] instead of the whole model.

```python
import torch
import torch_tpu  # noqa: F401 — registers the "tpu" torch.compile backend

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.transformer.compile_repeated_blocks(backend="tpu", fullgraph=True, dynamic=False)

# Warmup — triggers static graph compilation.
pipe(
    prompt="warmup",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
)

# Timed inference reuses the compiled graph.
image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]

image.save("output.png")
```

## Tensor parallelism

Shard models too large for one chip across several. FLUX.2-dev's text encoder (~48GB) and transformer (~64GB) each
exceed a single chip, so the example below shards both:

- the transformer with [`TensorParallelConfig`], passed to the `parallel_config` argument of [`~ModelMixin.from_pretrained`]. Each rank reads only its own slice of every sharded weight, so the full model is never materialized. For general TP details (`_tp_plan`, colwise/rowwise), see the [Tensor parallelism](../training/distributed_inference#tensor-parallelism) guide.
- the text encoder with Transformers' own [tensor parallelism](https://huggingface.co/docs/transformers/perf_infer_gpu_multi), passing `tp_plan="auto"` and the same mesh.

On TPU, initialize the process group with `backend="tpu_dist"` and build the mesh with `DeviceMesh("tpu", ...)`.

```python
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401
from torch.distributed.device_mesh import DeviceMesh
from transformers import Mistral3ForConditionalGeneration

from diffusers import Flux2Pipeline, Flux2Transformer2DModel, TensorParallelConfig

dist.init_process_group(backend="tpu_dist")
mesh = DeviceMesh("tpu", list(range(dist.get_world_size())))

repo_id = "black-forest-labs/FLUX.2-dev"
text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id, subfolder="text_encoder", dtype=torch.bfloat16, tp_plan="auto", device_mesh=mesh
)
transformer = Flux2Transformer2DModel.from_pretrained(
    repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, parallel_config=TensorParallelConfig(mesh=mesh)
)
pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=text_encoder, transformer=transformer, torch_dtype=torch.bfloat16
)
pipe.vae.to("tpu")

image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
if dist.get_rank() == 0:
    image.save("output.png")
```

Launch one process per chip. On a single host, use all of the host's chips:

```bash
eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
torchrun --nproc_per_node=8 flux2_tp.py
```
