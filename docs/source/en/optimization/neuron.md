<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AWS Trainium

[TorchNeuron](https://github.com/aws-neuron/torch-neuronx) is the native PyTorch backend for Trainium2 and Trainium3. Diffusers runs on these instances through the `torch_neuronx` package, which registers a `"neuron"` device, so pipelines run with the same code as on any other device.

Two execution modes are available:

| Mode | How to activate | Notes |
|---|---|---|
| Eager | `pipe.to("neuron")` | Operations dispatched one at a time, asynchronous |
| Compile | `torch.compile(module, backend="neuron")` | Whole-graph compilation, static shapes |

Follow the [AWS Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) to set up the Neuron
SDK and install `torch_neuronx`. Once installed, `import torch` loads it automatically and registers the `"neuron"`
device, so `pipe.to("neuron")` is the only change needed. Add `import torch_neuronx` only if you disabled backend
autoloading with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.

In eager mode, each operation compiles the first time it runs. In compiled mode, the first call compiles the graph. Both modes cache the compiled Neuron executable format (NEFF) under `tmp/neff_cache` by default, and that directory is cleared on reboot. Set `TORCH_NEURONX_NEFF_CACHE_DIR` to a persistent directory to keep the cache.

## Eager mode

FLUX.2-klein-4B fits on a single NeuronCore, so move the whole pipeline to the `"neuron"` device.

```python
import torch

from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=torch.bfloat16)
pipe.to("neuron")

image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=4,
).images[0]

image.save("output.png")
```

If a model is too large for a single NeuronCore, or you have several cores and want lower latency, shard the models
across cores instead. See the [Tensor parallelism](#tensor-parallelism) section.

## Compiled mode

TorchNeuron registers `"neuron"` as a `torch.compile` backend name, so components compile like any other
`torch.compile` target. The first call (warmup) is slow because it compiles; later calls with the same shapes reuse the compiled graph.

> [!IMPORTANT]
> TorchNeuron requires *static shapes*, so pass `dynamic=False`. A new `height` or `width` compiles again for that shape. Shapes already seen are reused. Changing `num_inference_steps` doesn't recompile.

```python
import torch

from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=torch.bfloat16)
pipe.to("neuron")
pipe.transformer.compile(backend="neuron", fullgraph=True, dynamic=False)

# Warmup — triggers static graph compilation.
pipe(prompt="warmup", height=1024, width=1024, num_inference_steps=4)

# Later calls with the same shapes reuse the compiled graph.
image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=4,
).images[0]

image.save("output.png")
```

For transformers with repeated blocks, use `compile_repeated_blocks(...)` to compile those blocks individually. This block is reused and reduces compile time considerably.

## Tensor parallelism

Follow the [Tensor parallelism](../training/distributed_inference#tensor-parallelism) guide for model loading,
sharding plans, and limitations. To adapt its example for Neuron, replace the CUDA process-group and device setup
with the following.

On Neuron, initialize the process group with `backend="neuron"` and build the mesh with `DeviceMesh("neuron", ...)`.

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from diffusers import TensorParallelConfig

dist.init_process_group(backend="neuron")
mesh = DeviceMesh("neuron", list(range(dist.get_world_size())))
device = torch.neuron.current_device()
parallel_config = TensorParallelConfig(mesh=mesh)
```

Pass this `parallel_config` to the transformer's [`~ModelMixin.from_pretrained`] and use `device` for the remaining
pipeline components.

Launch one process per NeuronCore. Set `--nproc_per_node` to the number of NeuronCores you want to shard across.

```bash
torchrun --nproc_per_node=8 tensor_parallel_neuron.py
```
