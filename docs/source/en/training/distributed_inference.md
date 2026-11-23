<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Distributed inference

Distributed inference splits the workload across multiple GPUs. It's useful for fitting larger models in memory and for processing more prompts at once.

Diffusers supports four strategies. Each one has its own guide, and this page helps you pick one.

## Choosing a strategy

Each strategy solves a different problem, so choose one based on the resource that limits you.

| Strategy | Splits | Reduces | Latency for one prompt | Best when |
|---|---|---|---|---|
| [Data parallelism](../optimization/data_parallelism) | prompts across replicas | nothing, since each GPU holds a full copy | unchanged | the model already fits and you have many prompts |
| [device_map](../optimization/device_map) | components across devices | weight memory | slightly worse | the model doesn't fit and the interconnect is slow |
| [Context parallelism](../optimization/context_parallelism) | the input sequence | activation memory | lower | sequences are long, like high-resolution images or video |
| [Tensor parallelism](../optimization/tensor_parallelism) | weight matrices | weight memory | lower | one component's weights don't fit and the interconnect is fast |

Some practical guidance:

- If the model already fits and you have many prompts, use data parallelism. It scales throughput with the number of GPUs without touching the model, and it leaves single-prompt latency alone. Set it up with [Accelerate](../optimization/data_parallelism#accelerate) or [PyTorch Distributed](../optimization/data_parallelism#pytorch-distributed).
- If a single component's weights don't fit, try tensor parallelism first, since it lowers both memory and latency. Tensor parallelism communicates at every block boundary, so it needs a fast interconnect like NVLink. On PCIe, the per-layer traffic can outweigh the compute it saves, and `device_map` becomes the better choice. `device_map` also handles the case where the components are individually fine but collectively too large.
- If activations, not weights, are the problem, use context parallelism. This helps with long sequences, like large images and videos with many frames. Pick a backend from the benchmarks in the [Context parallelism](../optimization/context_parallelism) guide.
- If both the weights and the sequence are too large, [combine tensor and context parallelism](#combining-context-and-tensor-parallelism) in one [`ParallelConfig`]. Raise `tp_degree` until the weights fit, then add context parallelism on top for latency.

The [xDiT](../optimization/xdit) and [ParaAttention](../optimization/para_attn) community integrations offer their own context parallelism implementations.

## Combining context and tensor parallelism

The two strategies split a model along different axes: tensor parallelism shards the weights, context parallelism splits the sequence. A model whose weights don't fit on one device *and* whose sequence is too long for one device's attention needs both, and neither alone can help: TP doesn't divide the sequence, and CP doesn't divide the weights.

Set both in one [`ParallelConfig`]. [`~ModelMixin.enable_parallelism`] (or `from_pretrained(..., parallel_config=...)`) builds a single device mesh with a dimension for each, `("tp", "ring", "ulysses")`, so the TP all-reduces and the CP all-gathers or all-to-alls each stay inside their own process group. The model must define both a `_tp_plan` and a `_cp_plan`.

```py
import os
import torch
from torch import distributed as dist
from diffusers import (
    ContextParallelConfig,
    DiffusionPipeline,
    Flux2Transformer2DModel,
    ParallelConfig,
    TensorParallelConfig,
)

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    # torchrun sets LOCAL_RANK to this process's GPU index on its node
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    # 8 devices: each weight is sharded 4 ways, and each shard's sequence is split 2 ways with Ulysses.
    transformer = Flux2Transformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        subfolder="transformer",
        dtype=torch.bfloat16,
        parallel_config=ParallelConfig(
            tensor_parallel_config=TensorParallelConfig(tp_degree=4),
            context_parallel_config=ContextParallelConfig(ulysses_degree=2),
        ),
    )

    pipeline = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", transformer=transformer, dtype=torch.bfloat16
    )
    # As with tensor parallelism alone, move the other components individually, not with `pipeline.to(device)`.
    pipeline.text_encoder.to(device)
    pipeline.vae.to(device)

    generator = torch.Generator().manual_seed(42)
    image = pipeline(prompt="a cat holding a sign that says hello", generator=generator).images[0]
    if rank == 0:
        image.save("output.png")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Save the script as `tp_cp_flux2.py` and launch it with `torchrun`.

```shell
torchrun --nproc-per-node 8 tp_cp_flux2.py
```

Only the transformer is sharded. Each rank holds a full copy of the text encoder, so each GPU needs enough memory for the text encoder plus its share of the transformer.

The degrees are checked before anything is sharded:

- `tp_degree` × `ring_degree` × `ulysses_degree` must not exceed the world size.
- Ulysses splits the heads that TP left on each rank, so `num_attention_heads` must be divisible by `tp_degree` × `ulysses_degree`. Ring attention doesn't touch the head dimension, so with `ring_degree` only `tp_degree` has to divide the head count. Pass `ulysses_anything=True` to pad the heads instead when they don't divide.

The default mesh puts `"tp"` outermost, so the Ulysses groups are runs of consecutive ranks. Some accelerator runtimes, AWS Neuron among them, only accept contiguous replica groups for all-to-all while tolerating strided groups for the all-reduce TP needs, and only one axis of the mesh can be contiguous. On multi-node CUDA the opposite order is usually better, since TP's per-layer all-reduce is the most bandwidth-hungry collective and wants to stay inside one NVLink domain. To choose the layout yourself, build the mesh with `"tp"`, `"ring"` and `"ulysses"` dimensions and pass it as `mesh=` on either config.

Which degree to raise first depends on what you are short of. TP is what lowers per-device memory; CP mostly lowers latency. For example, [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (a 33B audio-video DiT, 56 heads) at 1344×768 and 124 frames on one AWS Trainium2 instance (trn2.48xlarge, loaded with `from_pretrained(..., parallel_config=...)`; peak memory is measured during denoising):

| NeuronCores | config | weights / core | peak memory / core | s / step |
|---|---|---|---|---|
| 8 | TP=8 | 7.84 GiB | 18.7 GiB | 9.33 |
| 16 | TP=4 × Ulysses=4 | 15.53 GiB | 21.7 GiB | 5.31 |
| 32 | TP=4 × Ulysses=8 | 15.53 GiB | 20.5 GiB | 3.97 |
| 32 | TP=8 × Ulysses=4 | 7.84 GiB | 13.7 GiB | 3.81 |

At the same 32 cores, TP=8 × Ulysses=4 is both faster and lighter than TP=4 × Ulysses=8, so on this model it pays to max out TP first and add CP on top. Its 56 heads cap TP at 8, so going further means raising the CP degree.
