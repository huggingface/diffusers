<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Tensor parallelism

[Tensor parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) shards the weight matrices of a model across devices. Each device holds a column-wise (`"colwise"`) or row-wise (`"rowwise"`) slice of each layer and computes a partial result. An `AllReduce` or `AllGather` at the layer boundary combines the partial results into the full output. Unlike [context parallelism](./context_parallelism), it reduces *weight* memory per device, which helps when a model doesn't fit on one device. Refer to [Distributed inference](../training/distributed_inference) to compare it with the other strategies.

## Load a sharded model

Pass a [`TensorParallelConfig`] to the `parallel_config` argument of the model's [`~ModelMixin.from_pretrained`]. `tp_degree` is the number of devices to shard across and must divide the model's number of attention heads. The model's class must define a `_tp_plan`, which maps module names to a sharding style. To check whether a model supports tensor parallelism, look for this attribute on its class, such as [`Flux2Transformer2DModel`].

To add a plan to another model, refer to [Add tensor parallelism to a model](./tensor_parallel_plan).

Loading this way shards the checkpoint as it's read. Each rank reads only its own slice of each weight and places it directly on its GPU, so no rank ever holds a full-size weight. Memory per rank drops as `tp_degree` increases.

```py
import torch
from torch import distributed as dist
from diffusers import DiffusionPipeline, Flux2Transformer2DModel, TensorParallelConfig

def main():
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Each rank reads only its own shard of every planned weight, straight onto `cuda:rank`.
    transformer = Flux2Transformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        subfolder="transformer",
        dtype=torch.bfloat16,
        parallel_config=TensorParallelConfig(tp_degree=world_size),
    )

    pipeline = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", transformer=transformer, dtype=torch.bfloat16
    )
    # The transformer is already on its device; move the remaining components individually. Do not call
    # `pipeline.to(device)` — that would move every rank's shards onto the same device.
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

Save the script as `tensor_parallel_flux2.py` and launch it with `torchrun`.

```shell
torchrun --nproc-per-node 4 tensor_parallel_flux2.py
```

The script sets `tp_degree` to `world_size`, so `--nproc-per-node 4` shards the transformer across 4 GPUs.

Only the transformer is sharded. Each rank holds a full copy of the text encoder, so each GPU needs enough memory for the text encoder plus its share of the transformer.

## Loading performance

Loading with `parallel_config` is faster and uses less CPU memory per rank than loading the full model and then calling [`~ModelMixin.enable_parallelism`]. The table below shows results for [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (transformer only, 32B params, bf16) on 4x A10G (23GB).

| tp_degree | method | load time | peak CPU/rank |
|---|---|---|---|
| 4 | `from_pretrained(parallel_config=...)` | 12.5s | 6.8GB |
| 4 | `from_pretrained` + `enable_parallelism` | 30.4s | 64.1GB |

## Limitations

Tensor parallelism doesn't work with some loading options and features, and you can't save a sharded model.

> [!CAUTION]
> Loading with a tensor-parallel `parallel_config` doesn't work with `device_map`, `quantization_config`, `low_cpu_mem_usage=False`, `use_flashpack=True`, or non-safetensors weights. Each one raises an error instead of quietly loading the full checkpoint.
>
> Combining tensor parallelism with quantization, offloading, or LoRA adapters also raises an error, no matter how the model is sharded.
>
> To shard a model that is already in memory, call [`~ModelMixin.enable_parallelism`] with the same config instead. It loads the full model first and then shards it, so every rank needs memory for the full checkpoint.

Saving a tensor-parallel model raises an error in [`~ModelMixin.save_pretrained`]. Save the model before sharding it.
