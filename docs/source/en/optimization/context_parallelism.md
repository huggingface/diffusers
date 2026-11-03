<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Context parallelism

[Context parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism) splits the input sequence across GPUs, and each GPU processes its own slice. This lowers memory per GPU, which helps most with long sequences like high-resolution images and video. To compare it with the other strategies, see [Distributed inference](../training/distributed_inference).

Context parallelism runs inside the attention backend, and only some backends support it. If yours doesn't, `enable_parallelism` raises an error that lists the compatible backends. Switch backends with [`~ModelMixin.set_attention_backend`], and see the [available backends](./attention_backends#available-backends) for the full list.

Configure every variant below with [`ContextParallelConfig`].

| Variant | Config | Requirements | Use when |
|---|---|---|---|
| [Ring Attention](#ring-attention) | `ring_degree > 1` | the sequence length must divide evenly by `ring_degree` | sequences are long and interconnect bandwidth is limited |
| [Ulysses Attention](#ulysses-attention) | `ulysses_degree > 1` | the sequence length and the number of attention heads must divide evenly by `ulysses_degree` | your GPUs have a fast interconnect |
| [Unified Attention](#unified-attention) | `ring_degree > 1` and `ulysses_degree > 1` | at least 4 GPUs | you have more devices than Ulysses Attention can use alone |
| [Ulysses Anything Attention](#ulysses-anything-attention) | `ulysses_degree > 1`, `ulysses_anything=True` | `ring_degree` must be 1 | the sequence length or head count doesn't divide evenly |
| [Ring Anything Attention](#ring-anything-attention) | `ring_degree > 1`, `ring_anything=True` | `ulysses_degree` must be 1, inference only, `attn_mask` must be `None` | the sequence length doesn't divide evenly |

## Ring Attention

[Ring Attention](https://huggingface.co/papers/2310.01889) passes key (K) and value (V) blocks between GPUs arranged in a ring. Each GPU computes attention against the K/V block it holds, then passes that block to the next GPU. After one full loop, every GPU has seen every token's K/V, and no GPU ever holds the full sequence.

Set `ring_degree` to the number of GPUs, and pass the config to the transformer's [`~ModelMixin.enable_parallelism`] method.

```py
import torch
from torch import distributed as dist
from diffusers import DiffusionPipeline, ContextParallelConfig

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return device

def main():
    device = setup_distributed()
    world_size = dist.get_world_size()

    pipeline = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16
    ).to(device)
    pipeline.transformer.set_attention_backend("_native_cudnn")

    cp_config = ContextParallelConfig(ring_degree=world_size)
    pipeline.transformer.enable_parallelism(config=cp_config)

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """

    # Must specify generator so all ranks start with same latents (or pass your own)
    generator = torch.Generator().manual_seed(42)
    image = pipeline(
        prompt,
        guidance_scale=3.5,
        num_inference_steps=50,
        generator=generator,
    ).images[0]

    if dist.get_rank() == 0:
        image.save("output.png")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Save the script as `ring_attention.py` and launch it with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) or another PyTorch-compatible launcher. Set `--nproc-per-node` to the number of GPUs.

```bash
torchrun --nproc-per-node 2 ring_attention.py
```

To try the other variants, replace the `enable_parallelism` call in `ring_attention.py` with the one from their section.

## Ulysses Attention

[Ulysses Attention](https://huggingface.co/papers/2309.14509) uses *all-to-all* communication, where every GPU exchanges data with every other GPU. The first all-to-all regroups the data so each GPU holds every token for a subset of attention heads. Each GPU computes attention for its heads, and a second all-to-all regroups the results by token for the next layer.

Set `ulysses_degree` to the number of GPUs, and pass the config to [`~ModelMixin.enable_parallelism`].

```py
pipeline.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2))
```

## Unified Attention

[Unified Sequence Parallelism](https://huggingface.co/papers/2405.07719) combines Ring Attention and Ulysses Attention. It runs an all-to-all to redistribute heads and tokens, runs Ring Attention on the result, then reverses the all-to-all to restore the original layout. Ulysses parallelizes across attention heads and Ring parallelizes across the sequence, so together they can use more GPUs than Ulysses alone.

Use Unified Attention when you have at least 4 GPUs. Set both `ulysses_degree` and `ring_degree` to more than 1. The GPUs form a 2D grid of `ulysses_degree * ring_degree` devices, and each GPU belongs to one Ulysses group and one Ring group.

```py
pipeline.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2, ring_degree=2))
```

The benchmark below compares Ulysses, Ring and Unified Attention on 4 H100 GPUs with [this script](https://github.com/huggingface/diffusers/pull/12693#issuecomment-3694727532). Each image uses 50 inference steps.

| CP Backend         | Time / Image (ms) | Steps / Sec | Peak Memory (GB) |
|--------------------|------------------|-------------|------------------|
| ulysses            | 6670.789         | 7.50        | 33.85            |
| ring               | 13076.492        | 3.82        | 56.02            |
| unified_balanced   | 11068.705        | 4.52        | 33.85            |

Ulysses has the highest throughput, but it can't use more GPUs than the model has attention heads. Unified Attention removes that limit.

## Ulysses Anything Attention

Ulysses Attention requires the sequence length and the number of attention heads to divide evenly by `ulysses_degree`. [Ulysses Anything Attention](https://github.com/huggingface/diffusers/pull/12996) removes both requirements. You can't combine it with Unified Attention, so `ring_degree` must be 1.

Set `ulysses_degree` to more than 1 and `ulysses_anything=True`.

```py
pipeline.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2, ulysses_anything=True))
```

> [!TIP]
> Add a `gloo` backend for CPU tensors to [init_process_group](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group). It avoids the forced CUDA syncs that host-to-device and device-to-host transfers cause.

```py
import torch.distributed as dist

dist.init_process_group(backend="cpu:gloo,cuda:nccl")
```

The benchmark below runs FLUX.1-dev on 4 L20 GPUs with [this script](https://github.com/huggingface/diffusers/pull/12996#issuecomment-3797695999).

| CP Backend         | Time / Iter (ms) | Steps / Sec | Peak Memory (GB) | Shape (HxW)|
|--------------------|------------------|-------------|------------------|------------|
| ulysses            |   281.07         |    3.56     |     37.11        | 1024x1024  |
| ring               |   351.34         |    2.85     |     37.01        | 1024x1024  |
| unified_balanced   |   324.37         |    3.08     |     37.16        | 1024x1024  |
| ulysses_anything   |   280.94         |    3.56     |     37.11        | 1024x1024  |
| ulysses            |   failed         |    failed   |     failed       | 1008x1008  |
| ring               |   failed         |    failed   |     failed       | 1008x1008  |
| unified_balanced   |   failed         |    failed   |     failed       | 1008x1008  |
| ulysses_anything   |   278.40         |    3.59     |     36.99        | 1008x1008  |

At 1008x1008, where the sequence length doesn't divide evenly, only Ulysses Anything Attention runs. At 1024x1024, it performs the same as standard Ulysses Attention.

## Ring Anything Attention

Ring Attention requires the sequence length to divide evenly by `ring_degree`. [Ring Anything Attention](https://github.com/huggingface/diffusers/pull/13545#issuecomment-4302195582) supports any sequence length. It pads each GPU's local K/V to the longest local length, all-gathers the padded K/V, and slices each block back to its true length before computing attention.

You can't combine it with Unified Attention, so `ulysses_degree` must be 1. Set `ring_degree` to more than 1 and `ring_anything=True`.

```py
pipeline.transformer.enable_parallelism(config=ContextParallelConfig(ring_degree=2, ring_anything=True))
```

Ring Anything Attention also benefits from the `gloo` backend shown in [Ulysses Anything Attention](#ulysses-anything-attention).

> [!NOTE]
> Ring Anything Attention supports inference only, and `attn_mask` must be `None`.

The benchmark below runs FLUX.1-dev on 4 RTX 4090 (48GB) GPUs.

| CP Backend         | Time / Iter (ms) | Steps / Sec | Peak Memory (GB) | Shape (HxW)|
|--------------------|------------------|-------------|------------------|------------|
| ulysses            |   259.07         |    3.86     |     33.83        | 1024x1024  |
| ring               |   338.98         |    2.95     |     33.83        | 1024x1024  |
| unified_balanced   |   321.54         |    3.11     |     33.83        | 1024x1024  |
| ulysses_anything   |   259.07         |    3.86     |     33.83        | 1024x1024  |
| ring_anything      |   340.14         |    2.94     |     33.83        | 1024x1024  |
| ulysses            |   failed         |    failed   |     failed       | 1008x1008  |
| ring               |   failed         |    failed   |     failed       | 1008x1008  |
| unified_balanced   |   failed         |    failed   |     failed       | 1008x1008  |
| ulysses_anything   |   253.16         |    3.95     |     33.75        | 1008x1008  |
| ring_anything      |   335.57         |    2.98     |     33.75        | 1008x1008  |

Ring Anything Attention runs at 1008x1008 and performs about the same as standard Ring Attention. For the motivation and trade-offs, see [this comment](https://github.com/huggingface/diffusers/pull/13545#issuecomment-4304104462).

## Enable context parallelism while loading

Instead of calling `enable_parallelism` after loading, pass the [`ContextParallelConfig`] to the `parallel_config` argument of [`~ModelMixin.from_pretrained`]. Initialize the process group first, because `from_pretrained` raises an error if `torch.distributed` isn't initialized. Launch the script with `torchrun`, as in the [Ring Attention](#ring-attention) example.

```py
import torch
from torch import distributed as dist
from diffusers import AutoModel, ContextParallelConfig, DiffusionPipeline

CKPT_ID = "black-forest-labs/FLUX.1-dev"

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

cp_config = ContextParallelConfig(ring_degree=dist.get_world_size())
transformer = AutoModel.from_pretrained(
    CKPT_ID,
    subfolder="transformer",
    dtype=torch.bfloat16,
    parallel_config=cp_config
)

pipeline = DiffusionPipeline.from_pretrained(
    CKPT_ID, transformer=transformer, dtype=torch.bfloat16,
).to(device)
```
