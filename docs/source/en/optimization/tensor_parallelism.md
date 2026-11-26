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

To add a plan to another model, see [Add tensor parallelism to a model](#add-tensor-parallelism-to-a-model).

Loading this way shards the checkpoint as it's read. Each rank reads only its own slice of each weight and places it directly on its GPU, so no rank ever holds a full-size weight. Memory per rank drops as `tp_degree` increases.

```py
import os
import torch
from torch import distributed as dist
from diffusers import DiffusionPipeline, Flux2Transformer2DModel, TensorParallelConfig

def main():
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # torchrun sets LOCAL_RANK to this process's GPU index on its node
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Each rank reads only its own shard of every planned weight, straight onto its own GPU.
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

## Combine with context parallelism

Tensor parallelism shards the weights, and context parallelism splits the sequence. When a model's weights don't fit on one device and its sequence is too long for one device's attention, you need both, because neither one splits what the other does.

Set both in one [`ParallelConfig`]. [`~ModelMixin.enable_parallelism`] (or `from_pretrained(..., parallel_config=...)`) builds a single device mesh with a dimension for each, `("tp", "ring", "ulysses")`, so the communication for each strategy stays inside its own process group. The model must define both a `_tp_plan` and a `_cp_plan`. Both [Ring Attention](./context_parallelism#ring-attention) (`ring_degree`) and [Ulysses Attention](./context_parallelism#ulysses-attention) (`ulysses_degree`) work with tensor parallelism.

The example below needs 8 GPUs: `tp_degree` × `ulysses_degree` = 4 × 2.

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

    # Each weight is sharded 4 ways, and each shard's sequence is split 2 ways with Ulysses.
    transformer = Flux2Transformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        subfolder="transformer",
        dtype=torch.bfloat16,
        parallel_config=ParallelConfig(
            tensor_parallel_config=TensorParallelConfig(tp_degree=4),
            context_parallel_config=ContextParallelConfig(ulysses_degree=2),
        ),
    )

    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", transformer=transformer, dtype=torch.bfloat16
    )
    # As with tensor parallelism alone, move the other components individually, not with `pipe.to(device)`.
    pipe.text_encoder.to(device)
    pipe.vae.to(device)

    generator = torch.Generator().manual_seed(42)
    image = pipe(prompt="a cat holding a sign that says hello", generator=generator).images[0]
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

`from_pretrained` and `enable_parallelism` check the degrees before sharding any weights. [`ParallelConfig`] also checks the world size when you create it, if `torch.distributed` is already initialized.

- `tp_degree` × `ring_degree` × `ulysses_degree` must not exceed the world size.
- Ulysses splits the heads that TP left on each rank, so `num_attention_heads` must be divisible by `tp_degree` × `ulysses_degree`. Ring attention doesn't touch the head dimension, so with `ring_degree` only `tp_degree` has to divide the head count. If the heads don't divide evenly and `ring_degree` is 1, pass `ulysses_anything=True` to pad the head dimension instead. `tp_degree` still has to divide the head count.

The default mesh puts `"tp"` outermost, so each Ulysses group is a run of consecutive ranks. To use a different layout, for example to keep TP inside one NVLink domain on multi-node CUDA, build a mesh with `"tp"`, `"ring"`, and `"ulysses"` dimensions and pass it as `mesh=` on either config. The default layout suits runtimes like AWS Neuron, which only accept contiguous replica groups for all-to-all but tolerate strided groups for the all-reduce that TP needs. Only one axis of the mesh can be contiguous.

Which degree to raise first depends on what you are short of. Tensor parallelism lowers weight memory. Context parallelism lowers activation memory and latency, but leaves each device's share of the weights unchanged. For example, [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (a 33B audio-video DiT, 56 heads) was loaded with `from_pretrained(..., parallel_config=...)` at 1344×768 and 124 frames on one AWS Trainium2 instance (trn2.48xlarge). The peak memory measured during denoising was:

| NeuronCores | config | weights / core | peak memory / core | s / step |
|---|---|---|---|---|
| 8 | TP=8 | 7.84 GiB | 18.7 GiB | 9.33 |
| 16 | TP=4 × Ulysses=4 | 15.53 GiB | 21.7 GiB | 5.31 |
| 32 | TP=4 × Ulysses=8 | 15.53 GiB | 20.5 GiB | 3.97 |
| 32 | TP=8 × Ulysses=4 | 7.84 GiB | 13.7 GiB | 3.81 |

At the same 32 cores, TP=8 × Ulysses=4 is both faster and lighter than TP=4 × Ulysses=8, so on this model it pays to max out TP first and add CP on top. Its 56 heads cap TP at 8, so going further means raising the CP degree.

## Limitations

Tensor parallelism doesn't work with some loading options and features, and you can't save a sharded model.

- Loading with a tensor-parallel `parallel_config` doesn't work with `device_map`, `quantization_config`, `low_cpu_mem_usage=False`, `use_flashpack=True`, or non-safetensors weights. Each one raises an error instead of quietly loading the full checkpoint.
- Combining tensor parallelism with quantization, offloading, or LoRA adapters also raises an error, no matter how the model is sharded.
- To shard a model that's already in memory, call [`~ModelMixin.enable_parallelism`] with the same config. It loads the full model first and then shards it, so every rank needs memory for the full checkpoint. Loading with `parallel_config` avoids that cost.
- Saving a tensor-parallel model raises an error in [`~ModelMixin.save_pretrained`]. Save the model before sharding it.

## Add tensor parallelism to a model

This section is for contributors who want to add tensor parallelism support to a Diffusers model.

Tensor parallelism works only on models that define a `_tp_plan`, a class attribute that maps module names to a sharding style. Writing one mostly means pairing each projection that *expands* the hidden dimension with the projection that *contracts* it back.

Each key can contain at most one `*`, and the part before the `*` must resolve to an [`nn.ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html), so one entry covers every block. A key without a `*` applies to the model itself. Paths are relative to the model.

### Colwise and rowwise

| Style | Shards | Each rank | Use for |
|---|---|---|---|
| `"colwise"` | output features (`weight` dim 0) | computes a slice of the output | `to_q`, `to_k`, `to_v`, FFN in-projection |
| `"rowwise"` | input features (`weight` dim 1) | computes a partial sum | `to_out.0`, FFN out-projection |
| `"rowwise_split_input"` | input features (`weight` dim 1) | computes a partial sum from its slice of the replicated input | a standalone projection whose input is replicated, like an AdaLN modulation projection that reads the timestep embedding |

Always pair `"colwise"` and `"rowwise"` in that order. A `"colwise"` projection leaves its output sharded, the following `"rowwise"` projection consumes that shard directly, and a single `AllReduce` at the block boundary reconstructs the result. Sharding the pair any other way forces a gather in the middle and communicates far more.

In attention, the column-wise Q, K and V projections give each rank a subset of heads, so `tp_degree` must divide the head count. Encoder-stream duplicates (`add_q_proj`, `to_add_out`, `ff_context`) follow the same pattern as their image-stream counterparts.

### Fused projections

When one `Linear` packs several logical tensors along the dimension being sharded, plain `"colwise"`/`"rowwise"` slices straight across the concatenation and misaligns the pieces. Use `PackedColwiseParallel`/`PackedRowwiseParallel` instead, which shard each packed block independently.

```py
# in src/diffusers/models/transformers/your_model.py
from ...hooks.tensor_parallel import PackedColwiseParallel, PackedRowwiseParallel
```

Pass `blocks`, a list of proportional integers whose sum divides the packed dimension. For example, use `[1, 1]` for a SwiGLU gate and up projection of equal halves, or `[1, 1, 1, 3, 3]` for a fused Q, K, V, gate and up projection with `mlp_ratio=3`.

```py
"transformer_blocks.*.ff.linear_in": PackedColwiseParallel([1, 1]),
```

When the block sizes are only known from the config, omit `blocks` and store the absolute sizes on the `Linear` during `__init__` instead, as `_tp_packed_col_blocks` or `_tp_packed_row_blocks`.

```py
# in the attention module's __init__
self.to_out._tp_packed_row_blocks = [self.inner_dim, self.mlp_hidden_dim]
```

```py
# in the model's _tp_plan
"single_transformer_blocks.*.attn.to_out": PackedRowwiseParallel(),
```

The plan in [`transformer_flux2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py) uses both forms. The double-stream SwiGLU pair passes `blocks` directly. The fused single-stream QKV and MLP projection stores its sizes on the `Linear` in `Flux2ParallelSelfAttention.__init__`, because they depend on `mlp_ratio` and `mlp_mult_factor`.

### What to leave out

Modules left out of the plan are replicated on every rank. Leave out normalization layers, patch and text embeddings, and the final `norm_out` and `proj_out`. They're small, so sharding them saves little memory and adds communication. AdaLN modulation (`img_mod` and `txt_mod`) is usually small enough to leave out too. When it's large, like MiniMax-H3's `adaln_proj.linear`, shard it with `"rowwise_split_input"`.

### Constraints and verification

- `tp_degree` must divide `config.num_attention_heads`.
- Every packed block must *individually* be divisible by `tp_degree`, not just their sum.

[`~ModelMixin.from_pretrained`] and [`~ModelMixin.enable_parallelism`] check both before loading or sharding any weight.

Validate a new plan numerically rather than by eye: generate with a fixed seed on a single device, then again under tensor parallelism, and compare the outputs. A misplaced `"colwise"`/`"rowwise"` usually still runs and produces a plausible but wrong image.

> [!TIP]
> Start from an existing plan for a similar architecture.
>
> - [`transformer_qwenimage.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py) has no fused projections, so every entry is plain `"colwise"` or `"rowwise"`.
> - [`transformer_flux.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py) adds one packed row-wise projection.
> - [`transformer_flux2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py) uses both packed styles.
