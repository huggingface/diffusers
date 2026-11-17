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
