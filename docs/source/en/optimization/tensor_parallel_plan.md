<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Add tensor parallelism to a model

This guide is for contributors who want to add tensor parallelism support to a Diffusers model. To run a model that already supports it, refer to the [Tensor parallelism](./tensor_parallelism) guide.

Tensor parallelism works only on models that define a `_tp_plan`, a class attribute that maps module names to a sharding style. Writing one mostly means pairing each projection that *expands* the hidden dimension with the projection that *contracts* it back.

Each key can contain at most one `*`, and the part before the `*` must resolve to an [`nn.ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html), so one entry covers every block. A key without a `*` applies to the model itself. Paths are relative to the model.

## Colwise and rowwise

| Style | Shards | Each rank | Use for |
|---|---|---|---|
| `"colwise"` | output features (`weight` dim 0) | computes a slice of the output | `to_q`, `to_k`, `to_v`, FFN in-projection |
| `"rowwise"` | input features (`weight` dim 1) | computes a partial sum | `to_out.0`, FFN out-projection |

Always pair them in that order. A `"colwise"` projection leaves its output sharded, the following `"rowwise"` projection consumes that shard directly, and a single `AllReduce` at the block boundary reconstructs the result. Sharding the pair any other way forces a gather in the middle and communicates far more.

In attention, the column-wise Q, K and V projections give each rank a subset of heads, so `tp_degree` must divide the head count. Encoder-stream duplicates (`add_q_proj`, `to_add_out`, `ff_context`) follow the same pattern as their image-stream counterparts.

## Fused projections

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

## What to leave out

Modules left out of the plan are replicated on every rank. Leave out normalization layers, AdaLN modulation (`img_mod` and `txt_mod`), patch and text embeddings, and the final `norm_out` and `proj_out`. They're small, so sharding them saves little memory and adds communication.

## Constraints and verification

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
