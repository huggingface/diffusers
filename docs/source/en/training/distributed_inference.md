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
- If both the weights and the sequence are too large, Diffusers can't apply tensor and context parallelism to the same model. `ParallelConfig` raises an error if you pass both configs, and `enable_parallelism` can't be called a second time on a model that's already parallelized.

The [xDiT](../optimization/xdit) and [ParaAttention](../optimization/para_attn) community integrations offer their own context parallelism implementations.
