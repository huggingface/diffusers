<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IDLMBlockDiffusionScheduler

`IDLMBlockDiffusionScheduler` implements the block-N Introspective Strided Decoding step for I-DLM: speculative verification via `min(1, p/(alpha*q))` with `max(0, p - alpha*q)` resampling on reject, plus sampling of the next batch of speculative tokens from the MASK-position anchor logits. It is stateless and pure-math — the pipeline owns model and cache I/O.

## IDLMBlockDiffusionScheduler
[[autodoc]] IDLMBlockDiffusionScheduler

## IDLMBlockDiffusionSchedulerOutput
[[autodoc]] schedulers.scheduling_idlm_block_diffusion.IDLMBlockDiffusionSchedulerOutput
