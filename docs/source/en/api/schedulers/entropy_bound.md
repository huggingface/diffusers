<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EntropyBoundScheduler

The `EntropyBoundScheduler` commits the lowest-entropy positions whose joint entropy stays under `entropy_bound`, so
roughly independent tokens are accepted together and the rest are renoised. It anneals its sampling temperature from
`t_max` on the first step down to `t_min` on the last, matching the released checkpoint's sampler. Proposed in
[Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://huggingface.co/papers/2505.24857).

This scheduler is used by [`DiffusionGemmaPipeline`].

## EntropyBoundScheduler
[[autodoc]] EntropyBoundScheduler

## EntropyBoundSchedulerOutput
[[autodoc]] schedulers.scheduling_entropy_bound.EntropyBoundSchedulerOutput
