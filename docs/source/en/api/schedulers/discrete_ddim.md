<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DiscreteDDIMScheduler

The `DiscreteDDIMScheduler` samples each canvas position from the exact discrete posterior of the uniform corruption
process (D3PM), following [Structured Denoising Diffusion Models in Discrete State-Spaces](https://huggingface.co/papers/2107.03006).
The final step deterministically commits the predicted tokens. An optional predictor-corrector
mode adds the leave-one-out Gibbs sweeps of [Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation](https://huggingface.co/papers/2605.22765)
through `corrector_steps`.

This scheduler is used by [`DiffusionGemmaPipeline`].


This scheduler follows the shared [discrete diffusion scheduler](overview#discrete-diffusion-schedulers) contract: a decreasing
`float` corruption level in `(0, 1]`, `step(model_output, timestep, sample)`, sampling knobs on the config, and a
[`DiscreteSchedulerOutput`] return.

## DiscreteDDIMScheduler
[[autodoc]] DiscreteDDIMScheduler

## DiscreteSchedulerOutput
[[autodoc]] schedulers.scheduling_utils.DiscreteSchedulerOutput
