<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UniformRefinementScheduler

The `UniformRefinementScheduler` denoises the uniform corruption process by committing tokens in order of
confidence. Unlike the absorbing (masked) process of [`BlockRefinementScheduler`], there is no mask token: every
position always holds a real token, so the set of positions still undecided is tracked as scheduler state and the
undecided ones are renoised with uniformly random tokens after each step.

Because that state is per-block, call [`~UniformRefinementScheduler.set_timesteps`] at the start of each block.
Denoising past `num_inference_steps` raises rather than over-committing.

This scheduler is used by [`DiffusionGemmaPipeline`].


This scheduler follows the shared [discrete diffusion scheduler](overview#discrete-diffusion-schedulers) contract: a decreasing
`float` corruption level in `(0, 1]`, `step(model_output, timestep, sample)`, sampling knobs on the config, and a
[`DiscreteSchedulerOutput`] return.

## UniformRefinementScheduler
[[autodoc]] UniformRefinementScheduler

## DiscreteSchedulerOutput
[[autodoc]] schedulers.scheduling_utils.DiscreteSchedulerOutput
