<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LTXEulerAncestralRFScheduler

The `LTXEulerAncestralRFScheduler` implements a K-diffusion-style Euler-Ancestral sampler
for flow / CONST parameterization, closely mirroring ComfyUI's `sample_euler_ancestral_RF`
implementation used for [LTX-Video](https://huggingface.co/docs/diffusers/api/pipelines/ltx_video).

The scheduler operates on a normalized sigma schedule σ ∈ [0, 1] and reconstructs the clean
estimate as `x0 = x_t − σ_t · v_t` (CONST parametrization). Stochastic noise reinjection is
controlled by `eta` (`eta=0` gives a deterministic Euler step; `eta=1` matches ComfyUI's
default RF behavior).

This scheduler is used by [`LTXPipeline`], [`LTXImageToVideoPipeline`], and
[`LTXConditionPipeline`].

The `eta` parameter must be >= 0. `eta=0` gives a deterministic (DDIM-like) Euler step;
`eta=1` matches ComfyUI's default RF behavior. Values above 1 are accepted but trigger a
one-time warning when the schedule step is too coarse to keep `sigma_down` non-negative.

<Tip>

See also [`FlowMatchEulerDiscreteScheduler`], which this scheduler delegates to for
auto-generated sigma schedules and shares config compatibility with via `_compatibles`.

</Tip>

## LTXEulerAncestralRFScheduler
[[autodoc]] LTXEulerAncestralRFScheduler

## LTXEulerAncestralRFSchedulerOutput
[[autodoc]] schedulers.scheduling_ltx_euler_ancestral_rf.LTXEulerAncestralRFSchedulerOutput
