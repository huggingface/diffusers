<!-- Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# FlowMapEulerDiscreteScheduler

`FlowMapEulerDiscreteScheduler` is an Euler-style sampler designed for flow-map-distilled diffusion
models. Flow-map models learn arbitrary-interval transitions $\mathbf{z}_t \to \mathbf{z}_r$ rather than
the fixed $\mathbf{z}_t \to \mathbf{z}_0$ mapping of consistency models. Both endpoints of the step are
caller-provided, which is what enables any-step sampling: a single distilled checkpoint can be evaluated at
1, 2, 4, 8, 16... NFE without retraining.

The scheduler was introduced in
[AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://huggingface.co/papers/2605.13724)
and ships with the `AnyFlowPipeline` and `AnyFlowFARPipeline` integrations, but it is not
AnyFlow-specific — any flow-map-distilled checkpoint can use it.

## FlowMapEulerDiscreteScheduler

[[autodoc]] FlowMapEulerDiscreteScheduler
