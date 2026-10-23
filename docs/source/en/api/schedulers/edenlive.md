<!-- Copyright 2026 The VAST-AI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EdenLiveFlowMatchEulerDiscreteScheduler

[`EdenLiveFlowMatchEulerDiscreteScheduler`] implements deterministic flow Euler sampling for EdenLive 14B
bidirectional generation and stochastic sampling for the distilled causal checkpoints. The 5B bidirectional model
uses [`UniPCMultistepScheduler`].

Load the `scheduler` subfolder from the matching checkpoint repository. For causal generation, use
`VAST-AI/EdenLive-5B-Causal-Diffusers` or `VAST-AI/EdenLive-14B-Causal-Diffusers`; the released causal schedules have
four fixed steps per chunk. See the [pipeline guide](../pipelines/edenlive#causal-generation) for loading the pipeline.

## EdenLiveFlowMatchEulerDiscreteScheduler

[[autodoc]] EdenLiveFlowMatchEulerDiscreteScheduler
    - set_timesteps
    - step
