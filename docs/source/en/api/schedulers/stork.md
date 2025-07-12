<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# STORKScheduler
`STORKScheduler` is the sampling method from the paper [STORK: Improving the Fidelity of Mid-NFE Sampling for Diffusion and Flow Matching Models](https://arxiv.org/abs/2505.24210) by [Zheng Tan](https://zt220501.github.io/), [Weizhen Wang](https://weizhenwang-1210.github.io/), [Andrea L. Bertozzi](https://www.math.ucla.edu/~bertozzi/), and [Ernest K. Ryu](https://ernestryu.com/). It was motivated by stabilized Runge--Kutta methods, with Taylor expansion adaptation for diffusion and flow matching models.

--------------------

## STORKScheduler
[[autodoc]] STORKScheduler

## SchedulerOutput
[[autodoc]] schedulers.scheduling_utils.SchedulerOutput

