<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PNDMScheduler

`PNDMScheduler`, or pseudo numerical methods for diffusion models, uses more advanced ODE integration techniques like the Runge-Kutta and linear multi-step method. The original implementation can be found at [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181).

## PNDMScheduler
[[autodoc]] PNDMScheduler

## SchedulerOutput
[[autodoc]] schedulers.scheduling_utils.SchedulerOutput
