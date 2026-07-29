<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# FastLMSDiscreteScheduler

`FastLMSDiscreteScheduler` is a linear multistep scheduler for discrete beta schedules. It matches [`LMSDiscreteScheduler`]
numerically, but precomputes LMS coefficients in `set_timesteps` with a NumPy trapezoidal integral so the default sampling
path does not require SciPy.

## FastLMSDiscreteScheduler
[[autodoc]] FastLMSDiscreteScheduler

## FastLMSDiscreteSchedulerOutput
[[autodoc]] schedulers.scheduling_fast_lms_discrete.FastLMSDiscreteSchedulerOutput
