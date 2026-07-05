<!--Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# StableAudio3EulerScheduler

The `StableAudio3EulerScheduler` implements the deterministic rectified-flow Euler sampler used by the
**base** [Stable Audio 3](https://stability.ai/news/stable-audio-3) model.

Unlike [`PingPongScheduler`] (which resamples fresh noise each step and is used by the *distilled* model), this
scheduler performs a deterministic first-order flow-matching update:

`x_{t+1} = x_t + (σ_{i+1} − σ_i) · v(x_t, t)`

where `v` is the model's velocity prediction. The base model is not distilled, so it needs many more steps than
ping-pong — typically `num_inference_steps=100`. The noise schedule is identical to [`PingPongScheduler`]:
log-SNR-uniform over λ ∈ [−6.2, 2.0], mapped to the flow-matching `t` variable via `t = sigmoid(−λ)`.

## StableAudio3EulerScheduler

[[autodoc]] StableAudio3EulerScheduler
	- all
	- step

## StableAudio3EulerSchedulerOutput

[[autodoc]] StableAudio3EulerSchedulerOutput
