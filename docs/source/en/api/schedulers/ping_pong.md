<!--Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PingPongScheduler

The `PingPongScheduler` implements the stochastic re-noise denoising schedule used in
[Stable Audio 3](https://stability.ai/news/stable-audio-3).

At each step the scheduler:

1. Computes the predicted clean sample `x₀` from the model's velocity prediction.
2. **Re-noises** back to the next noise level: `x_{t+1} = (1 - t_next) * x₀ + t_next * ε`, where `ε` is fresh
   Gaussian noise.

This "ping-pong" of denoise → re-noise encourages diversity while remaining efficient. The default schedule uses
**8 steps** with a log-SNR-uniform noise schedule over λ ∈ [−6.2, 2.0].

## PingPongScheduler

[[autodoc]] PingPongScheduler
	- all
	- step

## PingPongSchedulerOutput

[[autodoc]] PingPongSchedulerOutput
