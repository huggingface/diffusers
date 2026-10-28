<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PikaH3FlexScheduler

`PikaH3FlexScheduler` is the few-step re-noising sampler of [Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Early-Access). At every knot of a fixed ladder the transformer predicts the clean sample, `x0 = x_t + sigma * v`, and the next state re-noises that prediction with *fresh* Gaussian noise to the next knot, `x_next = (1 - sigma_next) * x0 + sigma_next * noise`; the last evaluation returns `x0`. The knots are the released `[1000, 750, 500, 250, 100, 40, 0]` base grid pushed through the exponential shift `sigma' = s * sigma / (1 + (s - 1) * sigma)`, six evaluations in all, and they are part of the adapter's contract rather than a `num_inference_steps` argument.

The Pika H3 Flex pipeline registers **two** of them, as MiniMax-H3 does, because video and audio latents step down two different ladders inside a single transformer call per evaluation: `scheduler` carries the video ladder (`shift=12.0`) and `audio_scheduler` the audio one (`shift=3.0`).

## PikaH3FlexScheduler
[[autodoc]] PikaH3FlexScheduler

## PikaH3FlexSchedulerOutput
[[autodoc]] schedulers.scheduling_pika_h3_flex.PikaH3FlexSchedulerOutput
