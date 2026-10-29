<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MiniMaxH3Scheduler

`MiniMaxH3Scheduler` is the rectified-flow Euler scheduler (`eta = 0`) with an exponential sigma shift used by [MiniMax-H3](https://huggingface.co/MiniMaxAI), `sigma' = s * sigma / (1 + (s - 1) * sigma)`.

The MiniMax-H3 pipelines register **two** of them, because video and audio latents step down two different schedules inside a single transformer call per step: `scheduler` carries the video schedule (`shift=12.0` in the released checkpoints) and `audio_scheduler` the audio one (`shift=3.0`).

## MiniMaxH3Scheduler
[[autodoc]] MiniMaxH3Scheduler

## MiniMaxH3SchedulerOutput
[[autodoc]] schedulers.scheduling_minimax_h3.MiniMaxH3SchedulerOutput
