<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PiflowScheduler

`PiflowScheduler` is the few-step scheduler of the distilled Kandinsky 6 checkpoints, both the
text/image-to-video-and-audio model and the video super-resolution model. It implements
[π-Flow](https://huggingface.co/papers/2510.14974): the transformer predicts `n_grid` denoised estimates per latent
channel at a small number of grid points, and the scheduler integrates a network-free policy between them.

The reference implementation can be found at [Lakonik/LakonLab](https://github.com/Lakonik/LakonLab).

## PiflowScheduler

[[autodoc]] PiflowScheduler
  - set_timesteps
  - step

## PiflowSchedulerOutput

[[autodoc]] schedulers.scheduling_piflow.PiflowSchedulerOutput
