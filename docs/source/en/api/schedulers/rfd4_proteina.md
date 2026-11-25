<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# RFD4ProteinaScheduler

`RFD4ProteinaScheduler` is the temporal-score-rescaled flow sampler of RFD4-Proteína for one modality. The pipeline uses
two instances, one for the backbone (`bb_ca`, coordinates in nanometers) and one for the per-token local latents
(`local_latents`), each with its own time schedule, noise schedule and rescaling. See [`RFD4ProteinaPipeline`] for the
sampling loop.

## RFD4ProteinaScheduler

[[autodoc]] RFD4ProteinaScheduler
  - set_timesteps
  - add_noise
  - step

## RFD4ProteinaSchedulerOutput

[[autodoc]] schedulers.scheduling_rfd4_proteina.RFD4ProteinaSchedulerOutput
