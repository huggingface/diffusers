<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# TCDScheduler

[Trajectory Consistency Distillation](https://huggingface.co/papers/2402.19159) by Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao and Tat-Jen Cham introduced a Strategic Stochastic Sampling (Algorithm 4) that is capable of generating good samples in a small number of steps. Distinguishing it as an advanced iteration of the multistep scheduler (Algorithm 1) in the [Consistency Models](https://huggingface.co/papers/2303.01469), Strategic Stochastic Sampling specifically tailored for the trajectory consistency function.

The abstract from the paper is:

*Latent Consistency Model (LCM) extends the Consistency Model to the latent space and leverages the guided consistency distillation technique to achieve impressive performance in accelerating text-to-image synthesis. However, we observed that LCM struggles to generate images with both clarity and detailed intricacy. To address this limitation, we initially delve into and elucidate the underlying causes. Our investigation identifies that the primary issue stems from errors in three distinct areas. Consequently, we introduce Trajectory Consistency Distillation (TCD), which encompasses trajectory consistency function and strategic stochastic sampling. The trajectory consistency function diminishes the distillation errors by broadening the scope of the self-consistency boundary condition and endowing the TCD with the ability to accurately trace the entire trajectory of the Probability Flow ODE. Additionally, strategic stochastic sampling is specifically designed to circumvent the accumulated errors inherent in multi-step consistency sampling, which is meticulously tailored to complement the TCD model. Experiments demonstrate that TCD not only significantly enhances image quality at low NFEs but also yields more detailed results compared to the teacher model at high NFEs.*

The original codebase can be found at [jabir-zheng/TCD](https://github.com/jabir-zheng/TCD).

## TCDScheduler
[[autodoc]] TCDScheduler


## TCDSchedulerOutput
[[autodoc]] schedulers.scheduling_tcd.TCDSchedulerOutput

