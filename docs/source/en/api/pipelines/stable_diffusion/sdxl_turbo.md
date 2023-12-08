<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SDXL Turbo

Stable Diffusion XL (SDXL) Turbo was proposed in [Adversarial Diffusion Distillation](https://stability.ai/research/adversarial-diffusion-distillation) by Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach.

The abstract from the paper is:

*We introduce Adversarial Diffusion Distillation (ADD), a novel training approach that efficiently samples large-scale foundational image diffusion models in just 1–4 steps while maintaining high image quality. We use score distillation to leverage large-scale off-the-shelf image diffusion models as a teacher signal in combination with an adversarial loss to ensure high image fidelity even in the low-step regime of one or two sampling steps. Our analyses show that our model clearly outperforms existing few-step methods (GANs,Latent Consistency Models) in a single step and reaches the performance of state-of-the-art diffusion models (SDXL) in only four steps. ADD is the first method to unlock single-step, real-time image synthesis with foundation models.*

## Tips

- SDXL Turbo uses the exact same architecture as [SDXL](./stable_diffusion_xl), which means it also has the same API. Please refer to the [SDXL](./stable_diffusion_xl) API reference for more details.
- SDXL Turbo should disable guidance scale by setting `guidance_scale=0.0`
- SDXL Turbo should use `timestep_spacing='trailing'` for the scheduler and use between 1 and 4 steps.
- SDXL Turbo has been trained to generate images of size 512x512.
- SDXL Turbo is open-access, but not open-source meaning that one might have to buy a model license in order to use it for commercial applications. Make sure to read the [official model card](https://huggingface.co/stabilityai/sdxl-turbo) to learn more.

<Tip>

To learn how to use SDXL Turbo for various tasks, how to optimize performance, and other usage examples, take a look at the [SDXL Turbo](../../../using-diffusers/sdxl_turbo) guide.

Check out the [Stability AI](https://huggingface.co/stabilityai) Hub organization for the official base and refiner model checkpoints!

</Tip>
