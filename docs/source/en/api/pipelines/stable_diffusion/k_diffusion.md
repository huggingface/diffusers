<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

> [!WARNING]
> This pipeline is deprecated but it can still be used. However, we won't test the pipeline anymore and won't accept any changes to it. If you run into any issues, reinstall the last Diffusers version that supported this model.

# K-Diffusion

[k-diffusion](https://github.com/crowsonkb/k-diffusion) is a popular library created by [Katherine Crowson](https://github.com/crowsonkb/). We provide `StableDiffusionKDiffusionPipeline` and `StableDiffusionXLKDiffusionPipeline` that allow you to run Stable DIffusion with samplers from k-diffusion.

Note that most the samplers from k-diffusion are implemented in Diffusers and we recommend using existing schedulers. You can find a mapping between k-diffusion samplers and schedulers in Diffusers [here](https://huggingface.co/docs/diffusers/api/schedulers/overview)


## StableDiffusionKDiffusionPipeline

[[autodoc]] StableDiffusionKDiffusionPipeline


## StableDiffusionXLKDiffusionPipeline

[[autodoc]] StableDiffusionXLKDiffusionPipeline