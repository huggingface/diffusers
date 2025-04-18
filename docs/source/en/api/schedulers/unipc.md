<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UniPCMultistepScheduler

`UniPCMultistepScheduler` is a training-free framework designed for fast sampling of diffusion models. It was introduced in [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://huggingface.co/papers/2302.04867) by Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu.

It consists of a corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders.
UniPC is by design model-agnostic, supporting pixel-space/latent-space DPMs on unconditional/conditional sampling. It can also be applied to both noise prediction and data prediction models. The corrector UniC can be also applied after any off-the-shelf solvers to increase the order of accuracy.

The abstract from the paper is:

*Diffusion probabilistic models (DPMs) have demonstrated a very promising ability in high-resolution image synthesis. However, sampling from a pre-trained DPM is time-consuming due to the multiple evaluations of the denoising network, making it more and more important to accelerate the sampling of DPMs. Despite recent progress in designing fast samplers, existing methods still cannot generate satisfying images in many applications where fewer steps (e.g., <10) are favored. In this paper, we develop a unified corrector (UniC) that can be applied after any existing DPM sampler to increase the order of accuracy without extra model evaluations, and derive a unified predictor (UniP) that supports arbitrary order as a byproduct. Combining UniP and UniC, we propose a unified predictor-corrector framework called UniPC for the fast sampling of DPMs, which has a unified analytical form for any order and can significantly improve the sampling quality over previous methods, especially in extremely few steps. We evaluate our methods through extensive experiments including both unconditional and conditional sampling using pixel-space and latent-space DPMs. Our UniPC can achieve 3.87 FID on CIFAR10 (unconditional) and 7.51 FID on ImageNet 256×256 (conditional) with only 10 function evaluations. Code is available at [this https URL](https://github.com/wl-zhao/UniPC).*

## Tips

It is recommended to set `solver_order` to 2 for guide sampling, and `solver_order=3` for unconditional sampling.

Dynamic thresholding from [Imagen](https://huggingface.co/papers/2205.11487) is supported, and for pixel-space
diffusion models, you can set both `predict_x0=True` and `thresholding=True` to use dynamic thresholding. This thresholding method is unsuitable for latent-space diffusion models such as Stable Diffusion.

## UniPCMultistepScheduler
[[autodoc]] UniPCMultistepScheduler

## SchedulerOutput
[[autodoc]] schedulers.scheduling_utils.SchedulerOutput
