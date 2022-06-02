<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Denoising Diffusion Probabilistic Models (DDPM)

## Overview

DDPM was proposed in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by *Jonathan Ho, Ajay Jain, Pieter Abbeel*.

The abstract from the paper is the following:

*We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at this https URL*

Tips:

- ...
- ...

This model was contributed by [???](https://huggingface.co/???). The original code can be found [here](https://github.com/hojonathanho/diffusion).

![ddpm](https://user-images.githubusercontent.com/23423619/171627620-e3406711-1e20-4a99-8e30-ec5a86a465be.png)
