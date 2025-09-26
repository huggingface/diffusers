<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# VDMScheduler

[Variational Diffusion Models](https://arxiv.org/abs/2107.00630) (VDM) by Diederik P. Kingma, Tim Salimans, Ben 
Poole and Jonathan Ho introduces a family of diffusion-based generative models that achieve state-of-the-art 
log-likelihoods on standard image density estimation benchmarks by formulating diffusion as a continuous-time problem 
in terms of the signal-to-noise ratio.

The abstract from the paper is:

*Diffusion-based generative models have demonstrated a capacity for perceptually impressive synthesis, but can they 
also be great likelihood-based models? We answer this in the affirmative, and introduce a family of diffusion-based 
generative models that obtain state-of-the-art likelihoods on standard image density estimation benchmarks. Unlike 
other diffusion-based models, our method allows for efficient optimization of the noise schedule jointly with the 
rest of the model. We show that the variational lower bound (VLB) simplifies to a remarkably short expression in terms
of the signal-to-noise ratio of the diffused data, thereby improving our theoretical understanding of this model class.
Using this insight, we prove an equivalence between several models proposed in the literature. In addition, we show that
the continuous-time VLB is invariant to the noise schedule, except for the signal-to-noise ratio at its endpoints. This
enables us to learn a noise schedule that minimizes the variance of the resulting VLB estimator, leading to faster
optimization. Combining these advances with architectural improvements, we obtain state-of-the-art likelihoods on image
density estimation benchmarks, outperforming autoregressive models that have dominated these benchmarks for many years,
with often significantly faster optimization. In addition, we show how to use the model as part of a bits-back
compression scheme, and demonstrate lossless compression rates close to the theoretical optimum. Code is available at
[this https URL](https://github.com/google-research/vdm).*

## VDMScheduler
[[autodoc]] VDMScheduler

## VDMSchedulerOutput
[[autodoc]] schedulers.scheduling_vdm.VDMSchedulerOutput
