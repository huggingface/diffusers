<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Denoising Diffusion Implicit Models (DDIM)

## Overview

[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM) by Jiaming Song, Chenlin Meng and Stefano Ermon.

The abstract of the paper is the following:

*Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, 
yet they require simulating a Markov chain for many steps to produce a sample. 
To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models
with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process. 
We construct a class of non-Markovian diffusion processes that lead to the same training objective, but whose reverse process can be much faster to sample from.
We empirically demonstrate that DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs, allow us to trade off 
computation for sample quality, and can perform semantically meaningful image interpolation directly in the latent space.*

The original codebase of this paper can be found here: [ermongroup/ddim](https://github.com/ermongroup/ddim).
For questions, feel free to contact the author on [tsong.me](https://tsong.me/).

### Experimental: "Common Diffusion Noise Schedules and Sample Steps are Flawed": 

The paper **[Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)** 
claims that a mismatch between the training and inference settings leads to suboptimal inference generation results for Stable Diffusion.

The abstract reads as follows:

*We discover that common diffusion noise schedules do not enforce the last timestep to have zero signal-to-noise ratio (SNR),
and some implementations of diffusion samplers do not start from the last timestep.
Such designs are flawed and do not reflect the fact that the model is given pure Gaussian noise at inference, creating a discrepancy between training and inference.
We show that the flawed design causes real problems in existing implementations. 
In Stable Diffusion, it severely limits the model to only generate images with medium brightness and 
prevents it from generating very bright and dark samples. We propose a few simple fixes: 
- (1) rescale the noise schedule to enforce zero terminal SNR; 
- (2) train the model with v prediction; 
- (3) change the sampler to always start from the last timestep; 
- (4) rescale classifier-free guidance to prevent over-exposure. 
These simple changes ensure the diffusion process is congruent between training and inference and 
allow the model to generate samples more faithful to the original data distribution.*

You can apply all of these changes in `diffusers` when using [`DDIMScheduler`]:
- (1) rescale the noise schedule to enforce zero terminal SNR; 
```py
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True)
```
- (2) train the model with v prediction; 
Continue fine-tuning a checkpoint with [`train_text_to_image.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) or [`train_text_to_image_lora.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
and `--prediction_type="v_prediction"`.
- (3) change the sampler to always start from the last timestep; 
```py
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
```
- (4) rescale classifier-free guidance to prevent over-exposure. 
```py
pipe(..., guidance_rescale=0.7)
```

An example is to use [this checkpoint](https://huggingface.co/ptx0/pseudo-journey-v2) 
which has been fine-tuned using the `"v_prediction"`.

The checkpoint can then be run in inference as follows:

```py
from diffusers import DiffusionPipeline, DDIMScheduler

pipe = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipe.to("cuda")

prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"
image = pipeline(prompt, guidance_rescale=0.7).images[0]
```

## DDIMScheduler
[[autodoc]] DDIMScheduler
