<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoencoderKL

The variational autoencoder (VAE) model with KL loss was introduced in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114v11) by Diederik P. Kingma and Max Welling. The model is used in 🤗 Diffusers to encode images into latents and to decode latent representations into images.

The abstract from the paper is:

*How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions are two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.*

## Loading from the original format

By default the [`AutoencoderKL`] should be loaded with [`~ModelMixin.from_pretrained`], but it can also be loaded
from the original format using [`FromOriginalModelMixin.from_single_file`] as follows:

```py
from diffusers import AutoencoderKL

url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
model = AutoencoderKL.from_single_file(url)
```

## AutoencoderKL

[[autodoc]] AutoencoderKL
    - decode
    - encode
    - all

## AutoencoderKLOutput

[[autodoc]] models.autoencoders.autoencoder_kl.AutoencoderKLOutput

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput

## FlaxAutoencoderKL

[[autodoc]] FlaxAutoencoderKL

## FlaxAutoencoderKLOutput

[[autodoc]] models.vae_flax.FlaxAutoencoderKLOutput

## FlaxDecoderOutput

[[autodoc]] models.vae_flax.FlaxDecoderOutput
