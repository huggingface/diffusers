<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoencoderOobleck

The Oobleck variational autoencoder (VAE) model with KL loss was introduced in [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) and [Stable Audio Open](https://huggingface.co/papers/2407.14358) by Stability AI. The model is used in 🤗 Diffusers to encode audio waveforms into latents and to decode latent representations into audio waveforms.

The abstract from the paper is:

*Open generative models are vitally important for the community, allowing for fine-tunes and serving as baselines when presenting new models. However, most current text-to-audio models are private and not accessible for artists and researchers to build upon. Here we describe the architecture and training process of a new open-weights text-to-audio model trained with Creative Commons data. Our evaluation shows that the model's performance is competitive with the state-of-the-art across various metrics. Notably, the reported FDopenl3 results (measuring the realism of the generations) showcase its potential for high-quality stereo sound synthesis at 44.1kHz.*

## AutoencoderOobleck

[[autodoc]] AutoencoderOobleck
    - decode
    - encode
    - all

## OobleckDecoderOutput

[[autodoc]] models.autoencoders.autoencoder_oobleck.OobleckDecoderOutput

## OobleckDecoderOutput

[[autodoc]] models.autoencoders.autoencoder_oobleck.OobleckDecoderOutput

## AutoencoderOobleckOutput

[[autodoc]] models.autoencoders.autoencoder_oobleck.AutoencoderOobleckOutput
