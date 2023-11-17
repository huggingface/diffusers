<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AudioLDM

AudioLDM was proposed in [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://huggingface.co/papers/2301.12503) by Haohe Liu et al. Inspired by [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview), AudioLDM
is a text-to-audio _latent diffusion model (LDM)_ that learns continuous audio representations from [CLAP](https://huggingface.co/docs/transformers/main/model_doc/clap)
latents. AudioLDM takes a text prompt as input and predicts the corresponding audio. It can generate text-conditional
sound effects, human speech and music.

The abstract from the paper is:

*Text-to-audio (TTA) system has recently gained attention for its ability to synthesize general audio based on text descriptions. However, previous studies in TTA have limited generation quality with high computational costs. In this study, we propose AudioLDM, a TTA system that is built on a latent space to learn the continuous audio representations from contrastive language-audio pretraining (CLAP) latents. The pretrained CLAP models enable us to train LDMs with audio embedding while providing text embedding as a condition during sampling. By learning the latent representations of audio signals and their compositions without modeling the cross-modal relationship, AudioLDM is advantageous in both generation quality and computational efficiency. Trained on AudioCaps with a single GPU, AudioLDM achieves state-of-the-art TTA performance measured by both objective and subjective metrics (e.g., frechet distance). Moreover, AudioLDM is the first TTA system that enables various text-guided audio manipulations (e.g., style transfer) in a zero-shot fashion. Our implementation and demos are available at [this https URL](https://audioldm.github.io/).*

The original codebase can be found at [haoheliu/AudioLDM](https://github.com/haoheliu/AudioLDM).

## Tips

When constructing a prompt, keep in mind:

* Descriptive prompt inputs work best; you can use adjectives to describe the sound (for example, "high quality" or "clear") and make the prompt context specific (for example, "water stream in a forest" instead of "stream").
* It's best to use general terms like "cat" or "dog" instead of specific names or abstract objects the model may not be familiar with.

During inference:

* The _quality_ of the predicted audio sample can be controlled by the `num_inference_steps` argument; higher steps give higher quality audio at the expense of slower inference.
* The _length_ of the predicted audio sample can be controlled by varying the `audio_length_in_s` argument.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## AudioLDMPipeline
[[autodoc]] AudioLDMPipeline
	- all
	- __call__

## AudioPipelineOutput
[[autodoc]] pipelines.AudioPipelineOutput
