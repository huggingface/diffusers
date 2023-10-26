<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MusicLDM

MusicLDM was proposed in [MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies](https://huggingface.co/papers/2308.01546) by Ke Chen, Yusong Wu, Haohe Liu, Marianna Nezhurina, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
MusicLDM takes a text prompt as input and predicts the corresponding music sample. 

Inspired by [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) and [AudioLDM](https://huggingface.co/docs/diffusers/api/pipelines/audioldm/overview),
MusicLDM is a text-to-music _latent diffusion model (LDM)_ that learns continuous audio representations from [CLAP](https://huggingface.co/docs/transformers/main/model_doc/clap)
latents.

MusicLDM is trained on a corpus of 466 hours of music data. Beat-synchronous data augmentation strategies are applied to 
the music samples, both in the time domain and in the latent space. Using beat-synchronous data augmentation strategies 
encourages the model to interpolate between the training samples, but stay within the domain of the training data. The 
result is generated music that is more diverse while staying faithful to the corresponding style.

The abstract of the paper is the following:

*In this paper, we present MusicLDM, a state-of-the-art text-to-music model that adapts Stable Diffusion and AudioLDM architectures to the music domain. We achieve this by retraining the contrastive language-audio pretraining model (CLAP) and the Hifi-GAN vocoder, as components of MusicLDM, on a collection of music data samples. Then, we leverage a beat tracking model and propose two different mixup strategies for data augmentation: beat-synchronous audio mixup and beat-synchronous latent mixup, to encourage the model to generate music more diverse while still staying faithful to the corresponding style.*

This pipeline was contributed by [sanchit-gandhi](https://huggingface.co/sanchit-gandhi).

## Tips

When constructing a prompt, keep in mind:

* Descriptive prompt inputs work best; use adjectives to describe the sound (for example, "high quality" or "clear") and make the prompt context specific where possible (e.g. "melodic techno with a fast beat and synths" works better than "techno").
* Using a *negative prompt* can significantly improve the quality of the generated audio. Try using a negative prompt of "low quality, average quality".

During inference:

* The _quality_ of the generated audio sample can be controlled by the `num_inference_steps` argument; higher steps give higher quality audio at the expense of slower inference.
* Multiple waveforms can be generated in one go: set `num_waveforms_per_prompt` to a value greater than 1 to enable. Automatic scoring will be performed between the generated waveforms and prompt text, and the audios ranked from best to worst accordingly.
* The _length_ of the generated audio sample can be controlled by varying the `audio_length_in_s` argument.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## MusicLDMPipeline
[[autodoc]] MusicLDMPipeline
	- all
	- __call__