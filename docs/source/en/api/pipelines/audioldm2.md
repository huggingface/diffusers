<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AudioLDM 2

AudioLDM 2 was proposed in [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) 
by Haohe Liu et al. AudioLDM 2 takes a text prompt as input and predicts the corresponding audio. It can generate 
text-conditional sound effects, human speech and music.

Inspired by [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview), AudioLDM 2
is a text-to-audio _latent diffusion model (LDM)_ that learns continuous audio representations from text embeddings. Two 
text encoder models are used to compute the text embeddings from a prompt input: the text-branch of [CLAP](https://huggingface.co/docs/transformers/main/en/model_doc/clap)
and the encoder of [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5). These text embeddings 
are then projected to a shared embedding space by an [AudioLDM2ProjectionModel](https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2#diffusers.AudioLDM2ProjectionModel). 
A [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2) _language model (LM)_ is used to auto-regressively 
predict eight new embedding vectors, conditional on the projected CLAP and Flan-T5 embeddings. The generated embedding 
vectors and Flan-T5 text embeddings are used as cross-attention conditioning in the LDM. The [UNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2UNet2DConditionModel) 
of AudioLDM 2 is unique in the sense that it takes **two** cross-attention embeddings, as opposed to one cross-attention 
conditioning, as in most other LDMs.

The abstract of the paper is the following:

*Although audio generation shares commonalities across different types of audio, such as speech, music, and sound effects, designing models for each type requires careful consideration of specific objectives and biases that can significantly differ from those of other types. To bring us closer to a unified perspective of audio generation, this paper proposes a framework that utilizes the same learning method for speech, music, and sound effect generation. Our framework introduces a general representation of audio, called language of audio (LOA). Any audio can be translated into LOA based on AudioMAE, a self-supervised pre-trained representation learning model. In the generation process, we translate any modalities into LOA by using a GPT-2 model, and we perform self-supervised audio generation learning with a latent diffusion model conditioned on LOA. The proposed framework naturally brings advantages such as in-context learning abilities and reusable self-supervised pretrained AudioMAE and latent diffusion models. Experiments on the major benchmarks of text-to-audio, text-to-music, and text-to-speech demonstrate new state-of-the-art or competitive performance to previous approaches.*

This pipeline was contributed by [sanchit-gandhi](https://huggingface.co/sanchit-gandhi). The original codebase can be 
found at [haoheliu/audioldm2](https://github.com/haoheliu/audioldm2). 

## Tips

### Choosing a checkpoint

AudioLDM2 comes in three variants. Two of these checkpoints are applicable to the general task of text-to-audio 
generation. The third checkpoint is trained exclusively on text-to-music generation.

All checkpoints share the same model size for the text encoders and VAE. They differ in the size and depth of the UNet. 
See table below for details on the three checkpoints:

| Checkpoint                                                      | Task          | UNet Model Size | Total Model Size | Training Data / h |
|-----------------------------------------------------------------|---------------|-----------------|------------------|-------------------|
| [audioldm2](https://huggingface.co/cvssp/audioldm2)             | Text-to-audio | 350M            | 1.1B             | 1150k             |
| [audioldm2-large](https://huggingface.co/cvssp/audioldm2-large) | Text-to-audio | 750M            | 1.5B             | 1150k             |
| [audioldm2-music](https://huggingface.co/cvssp/audioldm2-music) | Text-to-music | 350M            | 1.1B             | 665k              |

### Constructing a prompt

* Descriptive prompt inputs work best: use adjectives to describe the sound (e.g. "high quality" or "clear") and make the prompt context specific (e.g. "water stream in a forest" instead of "stream").
* It's best to use general terms like "cat" or "dog" instead of specific names or abstract objects the model may not be familiar with.
* Using a **negative prompt** can significantly improve the quality of the generated waveform, by guiding the generation away from terms that correspond to poor quality audio. Try using a negative prompt of "Low quality." 

### Controlling inference

* The _quality_ of the predicted audio sample can be controlled by the `num_inference_steps` argument; higher steps give higher quality audio at the expense of slower inference.
* The _length_ of the predicted audio sample can be controlled by varying the `audio_length_in_s` argument.

### Evaluating generated waveforms:

* The quality of the generated waveforms can vary significantly based on the seed. Try generating with different seeds until you find a satisfactory generation
* Multiple waveforms can be generated in one go: set `num_waveforms_per_prompt` to a value greater than 1. Automatic scoring will be performed between the generated waveforms and prompt text, and the audios ranked from best to worst accordingly.

The following example demonstrates how to construct good music generation using the aforementioned tips: [example](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.example).

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## AudioLDM2Pipeline
[[autodoc]] AudioLDM2Pipeline
	- all
	- __call__

## AudioLDM2ProjectionModel
[[autodoc]] AudioLDM2ProjectionModel
	- forward

## AudioLDM2UNet2DConditionModel
[[autodoc]] AudioLDM2UNet2DConditionModel
	- forward

## AudioPipelineOutput
[[autodoc]] pipelines.AudioPipelineOutput