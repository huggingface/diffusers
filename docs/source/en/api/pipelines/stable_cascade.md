<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Cascade

This model is built upon the [Würstchen](https://openreview.net/forum?id=gU58d5QeGv) architecture and its main 
difference to other models like Stable Diffusion is that it is working at a much smaller latent space. Why is this 
important? The smaller the latent space, the **faster** you can run inference and the **cheaper** the training becomes. 
How small is the latent space? Stable Diffusion uses a compression factor of 8, resulting in a 1024x1024 image being 
encoded to 128x128. Stable Cascade achieves a compression factor of 42, meaning that it is possible to encode a 
1024x1024 image to 24x24, while maintaining crisp reconstructions. The text-conditional model is then trained in the 
highly compressed latent space. Previous versions of this architecture, achieved a 16x cost reduction over Stable 
Diffusion 1.5.

Therefore, this kind of model is well suited for usages where efficiency is important. Furthermore, all known extensions
like finetuning, LoRA, ControlNet, IP-Adapter, LCM etc. are possible with this method as well.

The original codebase can be found at [Stability-AI/StableCascade](https://github.com/Stability-AI/StableCascade).

## Model Overview
Stable Cascade consists of three models: Stage A, Stage B and Stage C, representing a cascade to generate images,
hence the name "Stable Cascade".

Stage A & B are used to compress images, similar to what the job of the VAE is in Stable Diffusion. 
However, with this setup, a much higher compression of images can be achieved. While the Stable Diffusion models use a 
spatial compression factor of 8, encoding an image with resolution of 1024 x 1024 to 128 x 128, Stable Cascade achieves 
a compression factor of 42. This encodes a 1024 x 1024 image to 24 x 24, while being able to accurately decode the 
image. This comes with the great benefit of cheaper training and inference. Furthermore, Stage C is responsible 
for generating the small 24 x 24 latents given a text prompt.

## Uses

### Direct Use

The model is intended for research purposes for now. Possible research areas and tasks include

- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, 
and therefore using the model to generate such content is out-of-scope for the abilities of this model.
The model should not be used in any way that violates Stability AI's [Acceptable Use Policy](https://stability.ai/use-policy).

## Limitations and Bias

### Limitations
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.


## StableCascadeCombinedPipeline

[[autodoc]] StableCascadeCombinedPipeline
	- all
	- __call__

## StableCascadePriorPipeline

[[autodoc]] StableCascadePriorPipeline
	- all
	- __call__

## StableCascadePriorPipelineOutput

[[autodoc]] pipelines.stable_cascade.pipeline_stable_cascade_prior.StableCascadePriorPipelineOutput

## StableCascadeDecoderPipeline

[[autodoc]] StableCascadeDecoderPipeline
	- all
	- __call__

