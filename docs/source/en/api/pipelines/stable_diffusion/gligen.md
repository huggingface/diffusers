<!--Copyright 2023 The GLIGEN Authors and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# GLIGEN: Grounded Text-to-image

The Stable Diffusion model was created by researchers and engineers from [University of Wisconsin-Madison, Columbia University, and Microsoft](https://github.com/gligen/GLIGEN). The [`StableDiffusionGLIGENPipeline`] is capable of generating photorealistic images conditioned on grounding inputs. Along with text and bounding boxes, if input images is given, this pipeline can insert objects described by text at the region defined by bounding boxes. Otherwise, this pipeline can generated an image described by caption/prompt and inserting objects described by text at the region defined by bounding boxes. It's trained on COCO2014D and COCO2014CD dataset. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on grounding inputs.

The abstract from the paper is:

*Large-scale text-to-image diffusion models have made amazing advances. However, the status quo is to use text input alone, which can impede controllability. In this work, we propose GLIGEN, Grounded-Language-to-Image Generation, a novel approach that builds upon and extends the functionality of existing pre-trained text-to-image diffusion models by enabling them to also be conditioned on grounding inputs. To preserve the vast concept knowledge of the pre-trained model, we freeze all of its weights and inject the grounding information into new trainable layers via a gated mechanism. Our model achieves open-world grounded text2img generation with caption and bounding box condition inputs, and the grounding ability generalizes well to novel spatial configurations and concepts. GLIGEN’s zeroshot performance on COCO and LVIS outperforms existing supervised layout-to-image baselines by a large margin.*

<Tip>

If you're interested in using one of the official checkpoints for a task, explore the [gligen](https://huggingface.co/gligen) Hub organizations!

</Tip>

## StableDiffusionGLIGENPipeline

[[autodoc]] StableDiffusionGLIGENPipeline
	- all
	- __call__
	- enable_vae_slicing
	- disable_vae_slicing
	- enable_vae_tiling
	- disable_vae_tiling
	- enable_model_cpu_offload
	- prepare_latents
	- enable_fuser
