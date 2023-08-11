<!--Copyright 2023 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 2.1

Kandinsky 2.1 is created by [Arseniy Shakhmatov](https://github.com/cene555), [Anton Razzhigaev](https://github.com/razzant), [Aleksandr Nikolich](https://github.com/AlexWortega), [Igor Pavlov](https://github.com/boomb0om), [Andrey Kuznetsov](https://github.com/kuznetsoffandrey) and [Denis Dimitrov](https://github.com/denndimitrov).

The description from it's GitHub page is:

*Kandinsky 2.1 inherits best practicies from Dall-E 2 and Latent diffusion, while introducing some new ideas. As text and image encoder it uses CLIP model and diffusion image prior (mapping) between latent spaces of CLIP modalities. This approach increases the visual performance of the model and unveils new horizons in blending images and text-guided image manipulation.*

The original codebase can be found at [ai-forever/Kandinsky-2](https://github.com/ai-forever/Kandinsky-2).

<Tip>

Check out the [Kandinsky Community](https://huggingface.co/kandinsky-community) organization on the Hub for the official model checkpoints for tasks like text-to-image, image-to-image, and inpainting.

</Tip>

## KandinskyPriorPipeline

[[autodoc]] KandinskyPriorPipeline
	- all
	- __call__
	- interpolate
	
## KandinskyPipeline

[[autodoc]] KandinskyPipeline
	- all
	- __call__

## KandinskyCombinedPipeline

[[autodoc]] KandinskyCombinedPipeline
	- all
	- __call__

## KandinskyImg2ImgPipeline

[[autodoc]] KandinskyImg2ImgPipeline
	- all
	- __call__

## KandinskyImg2ImgCombinedPipeline

[[autodoc]] KandinskyImg2ImgCombinedPipeline
	- all
	- __call__

## KandinskyInpaintPipeline

[[autodoc]] KandinskyInpaintPipeline
	- all
	- __call__

## KandinskyInpaintCombinedPipeline

[[autodoc]] KandinskyInpaintCombinedPipeline
	- all
	- __call__
