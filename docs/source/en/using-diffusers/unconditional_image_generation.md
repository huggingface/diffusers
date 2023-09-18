<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Unconditional image generation

[[open-in-colab]]

Unconditional image generation is a relatively straightforward task. The model only generates images - without any additional context like text or an image - resembling the training data it was trained on.

The [`DiffusionPipeline`] is the easiest way to use a pre-trained diffusion system for inference.

Start by creating an instance of [`DiffusionPipeline`] and specify which pipeline checkpoint you would like to download.
You can use any of the 🧨 Diffusers [checkpoints](https://huggingface.co/models?library=diffusers&sort=downloads) from the Hub (the checkpoint you'll use generates images of butterflies).

<Tip>

💡 Want to train your own unconditional image generation model? Take a look at the training [guide](training/unconditional_training) to learn how to generate your own images.

</Tip>

In this guide, you'll use [`DiffusionPipeline`] for unconditional image generation with [DDPM](https://arxiv.org/abs/2006.11239):

```python
>>> from diffusers import DiffusionPipeline

>>> generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128", use_safetensors=True)
```

The [`DiffusionPipeline`] downloads and caches all modeling, tokenization, and scheduling components. 
Because the model consists of roughly 1.4 billion parameters, we strongly recommend running it on a GPU.
You can move the generator object to a GPU, just like you would in PyTorch:

```python
>>> generator.to("cuda")
```

Now you can use the `generator` to generate an image:

```python
>>> image = generator().images[0]
```

The output is by default wrapped into a [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) object.

You can save the image by calling:

```python
>>> image.save("generated_image.png")
```

Try out the Spaces below, and feel free to play around with the inference steps parameter to see how it affects the image quality!

<iframe
	src="https://stevhliu-ddpm-butterflies-128.hf.space"
	frameborder="0"
	width="850"
	height="500"
></iframe>


