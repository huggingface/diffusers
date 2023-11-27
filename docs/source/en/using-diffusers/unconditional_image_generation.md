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

Unconditional image generation generates images that look like a random sample from the training data the model was trained on because the denoising process is not guided by any additional context like text or image.

To get started, use the [`DiffusionPipeline`] to load the [anton-l/ddpm-butterflies-128](https://huggingface.co/anton-l/ddpm-butterflies-128) checkpoint to generate images of butterflies. The [`DiffusionPipeline`] downloads and caches all the model components required to generate an image.

```py
from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128").to("cuda")
image = generator().images[0]
image
```

<Tip>

Want to generate images of something else? Take a look at the training [guide](../training/unconditional_training) to learn how to train a model to generate your own images.

</Tip>

The output image is a [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) object that can be saved:

```py
image.save("generated_image.png")
```

You can also try experimenting with the `num_inference_steps` parameter, which controls the number of denoising steps. More denoising steps typically produce higher quality images, but it'll take longer to generate. Feel free to play around with this parameter to see how it affects the image quality.

```py
image = generator(num_inference_steps=100).images[0]
image
```

Try out the Space below to generate an image of a butterfly!

<iframe
	src="https://stevhliu-unconditional-image-generation.hf.space"
	frameborder="0"
	width="850"
	height="500"
></iframe>
