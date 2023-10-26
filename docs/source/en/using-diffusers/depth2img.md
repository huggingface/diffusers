<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text-guided depth-to-image generation

[[open-in-colab]]

The [`StableDiffusionDepth2ImgPipeline`] lets you pass a text prompt and an initial image to condition the generation of new images. In addition, you can also pass a `depth_map` to preserve the image structure. If no `depth_map` is provided, the pipeline automatically predicts the depth via an integrated [depth-estimation model](https://github.com/isl-org/MiDaS).

Start by creating an instance of the [`StableDiffusionDepth2ImgPipeline`]:

```python
import torch
import requests
from PIL import Image

from diffusers import StableDiffusionDepth2ImgPipeline

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
```

Now pass your prompt to the pipeline. You can also pass a `negative_prompt` to prevent certain words from guiding how an image is generated:

```python
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)
prompt = "two tigers"
n_prompt = "bad, deformed, ugly, bad anatomy"
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_prompt, strength=0.7).images[0]
image
```

| Input                                                                           | Output                                                                                                                                |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/coco-cats.png" width="500"/> | <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/depth2img-tigers.png" width="500"/> |

Play around with the Spaces below and see if you notice a difference between generated images with and without a depth map!

<iframe
	src="https://radames-stable-diffusion-depth2img.hf.space"
	frameborder="0"
	width="850"
	height="500"
></iframe>
