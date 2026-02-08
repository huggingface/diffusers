<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Z-Image

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Z-Image](https://huggingface.co/papers/2511.22699) is a powerful and highly efficient image generation model with 6B parameters. Currently there's only one model with two more to be released:

|Model|Hugging Face|
|---|---|
|Z-Image-Turbo|https://huggingface.co/Tongyi-MAI/Z-Image-Turbo|

## Z-Image-Turbo

Z-Image-Turbo is a distilled version of Z-Image that matches or exceeds leading competitors with only 8 NFEs (Number of Function Evaluations). It offers sub-second inference latency on enterprise-grade H800 GPUs and fits comfortably within 16G VRAM consumer devices. It excels in photorealistic image generation, bilingual text rendering (English & Chinese), and robust instruction adherence.

## Image-to-image

Use [`ZImageImg2ImgPipeline`] to transform an existing image based on a text prompt.

```python
import torch
from diffusers import ZImageImg2ImgPipeline
from diffusers.utils import load_image

pipe = ZImageImg2ImgPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

prompt = "A fantasy landscape with mountains and a river, detailed, vibrant colors"
image = pipe(
    prompt,
    image=init_image,
    strength=0.6,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("zimage_img2img.png")
```

## Inpainting

Use [`ZImageInpaintPipeline`] to inpaint specific regions of an image based on a text prompt and mask.

```python
import torch
import numpy as np
from PIL import Image
from diffusers import ZImageInpaintPipeline
from diffusers.utils import load_image

pipe = ZImageInpaintPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

# Create a mask (white = inpaint, black = preserve)
mask = np.zeros((1024, 1024), dtype=np.uint8)
mask[256:768, 256:768] = 255  # Inpaint center region
mask_image = Image.fromarray(mask)

prompt = "A beautiful lake with mountains in the background"
image = pipe(
    prompt,
    image=init_image,
    mask_image=mask_image,
    strength=1.0,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("zimage_inpaint.png")
```

## ZImagePipeline

[[autodoc]] ZImagePipeline
	- all
	- __call__

## ZImageImg2ImgPipeline

[[autodoc]] ZImageImg2ImgPipeline
	- all
	- __call__

## ZImageInpaintPipeline

[[autodoc]] ZImageInpaintPipeline
	- all
	- __call__
