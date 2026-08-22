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

pipe = ZImageImg2ImgPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", dtype=torch.bfloat16)
pipe.to("cuda")

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

prompt = "A fantasy landscape with mountains and a river, detailed, vibrant colors"
image = pipe(
    prompt,
    image=init_image,
    strength=0.6,
    num_inference_steps=8,
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

pipe = ZImageInpaintPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", dtype=torch.bfloat16)
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
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("zimage_inpaint.png")
```

## Modular inpainting

[`ModularPipeline`] automatically selects the Z-Image inpainting workflow when both `image` and `mask_image` are provided. White mask regions are regenerated and black regions are preserved.
Use `padding_mask_crop` to generate only around the masked region; it requires the default PIL output so the result can be overlaid onto the original image.

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipe = ModularPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image("path/to/image.png").convert("RGB")
mask_image = load_image("path/to/mask.png").convert("L")

output = pipe(
    prompt="A beautiful lake with mountains in the background",
    image=image,
    mask_image=mask_image,
    height=image.height,
    width=image.width,
    strength=1.0,
    num_inference_steps=8,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output="images",
)[0]
output.save("zimage_modular_inpaint.png")
```

To add a ControlNet inpaint condition, load a compatible [`ZImageControlNetModel`] and update the modular pipeline. The control image is used together with the source image and mask. `control_guidance_start` and `control_guidance_end` specify the normalized denoising interval in which ControlNet is active.

```python
import torch
from huggingface_hub import hf_hub_download
from diffusers import ModularPipeline, ZImageControlNetModel
from diffusers.utils import load_image

controlnet = ZImageControlNetModel.from_single_file(
    hf_hub_download(
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0",
        filename="Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
    ),
    torch_dtype=torch.bfloat16,
)

pipe = ModularPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo")
pipe.load_components(dtype=torch.bfloat16)
pipe.update_components(controlnet=controlnet)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0/resolve/main/asset/inpaint.jpg?download=true"
).convert("RGB")
mask_image = load_image(
    "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0/resolve/main/asset/mask.jpg?download=true"
).convert("L")
control_image = load_image(
    "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0/resolve/main/asset/pose.jpg?download=true"
).convert("RGB")

output = pipe(
    prompt="A woman standing on a sunny coast, full-body portrait",
    image=image,
    mask_image=mask_image,
    control_image=control_image,
    controlnet_conditioning_scale=0.75,
    control_guidance_start=0.0,
    control_guidance_end=1.0,
    height=image.height,
    width=image.width,
    num_inference_steps=25,
    generator=torch.Generator(device="cuda").manual_seed(43),
    output="images",
)[0]
output.save("zimage_modular_controlnet_inpaint.png")
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
