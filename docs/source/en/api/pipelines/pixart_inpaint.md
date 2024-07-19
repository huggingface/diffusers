# PixArt Inpainting
The inpainting method based on the PixArt model is similar to the usage of the Stable Diffusion model for inpainting.

```python
import torch

import diffusers
from diffusers import PixArtAlphaInpaintPipeline

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = diffusers.utils.load_image(img_url).resize((1024, 1024))
mask_image = diffusers.utils.load_image(mask_url).resize((1024, 1024))

# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too
pipe = PixArtAlphaInpaintPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16
     )
pipe = pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
```
## PixArtAlphaPipeline

[[autodoc]] PixArtAlphaPipeline
	- all
	- __call__