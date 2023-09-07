# Tiny AutoEncoder

Tiny AutoEncoder for Stable Diffusion (TAESD) was introduced in [madebyollin/taesd](https://github.com/madebyollin/taesd) by Ollin Boer Bohan. It is a tiny distilled version of Stable Diffusion's VAE that can quickly decode the latents in a [`StableDiffusionPipeline`] or [`StableDiffusionXLPipeline`] almost instantly. 

To use with Stable Diffusion v-2.1:

```python
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "slice of delicious New York-style berry cheesecake"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("cheesecake.png")
```

To use with Stable Diffusion XL 1.0

```python
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "slice of delicious New York-style berry cheesecake"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("cheesecake_sdxl.png")
```

## AutoencoderTiny

[[autodoc]] AutoencoderTiny

## AutoencoderTinyOutput

[[autodoc]] models.autoencoder_tiny.AutoencoderTinyOutput