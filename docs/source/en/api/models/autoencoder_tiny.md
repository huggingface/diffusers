# Tiny Autoencoder

This Autoencoder was introduced in [this repository](https://github.com/madebyollin/taesd). It is tiny in shape and can decode the latents in a [`StableDiffusionPipeline`] or [`StableDiffusionXLPipeline`] almost instantly. 

## Using with Stable Diffusion v-2.1

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

## Using with Stable Diffusion XL 1.0

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