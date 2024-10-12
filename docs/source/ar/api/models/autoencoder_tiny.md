# Tiny AutoEncoder

TAESD) هو إصدار مصغر ومقطّر من VAE الخاص بـ Stable Diffusion، والذي قدمه [madebyollin/taesd](https://github.com/madebyollin/taesd) بواسطة Ollin Boer Bohan. يمكنه فك تشفير الرموز المخفية بسرعة في [`StableDiffusionPipeline`] أو [`StableDiffusionXLPipeline`] في غضون لحظات.

للاستخدام مع Stable Diffusion v-2.1:

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
image
```

للاستخدام مع Stable Diffusion XL 1.0:

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
image
```

## AutoencoderTiny

[[autodoc]] AutoencoderTiny

## AutoencoderTinyOutput

[[autodoc]] models.autoencoders.autoencoder_tiny.AutoencoderTinyOutput