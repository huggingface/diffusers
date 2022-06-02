# Diffusers

## Definitions

**Models**: Single neural network that models p_θ(x_t-1|x_t) and is trained to “denoise” to image
*Examples: UNet, Conditioned UNet, 3D UNet, Transformer UNet*

![model_diff_1_50](https://user-images.githubusercontent.com/23423619/171610307-dab0cd8b-75da-4d4e-9f5a-5922072e2bb5.png)

**Samplers**: Algorithm to *train* and *sample* from **Model**. Defines alpha and beta schedule, timesteps, etc..
*Example: Vanilla DDPM, DDIM, PMLS, DEIN*

![sampling](https://user-images.githubusercontent.com/23423619/171608981-3ad05953-a684-4c82-89f8-62a459147a07.png)
![training](https://user-images.githubusercontent.com/23423619/171608964-b3260cce-e6b4-4841-959d-7d8ba4b8d1b2.png)

**Diffusion Pipeline**: End-to-end pipeline that includes multiple diffusion models, possible text encoders, CLIP
*Example: GLIDE,CompVis/Latent-Diffusion, Imagen, DALL-E*

![imagen](https://user-images.githubusercontent.com/23423619/171609001-c3f2c1c9-f597-4a16-9843-749bf3f9431c.png)

## Library structure:

```
├── models
│   ├── audio
│   │   └── fastdiff
│   │       ├── modeling_fastdiff.py
│   │       ├── README.md
│   │       └── run_fastdiff.py
│   └── vision
│       ├── dalle2
│       │   ├── modeling_dalle2.py
│       │   ├── README.md
│       │   └── run_dalle2.py
│       ├── ddpm
│       │   ├── modeling_ddpm.py
│       │   ├── README.md
│       │   └── run_ddpm.py
│       ├── glide
│       │   ├── modeling_glide.py
│       │   ├── README.md
│       │   └── run_dalle2.py
│       ├── imagen
│       │   ├── modeling_dalle2.py
│       │   ├── README.md
│       │   └── run_dalle2.py
│       └── latent_diffusion
│           ├── modeling_latent_diffusion.py
│           ├── README.md
│           └── run_latent_diffusion.py

├── src
│   └── diffusers
│       ├── configuration_utils.py
│       ├── __init__.py
│       ├── modeling_utils.py
│       ├── models
│       │   └── unet.py
│       ├── processors
│       └── samplers
│           ├── gaussian.py
├── tests
│   └── test_modeling_utils.py
```

## 1. `diffusers` as a central modular diffusion and sampler library

`diffusers` should be more modularized than `transformers` so that parts of it can be easily used in other libraries.
It could become a central place for all kinds of models, samplers, training utils and processors required when using diffusion models in audio, vision, ... 
One should be able to save both models and samplers as well as load them from the Hub.

Example:

```python
from diffusers import UNetModel, GaussianDiffusion
import torch

# 1. Load model
unet = UNetModel.from_pretrained("fusing/ddpm_dummy")

# 2. Do one denoising step with model
batch_size, num_channels, height, width = 1, 3, 32, 32
dummy_noise = torch.ones((batch_size, num_channels, height, width))
time_step = torch.tensor([10])
image = unet(dummy_noise, time_step)

# 3. Load sampler
sampler = GaussianDiffusion.from_config("fusing/ddpm_dummy")

# 4. Sample image from sampler passing the model
image = sampler.sample(model, batch_size=1)

print(image)
```
