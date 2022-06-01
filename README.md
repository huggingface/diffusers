# Diffusers

## Library structure:

```
├── models
│   ├── dalle2
│   │   ├── modeling_dalle2.py
│   │   ├── README.md
│   │   └── run_dalle2.py
│   ├── ddpm
│   │   ├── modeling_ddpm.py
│   │   ├── README.md
│   │   └── run_ddpm.py
│   ├── glide
│   │   ├── modeling_glide.py
│   │   ├── README.md
│   │   └── run_dalle2.py
│   ├── imagen
│   │   ├── modeling_dalle2.py
│   │   ├── README.md
│   │   └── run_dalle2.py
│   └── latent_diffusion
│       ├── modeling_latent_diffusion.py
│       ├── README.md
│       └── run_latent_diffusion.py
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

## Dummy Example
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
