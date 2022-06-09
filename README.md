# Diffusers

## Definitions

**Models**: Single neural network that models p_θ(x_t-1|x_t) and is trained to “denoise” to image
*Examples: UNet, Conditioned UNet, 3D UNet, Transformer UNet*

![model_diff_1_50](https://user-images.githubusercontent.com/23423619/171610307-dab0cd8b-75da-4d4e-9f5a-5922072e2bb5.png)

**Schedulers**: Algorithm to sample noise schedule for both *training* and *inference*. Defines alpha and beta schedule, timesteps, etc..
*Example: Gaussian DDPM, DDIM, PMLS, DEIN*

![sampling](https://user-images.githubusercontent.com/23423619/171608981-3ad05953-a684-4c82-89f8-62a459147a07.png)
![training](https://user-images.githubusercontent.com/23423619/171608964-b3260cce-e6b4-4841-959d-7d8ba4b8d1b2.png)

**Diffusion Pipeline**: End-to-end pipeline that includes multiple diffusion models, possible text encoders, CLIP
*Example: GLIDE,CompVis/Latent-Diffusion, Imagen, DALL-E*

![imagen](https://user-images.githubusercontent.com/23423619/171609001-c3f2c1c9-f597-4a16-9843-749bf3f9431c.png)

## 1. `diffusers` as a central modular diffusion and sampler library

`diffusers` is more modularized than `transformers`. The idea is that researchers and engineers can use only parts of the library easily for the own use cases.
It could become a central place for all kinds of models, schedulers, training utils and processors that one can mix and match for one's own use case.
Both models and scredulers should be load- and saveable from the Hub.

Example:

```python
import torch
from diffusers import UNetModel, GaussianDDPMScheduler
import PIL
import numpy as np

generator = torch.Generator()
generator = generator.manual_seed(6694729458485568)
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load models
scheduler = GaussianDDPMScheduler.from_config("fusing/ddpm-lsun-church")
model = UNetModel.from_pretrained("fusing/ddpm-lsun-church").to(torch_device)

# 2. Sample gaussian noise
image = scheduler.sample_noise((1, model.in_channels, model.resolution, model.resolution), device=torch_device, generator=generator)

# 3. Denoise                                                                                                                                           
for t in reversed(range(len(scheduler))):
	# 1. predict noise residual
	with torch.no_grad():
		pred_noise_t = self.unet(image, t)

	# 2. compute alphas, betas
	alpha_prod_t = self.noise_scheduler.get_alpha_prod(t)
	alpha_prod_t_prev = self.noise_scheduler.get_alpha_prod(t - 1)
	beta_prod_t = 1 - alpha_prod_t
	beta_prod_t_prev = 1 - alpha_prod_t_prev

	# 3. compute predicted image from residual
	# First: compute predicted original image from predicted noise also called
	# "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
	pred_original_image = (image - beta_prod_t.sqrt() * pred_noise_t) / alpha_prod_t.sqrt()

	# Second: Clip "predicted x_0"
	pred_original_image = torch.clamp(pred_original_image, -1, 1)

	# Third: Compute coefficients for pred_original_image x_0 and current image x_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
	pred_original_image_coeff = (alpha_prod_t_prev.sqrt() * self.noise_scheduler.get_beta(t)) / beta_prod_t
	current_image_coeff = self.noise_scheduler.get_alpha(t).sqrt() * beta_prod_t_prev / beta_prod_t
	# Fourth: Compute predicted previous image µ_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
	pred_prev_image = pred_original_image_coeff * pred_original_image + current_image_coeff * image

	# 5. For t > 0, compute predicted variance βt (see formala (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
	# and sample from it to get previous image
	# x_{t-1} ~ N(pred_prev_image, variance) == add variane to pred_image
	if t > 0:
		variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.noise_scheduler.get_beta(t).sqrt()
		noise = self.noise_scheduler.sample_noise(image.shape, device=image.device, generator=generator)
		prev_image = pred_prev_image + variance * noise
	else:
		prev_image = pred_prev_image

	# 6. Set current image to prev_image: x_t -> x_t-1
	image = prev_image

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("test.png")
```

## 2. `diffusers` as a collection of most important Diffusion systems (GLIDE, Dalle, ...)
`models` directory in repository hosts the complete code necessary for running a diffusion system as well as to train it. A `DiffusionPipeline` class allows to easily run the diffusion model in inference:

Example:

```python
from diffusers import DiffusionPipeline
import PIL.Image
import numpy as np

# load model and scheduler
ddpm = DiffusionPipeline.from_pretrained("fusing/ddpm-lsun-bedroom")

# run pipeline in inference (sample random noise and denoise)
image = ddpm()

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("test.png")
```

## Library structure:

```
├── models
│   ├── audio
│   │   └── fastdiff
│   │       ├── modeling_fastdiff.py
│   │       ├── README.md
│   │       └── run_fastdiff.py
│   ├── __init__.py
│   └── vision
│       ├── dalle2
│       │   ├── modeling_dalle2.py
│       │   ├── README.md
│       │   └── run_dalle2.py
│       ├── ddpm
│       │   ├── example.py
│       │   ├── modeling_ddpm.py
│       │   ├── README.md
│       │   └── run_ddpm.py
│       ├── glide
│       │   ├── modeling_glide.py
│       │   ├── modeling_vqvae.py.py
│       │   ├── README.md
│       │   └── run_glide.py
│       ├── imagen
│       │   ├── modeling_dalle2.py
│       │   ├── README.md
│       │   └── run_dalle2.py
│       ├── __init__.py
│       └── latent_diffusion
│           ├── modeling_latent_diffusion.py
│           ├── README.md
│           └── run_latent_diffusion.py
├── pyproject.toml
├── README.md
├── setup.cfg
├── setup.py
├── src
│   └── diffusers
│       ├── configuration_utils.py
│       ├── __init__.py
│       ├── modeling_utils.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── unet_glide.py
│       │   └── unet.py
│       ├── pipeline_utils.py
│       └── schedulers
│           ├── gaussian_ddpm.py
│           ├── __init__.py
├── tests
│   └── test_modeling_utils.py
```
