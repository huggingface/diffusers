<p align="center">
    <br>
    <img src="docs/source/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
    <a href="https://github.com/huggingface/diffusers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>

🤗 Diffusers provides pretrained diffusion models across multiple modalities, such as vision and audio, and serves
as a modular toolbox for inference and training of diffusion models.

More precisely, 🤗 Diffusers offers:

- State-of-the-art diffusion pipelines that can be run in inference with just a couple of lines of code (see [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)).
- Various noise schedulers that can be used interchangeably for the prefered speed vs. quality trade-off in inference (see [src/diffusers/schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)).
- Multiple types of models, such as UNet, that can be used as building blocks in an end-to-end diffusion system (see [src/diffusers/models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)).
- Training examples to show how to train the most popular diffusion models (see [examples](https://github.com/huggingface/diffusers/tree/main/examples)).

## Definitions

**Models**: Neural network that models $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ (see image below) and is trained end-to-end to *denoise* a noisy input to an image.
*Examples*: UNet, Conditioned UNet, 3D UNet, Transformer UNet

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>
    
**Schedulers**: Algorithm class for both **inference** and **training**.
The class provides functionality to compute previous image according to alpha, beta schedule as well as predict noise for training.
*Examples*: [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), [PNDM](https://arxiv.org/abs/2202.09778), [DEIS](https://arxiv.org/abs/2204.13902)

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349706-53d58acc-a4d1-4cda-b3e8-432d9dc7ad38.png" width="800"/>
    <br>
    <em> Sampling and training algorithms. Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>
    

**Diffusion Pipeline**: End-to-end pipeline that includes multiple diffusion models, possible text encoders, ...
*Examples*: Glide, Latent-Diffusion, Imagen, DALL-E 2

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174348898-481bd7c2-5457-4830-89bc-f0907756f64c.jpeg" width="550"/>
    <br>
    <em> Figure from ImageGen (https://imagen.research.google/). </em>
<p>
    
## Philosophy

- Readability and clarity is prefered over highly optimized code. A strong importance is put on providing readable, intuitive and elementary code design. *E.g.*, the provided [schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers) are separated from the provided [models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models) and provide well-commented code that can be read alongside the original paper.
- Diffusers is **modality independent** and focusses on providing pretrained models and tools to build systems that generate **continous outputs**, *e.g.* vision and audio.
- Diffusion models and schedulers are provided as consise, elementary building blocks whereas diffusion pipelines are a collection of end-to-end diffusion systems that can be used out-of-the-box, should stay as close as possible to their original implementation and can include components of other library, such as text-encoders. Examples for diffusion pipelines are [Glide](https://github.com/openai/glide-text2im) and [Latent Diffusion](https://github.com/CompVis/latent-diffusion).

## Quickstart

**Check out this notebook: https://colab.research.google.com/drive/1nMfF04cIxg6FujxsNYi9kiTRrzj4_eZU?usp=sharing**

### Installation

```
pip install diffusers  # should install diffusers 0.0.4
```

### 1. `diffusers` as a toolbox for schedulers and models

`diffusers` is more modularized than `transformers`. The idea is that researchers and engineers can use only parts of the library easily for the own use cases.
It could become a central place for all kinds of models, schedulers, training utils and processors that one can mix and match for one's own use case.
Both models and schedulers should be load- and saveable from the Hub.

For more examples see [schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers) and [models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)

#### **Example for Unconditonal Image generation [DDPM](https://arxiv.org/abs/2006.11239):**

```python
import torch
from diffusers import UNetUnconditionalModel, DDIMScheduler
import PIL.Image
import numpy as np
import tqdm

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load models
scheduler = DDIMScheduler.from_config("fusing/ddpm-celeba-hq", tensor_format="pt")
unet = UNetUnconditionalModel.from_pretrained("fusing/ddpm-celeba-hq", ddpm=True).to(torch_device)

# 2. Sample gaussian noise
generator = torch.manual_seed(23)
unet.image_size = unet.resolution
image = torch.randn(
   (1, unet.in_channels, unet.image_size, unet.image_size),
   generator=generator,
)
image = image.to(torch_device)

# 3. Denoise
num_inference_steps = 50
eta = 0.0  # <- deterministic sampling
scheduler.set_timesteps(num_inference_steps)

for t in tqdm.tqdm(scheduler.timesteps):
    # 1. predict noise residual
    with torch.no_grad():
        residual = unet(image, t)["sample"]

    prev_image = scheduler.step(residual, t, image, eta)["prev_sample"]

    # 3. set current image to prev_image: x_t -> x_t-1
    image = prev_image

# 4. process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# 5. save image
image_pil.save("generated_image.png")
``` 

#### **Example for Unconditonal Image generation [LDM](https://github.com/CompVis/latent-diffusion):**

```python
```
