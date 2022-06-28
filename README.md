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

### Installation

```
pip install diffusers  # should install diffusers 0.0.4
```

### 1. `diffusers` as a toolbox for schedulers and models

`diffusers` is more modularized than `transformers`. The idea is that researchers and engineers can use only parts of the library easily for the own use cases.
It could become a central place for all kinds of models, schedulers, training utils and processors that one can mix and match for one's own use case.
Both models and schedulers should be load- and saveable from the Hub.

For more examples see [schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers) and [models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)

#### **Example for [DDPM](https://arxiv.org/abs/2006.11239):**

```python
import torch
from diffusers import UNetModel, DDPMScheduler
import PIL
import numpy as np
import tqdm

generator = torch.manual_seed(0)
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load models
noise_scheduler = DDPMScheduler.from_config("fusing/ddpm-lsun-church", tensor_format="pt")
unet = UNetModel.from_pretrained("fusing/ddpm-lsun-church").to(torch_device)

# 2. Sample gaussian noise
image = torch.randn(
    (1, unet.in_channels, unet.resolution, unet.resolution),
    generator=generator,
)
image = image.to(torch_device)

# 3. Denoise
num_prediction_steps = len(noise_scheduler)
for t in tqdm.tqdm(reversed(range(num_prediction_steps)), total=num_prediction_steps):
    # predict noise residual
    with torch.no_grad():
        residual = unet(image, t)

    # predict previous mean of image x_t-1
    pred_prev_image = noise_scheduler.step(residual, image, t)

    # optionally sample variance
    variance = 0
    if t > 0:
        noise = torch.randn(image.shape, generator=generator).to(image.device)
        variance = noise_scheduler.get_variance(t).sqrt() * noise

    # set current image to prev_image: x_t -> x_t-1
    image = pred_prev_image + variance

# 5. process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# 6. save image
image_pil.save("test.png")
```

#### **Example for [DDIM](https://arxiv.org/abs/2010.02502):**

```python
import torch
from diffusers import UNetModel, DDIMScheduler
import PIL
import numpy as np
import tqdm

generator = torch.manual_seed(0)
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load models
noise_scheduler = DDIMScheduler.from_config("fusing/ddpm-celeba-hq", tensor_format="pt")
unet = UNetModel.from_pretrained("fusing/ddpm-celeba-hq").to(torch_device)

# 2. Sample gaussian noise
image = torch.randn(
   (1, unet.in_channels, unet.resolution, unet.resolution),
   generator=generator,
)
image = image.to(torch_device)

# 3. Denoise                                                                                                                                           
num_inference_steps = 50
eta = 0.0  # <- deterministic sampling

for t in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
    # 1. predict noise residual
	orig_t = len(noise_scheduler) // num_inference_steps * t

    with torch.inference_mode():
        residual = unet(image, orig_t)

    # 2. predict previous mean of image x_t-1
    pred_prev_image = noise_scheduler.step(residual, image, t, num_inference_steps, eta)

    # 3. optionally sample variance
    variance = 0
    if eta > 0:
        noise = torch.randn(image.shape, generator=generator).to(image.device)
        variance = noise_scheduler.get_variance(t).sqrt() * eta * noise

    # 4. set current image to prev_image: x_t -> x_t-1
    image = pred_prev_image + variance

# 5. process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# 6. save image
image_pil.save("test.png")
```

#### **Examples for other modalities:**

[Diffuser](https://diffusion-planning.github.io/) for planning in reinforcement learning (currenlty only inference): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TmBmlYeKUZSkUZoJqfBmaicVTKx6nN1R?usp=sharing)

### 2. `diffusers` as a collection of popular Diffusion systems (Glide, Dalle, ...)

For more examples see [pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines).

#### **Example image generation with PNDM**

```python
from diffusers import PNDM, UNetModel, PNDMScheduler
import PIL.Image
import numpy as np
import torch

model_id = "fusing/ddim-celeba-hq"

model = UNetModel.from_pretrained(model_id)
scheduler = PNDMScheduler()

# load model and scheduler
pndm = PNDM(unet=model, noise_scheduler=scheduler)

# run pipeline in inference (sample random noise and denoise)
with torch.no_grad():
    image = pndm()

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) / 2
image_processed = torch.clamp(image_processed, 0.0, 1.0)
image_processed = image_processed * 255
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("test.png")
```

#### **Example 1024x1024 image generation with SDE VE**

See [paper](https://arxiv.org/abs/2011.13456) for more information on SDE VE.

```python
from diffusers import DiffusionPipeline
import torch
import PIL.Image
import numpy as np

torch.manual_seed(32)

score_sde_sv = DiffusionPipeline.from_pretrained("fusing/ffhq_ncsnpp")

# Note this might take up to 3 minutes on a GPU
image = score_sde_sv(num_inference_steps=2000)

image = image.permute(0, 2, 3, 1).cpu().numpy()
image = np.clip(image * 255, 0, 255).astype(np.uint8)
image_pil = PIL.Image.fromarray(image[0])

# save image
image_pil.save("test.png")
```
#### **Example 32x32 image generation with SDE VP**
	
See [paper](https://arxiv.org/abs/2011.13456) for more information on SDE VE.

```python
from diffusers import DiffusionPipeline
import torch
import PIL.Image
import numpy as np

torch.manual_seed(32)

score_sde_sv = DiffusionPipeline.from_pretrained("fusing/cifar10-ddpmpp-deep-vp")

# Note this might take up to 3 minutes on a GPU
image = score_sde_sv(num_inference_steps=1000)

image = image.permute(0, 2, 3, 1).cpu().numpy()
image = np.clip(image * 255, 0, 255).astype(np.uint8)
image_pil = PIL.Image.fromarray(image[0])

# save image
image_pil.save("test.png")
```


#### **Text to Image generation with Latent Diffusion**

_Note: To use latent diffusion install transformers from [this branch](https://github.com/patil-suraj/transformers/tree/ldm-bert)._

```python
from diffusers import DiffusionPipeline

ldm = DiffusionPipeline.from_pretrained("fusing/latent-diffusion-text2im-large")

generator = torch.manual_seed(42)

prompt = "A painting of a squirrel eating a burger"
image = ldm([prompt], generator=generator, eta=0.3, guidance_scale=6.0, num_inference_steps=50)

image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = image_processed  * 255.
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("test.png")
```

#### **Text to speech with GradTTS and BDDMPipeline**

```python
import torch
from diffusers import BDDMPipeline, GradTTSPipeline

torch_device = "cuda"

# load grad tts and bddm pipelines
grad_tts = GradTTSPipeline.from_pretrained("fusing/grad-tts-libri-tts")
bddm = BDDMPipeline.from_pretrained("fusing/diffwave-vocoder-ljspeech")

text = "Hello world, I missed you so much."

# generate mel spectograms using text
mel_spec = grad_tts(text, torch_device=torch_device)

#  generate the speech by passing mel spectograms to BDDMPipeline pipeline
generator = torch.manual_seed(42)
audio = bddm(mel_spec, generator, torch_device=torch_device)

# save generated audio
from scipy.io.wavfile import write as wavwrite
sampling_rate = 22050
wavwrite("generated_audio.wav", sampling_rate, audio.squeeze().cpu().numpy())
```

## TODO

- [ ] Create common API for models
- [ ] Add tests for models
- [ ] Adapt schedulers for training
- [ ] Write google colab for training
- [ ] Write docs / Think about how to structure docs
- [ ] Add tests to circle ci
- [ ] Add [Diffusion LM models](https://arxiv.org/pdf/2205.14217.pdf)
- [ ] Add more vision models
- [ ] Add more speech models
- [ ] Add RL model
- [ ] Add FID and KID metrics
