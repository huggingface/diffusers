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

- State-of-the-art diffusion pipelines that can be run in inference with just a couple of lines of code (see [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)). Check [this overview](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/README.md#pipelines-summary) to see all supported pipelines and their corresponding official papers.
- Various noise schedulers that can be used interchangeably for the prefered speed vs. quality trade-off in inference (see [src/diffusers/schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)).
- Multiple types of models, such as UNet, can be used as building blocks in an end-to-end diffusion system (see [src/diffusers/models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)).
- Training examples to show how to train the most popular diffusion models (see [examples/training](https://github.com/huggingface/diffusers/tree/main/examples/training)).
- Inference examples to show how to create custom pipelines for advanced tasks such as image2image, in-painting (see [examples/inference](https://github.com/huggingface/diffusers/tree/main/examples/inference))


## Quickstart

In order to get started, we recommend taking a look at two notebooks:

- The [Getting started with Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) notebook, which showcases an end-to-end example of usage for diffusion models, schedulers and pipelines.
  Take a look at this notebook to learn how to use the pipeline abstraction, which takes care of everything (model, scheduler, noise handling) for you, and also to understand each independent building block in the library.
- The [Training a diffusers model](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook summarizes diffusion models training methods. This notebook takes a step-by-step approach to training your
  diffusion models on an image dataset, with explanatory graphics. 
  
## **New ** Stable Diffusion is now fully compatible with `diffusers`!  

Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). It's trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See the [model card](https://huggingface.co/CompVis/stable-diffusion) for more information.

You need to accept the model license before downloading or using the Stable Diffusion weights. Please, visit the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-3), read the license and tick the checkbox if you agree. You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section](https://huggingface.co/docs/hub/security-tokens) of the documentation.

### Text-to-Image generation with Stable Diffusion

```python
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    revision="fp16", 
    torch_dtype=torch.float16,
    scheduler=lms,
    use_auth_token=True
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  
    
image.save("astronaut_rides_horse.png")
```

### Image-to-Image text-guided generation with Stable Diffusion

The `StableDiffusionImg2ImgPipeline` lets you pass a text prompt and an initial image to condition the generation of new images.

```python
from torch import autocast
import requests
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(device)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)["sample"]

images[0].save("fantasy_landscape.png")
```
You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/image_2_image_using_diffusers.ipynb)

### In-painting using Stable Diffusion

The `StableDiffusionInpaintPipeline` lets you edit specific parts of an image by providing a mask and text prompt.

```python
from io import BytesIO

from torch import autocast
import requests
import PIL

from diffusers import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

device = "cuda"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(device)

prompt = "a cat sitting on a bench"
with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75)["sample"]

images[0].save("cat_on_bench.png")
```

### Tweak prompts reusing seeds and latents

You can generate your own latents to reproduce results, or tweak your prompt on a specific result you liked. [This notebook](stable-diffusion-seeds.ipynb) shows how to do it step by step. You can also run it in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb).


For more details, check out [the Stable Diffusion notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)
and have a look into the [release notes](https://github.com/huggingface/diffusers/releases/tag/v0.2.0).
  
## Examples

There are many ways to try running Diffusers! Here we outline code-focused tools (primarily using `DiffusionPipeline`s and Google Colab) and interactive web-tools.

### Running Code

If you want to run the code yourself 💻, you can try out:
- [Text-to-Image Latent Diffusion](https://huggingface.co/CompVis/ldm-text2im-large-256)
```python
# !pip install diffusers transformers
from diffusers import DiffusionPipeline

model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]

# save images
for idx, image in enumerate(images):
    image.save(f"squirrel-{idx}.png")
```
- [Unconditional Diffusion with discrete scheduler](https://huggingface.co/google/ddpm-celebahq-256)
```python
# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-celebahq-256"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

# run pipeline in inference (sample random noise and denoise)
image = ddpm()["sample"]

# save image
image[0].save("ddpm_generated_image.png")
```
- [Unconditional Latent Diffusion](https://huggingface.co/CompVis/ldm-celebahq-256)
- [Unconditional Diffusion with continous scheduler](https://huggingface.co/google/ncsnpp-ffhq-1024)

**Other Notebooks**:
* [image-to-image generation with Stable Diffusion](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/image_2_image_using_diffusers.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg),
* [tweak images via repeated Stable Diffusion seeds](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg),

### Web Demos
If you just want to play around with some web demos, you can try out the following 🚀 Spaces:
| Model                          	| Hugging Face Spaces                                                                                                                                               	|
|--------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| Text-to-Image Latent Diffusion 	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CompVis/text2img-latent-diffusion) 	|
| Faces generator                	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CompVis/celeba-latent-diffusion)    	|
| DDPM with different schedulers 	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/fusing/celeba-diffusion)           	|
| Conditional generation from sketch (*SOON*) 	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/huggingface/diffuse-the-rest)           	|
| Composable diffusion | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Shuang59/Composable-Diffusion)           	|

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
- Diffusers is **modality independent** and focuses on providing pretrained models and tools to build systems that generate **continous outputs**, *e.g.* vision and audio.
- Diffusion models and schedulers are provided as concise, elementary building blocks. In contrast, diffusion pipelines are a collection of end-to-end diffusion systems that can be used out-of-the-box, should stay as close as possible to their original implementation and can include components of another library, such as text-encoders. Examples for diffusion pipelines are [Glide](https://github.com/openai/glide-text2im) and [Latent Diffusion](https://github.com/CompVis/latent-diffusion).

## Installation

**With `pip`**
    
```bash
pip install --upgrade diffusers  # should install diffusers 0.2.4
```

**With `conda`**

```sh
conda install -c conda-forge diffusers
```

## In the works

For the first release, 🤗 Diffusers focuses on text-to-image diffusion techniques. However, diffusers can be used for much more than that! Over the upcoming releases, we'll be focusing on:

- Diffusers for audio
- Diffusers for reinforcement learning (initial work happening in https://github.com/huggingface/diffusers/pull/105).
- Diffusers for video generation
- Diffusers for molecule generation (initial work happening in https://github.com/huggingface/diffusers/pull/54)

A few pipeline components are already being worked on, namely:

- BDDMPipeline for spectrogram-to-sound vocoding
- GLIDEPipeline to support OpenAI's GLIDE model
- Grad-TTS for text to audio generation / conditional audio generation

We want diffusers to be a toolbox useful for diffusers models in general; if you find yourself limited in any way by the current API, or would like to see additional models, schedulers, or techniques, please open a [GitHub issue](https://github.com/huggingface/diffusers/issues) mentioning what you would like to see.

## Credits

This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:

- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim).
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)

We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.
