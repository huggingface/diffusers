# Stable Diffusion

## Overview

Stable Diffusion was proposed in [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-announcement) by Patrick Esser and Robin Rombach and the Stability AI team.

The summary of the model is the following:

*Stable Diffusion is a text-to-image model that will empower billions of people to create stunning art within seconds. It is a breakthrough in speed and quality meaning that it can run on consumer GPUs. You can see some of the amazing output that has been created by this model without pre or post-processing on this page. The model itself builds upon the work of the team at CompVis and Runway in their widely used latent diffusion model combined with insights from the conditional diffusion models by our lead generative AI developer Katherine Crowson, Dall-E 2 by Open AI, Imagen by Google Brain and many others. We are delighted that AI media generation is a cooperative field and hope it can continue this way to bring the gift of creativity to all.* 

## Tips:

- Stable Diffusion has the same architecture as [Latent Diffusion](https://arxiv.org/abs/2112.10752) but uses a frozen CLIP Text Encoder instead of training the text encoder jointly with the diffusion model.
- An in-detail explanation of the Stable Diffusion model can be found under [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion).
- If you don't want to rely on the Hugging Face Hub and having to pass a authentication token, you can 
download the weights with `git lfs install; git clone https://huggingface.co/runwayml/stable-diffusion-v1-5` and instead pass the local path to the cloned folder to `from_pretrained` as shown below.
- Stable Diffusion can work with a variety of different samplers as is shown below.

## Available Pipelines:

| Pipeline | Tasks | Colab
|---|---|:---:|
| [pipeline_stable_diffusion.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) | *Text-to-Image Generation* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)
| [pipeline_stable_diffusion_img2img](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py) | *Image-to-Image Text-Guided Generation* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb)
| [pipeline_stable_diffusion_inpaint](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py) | *Text-Guided Image Inpainting* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/in_painting_with_stable_diffusion_using_diffusers.ipynb)

## Examples:

### Using Stable Diffusion without being logged into the Hub.

If you want to download the model weights using a single Python line, you need to be logged in via `huggingface-cli login`. 

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
```

This however can make it difficult to build applications on top of `diffusers` as you will always have to pass the token around. A potential way to solve this issue is by downloading the weights to a local path `"./stable-diffusion-v1-5"`:

```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

and simply passing the local path to `from_pretrained`:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
```

### Text-to-Image with default PLMS scheduler

```python
# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).sample[0]  
    
image.save("astronaut_rides_horse.png")
```

### Text-to-Image with DDIM scheduler

```python
# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, DDIMScheduler

scheduler =  DDIMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    scheduler=scheduler,
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).sample[0]  
    
image.save("astronaut_rides_horse.png")
```

### Text-to-Image with K-LMS scheduler

```python
# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

lms = LMSDiscreteScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    scheduler=lms,
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).sample[0]  
    
image.save("astronaut_rides_horse.png")
```
