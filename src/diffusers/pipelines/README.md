# 🧨 Diffusers Pipelines

Pipelines provide a simple way to run state-of-the-art diffusion models in inference.
Most diffusion systems consist of multiple independently-trained models and highly adaptable scheduler
components - all of which are needed to have a functioning end-to-end diffusion system.

As an example, [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) has three independently trained models:
- [Autoencoder](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/models/vae.py#L392)
- [Conditional Unet](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/models/unet_2d_condition.py#L12)
- [CLIP text encoder](https://huggingface.co/docs/transformers/main/en/model_doc/clip#transformers.CLIPTextModel)
- a scheduler component, [scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py),
- a [CLIPImageProcessor](https://huggingface.co/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor),
- as well as a [safety checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py).
All of these components are necessary to run stable diffusion in inference even though they were trained
or created independently from each other.

To that end, we strive to offer all open-sourced, state-of-the-art diffusion system under a unified API.
More specifically, we strive to provide pipelines that
- 1. can load the officially published weights and yield 1-to-1 the same outputs as the original implementation according to the corresponding paper (*e.g.* [LDMTextToImagePipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/latent_diffusion), uses the officially released weights of [High-Resolution Image Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752)),
- 2. have a simple user interface to run the model in inference (see the [Pipelines API](#pipelines-api) section),
- 3. are easy to understand with code that is self-explanatory and can be read along-side the official paper (see [Pipelines summary](#pipelines-summary)),
- 4. can easily be contributed by the community (see the [Contribution](#contribution) section).

**Note** that pipelines do not (and should not) offer any training functionality.
If you are looking for *official* training examples, please have a look at [examples](https://github.com/huggingface/diffusers/tree/main/examples).


## Pipelines Summary

The following table summarizes all officially supported pipelines, their corresponding paper, and if
available a colab notebook to directly try them out.

| Pipeline                                                                                                                      | Source                                                                                                                       | Tasks | Colab
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|:---:|:---:|
| [dance diffusion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/dance_diffusion)                 | [**Dance Diffusion**](https://github.com/Harmonai-org/sample-generator)                                                      | *Unconditional Audio Generation* |
| [ddpm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddpm)                                       | [**Denoising Diffusion Probabilistic Models**](https://huggingface.co/papers/2006.11239)                                             | *Unconditional Image Generation* |
| [ddim](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim)                                       | [**Denoising Diffusion Implicit Models**](https://huggingface.co/papers/2010.02502)                                                  | *Unconditional Image Generation* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)
| [latent_diffusion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion)               | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://huggingface.co/papers/2112.10752)                         | *Text-to-Image Generation* |
| [latent_diffusion_uncond](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion_uncond) | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://huggingface.co/papers/2112.10752)                         | *Unconditional Image Generation* |
| [pndm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pndm)                                       | [**Pseudo Numerical Methods for Diffusion Models on Manifolds**](https://huggingface.co/papers/2202.09778)                           | *Unconditional Image Generation* |
| [score_sde_ve](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/score_sde_ve)                       | [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://openreview.net/forum?id=PxTIG12RRHS) | *Unconditional Image Generation* |
| [score_sde_vp](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/score_sde_vp)                       | [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://openreview.net/forum?id=PxTIG12RRHS) | *Unconditional Image Generation* |
| [stable_diffusion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion)               | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Text-to-Image Generation* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)
| [stable_diffusion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion)               | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Image-to-Image Text-Guided Generation* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb)
| [stable_diffusion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion)               | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release)                                            | *Text-Guided Image Inpainting* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/in_painting_with_stable_diffusion_using_diffusers.ipynb)
| [stochastic_karras_ve](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stochastic_karras_ve)       | [**Elucidating the Design Space of Diffusion-Based Generative Models**](https://huggingface.co/papers/2206.00364)                    | *Unconditional Image Generation* |

**Note**: Pipelines are simple examples of how to play around with the diffusion systems as described in the corresponding papers.
However, most of them can be adapted to use different scheduler components or even different model components. Some pipeline examples are shown in the [Examples](#examples) below.

## Pipelines API

Diffusion models often consist of multiple independently-trained models or other previously existing components.


Each model has been trained independently on a different task and the scheduler can easily be swapped out and replaced with a different one.
During inference, we however want to be able to easily load all components and use them in inference - even if one component, *e.g.* CLIP's text encoder, originates from a different library, such as [Transformers](https://github.com/huggingface/transformers). To that end, all pipelines provide the following functionality:

- [`from_pretrained` method](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/pipeline_utils.py#L139) that accepts a Hugging Face Hub repository id, *e.g.* [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) or a path to a local directory, *e.g.*
"./stable-diffusion". To correctly retrieve which models and components should be loaded, one has to provide a `model_index.json` file, *e.g.* [stable-diffusion-v1-5/stable-diffusion-v1-5/model_index.json](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/model_index.json), which defines all components that should be
loaded into the pipelines. More specifically, for each model/component one needs to define the format `<name>: ["<library>", "<class name>"]`. `<name>` is the attribute name given to the loaded instance of `<class name>` which can be found in the library or pipeline folder called `"<library>"`.
- [`save_pretrained`](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/pipeline_utils.py#L90) that accepts a local path, *e.g.* `./stable-diffusion` under which all models/components of the pipeline will be saved. For each component/model a folder is created inside the local path that is named after the given attribute name, *e.g.* `./stable_diffusion/unet`.
In addition, a `model_index.json` file is created at the root of the local path, *e.g.* `./stable_diffusion/model_index.json` so that the complete pipeline can again be instantiated
from the local path.
- [`to`](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/pipeline_utils.py#L118) which accepts a `string` or `torch.device` to move all models that are of type `torch.nn.Module` to the passed device. The behavior is fully analogous to [PyTorch's `to` method](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to).
- [`__call__`] method to use the pipeline in inference. `__call__` defines inference logic of the pipeline and should ideally encompass all aspects of it, from pre-processing to forwarding tensors to the different models and schedulers, as well as post-processing. The API of the `__call__` method can strongly vary from pipeline to pipeline. *E.g.* a text-to-image pipeline, such as [`StableDiffusionPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) should accept among other things the text prompt to generate the image. A pure image generation pipeline, such as [DDPMPipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ddpm) on the other hand can be run without providing any inputs. To better understand what inputs can be adapted for
each pipeline, one should look directly into the respective pipeline.

**Note**: All pipelines have PyTorch's autograd disabled by decorating the `__call__` method with a [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html) decorator because pipelines should
not be used for training. If you want to store the gradients during the forward pass, we recommend writing your own pipeline, see also our [community-examples](https://github.com/huggingface/diffusers/tree/main/examples/community)

## Contribution

We are more than happy about any contribution to the officially supported pipelines 🤗. We aspire
all of our pipelines to be  **self-contained**, **easy-to-tweak**, **beginner-friendly** and for **one-purpose-only**.

- **Self-contained**: A pipeline shall be as self-contained as possible. More specifically, this means that all functionality should be either directly defined in the pipeline file itself, should be inherited from (and only from) the [`DiffusionPipeline` class](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/pipeline_utils.py#L56) or be directly attached to the model and scheduler components of the pipeline.
- **Easy-to-use**: Pipelines should be extremely easy to use - one should be able to load the pipeline and
use it for its designated task, *e.g.* text-to-image generation, in just a couple of lines of code. Most
logic including pre-processing, an unrolled diffusion loop, and post-processing should all happen inside the `__call__` method.
- **Easy-to-tweak**: Certain pipelines will not be able to handle all use cases and tasks that you might like them to. If you want to use a certain pipeline for a specific use case that is not yet supported, you might have to copy the pipeline file and tweak the code to your needs. We try to make the pipeline code as readable as possible so that each part –from pre-processing to diffusing to post-processing– can easily be adapted. If you would like the community to benefit from your customized pipeline, we would love to see a contribution to our [community-examples](https://github.com/huggingface/diffusers/tree/main/examples/community). If you feel that an important pipeline should be part of the official pipelines but isn't, a contribution to the [official pipelines](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines) would be even better.
- **One-purpose-only**: Pipelines should be used for one task and one task only. Even if two tasks are very similar from a modeling point of view, *e.g.* image2image translation and in-painting, pipelines shall be used for one task only to keep them *easy-to-tweak* and *readable*.

## Examples

### Text-to-Image generation with Stable Diffusion

```python
# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")
```

### Image-to-Image text-guided generation with Stable Diffusion

The `StableDiffusionImg2ImgPipeline` lets you pass a text prompt and an initial image to condition the generation of new images.

```python
import requests
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to(device)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

images[0].save("fantasy_landscape.png")
```
You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb)

### Tweak prompts reusing seeds and latents

You can generate your own latents to reproduce results, or tweak your prompt on a specific result you liked. [This notebook](https://github.com/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb) shows how to do it step by step. You can also run it in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb).


### In-painting using Stable Diffusion

The `StableDiffusionInpaintPipeline` lets you edit specific parts of an image by providing a mask and text prompt.

```python
import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
```

You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/in_painting_with_stable_diffusion_using_diffusers.ipynb)
