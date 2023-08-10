# Stable Diffusion XL

[[open-in-colab]]

[Stable Diffusion XL](https://huggingface.co/papers/2307.01952) (SDXL) is a powerful text-to-image generation model that iterates on the previous Stable Diffusion models in three key ways:

1. the UNet is 3x larger and SDXL combines a second text encoder (OpenCLIP ViT-bigG/14) with the original text encoder to significantly increase the number of parameters
2. introduces size and crop-conditioning to preserve training data from being discarded and gain more control over how a generated image should be cropped
3. introduces a two-stage model process; the *base* model (can also be run as a standalone model) generates an image as an input to the *refiner* model which adds additional high-quality details

This guide will show you how to use SDXL for text-to-image, image-to-image, and inpainting.

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries in Colab
#!pip install transformers accelerate safetensors invisible-watermark>=0.2.0
```

<Tip warning={true}>

We recommend installing the [invisible-watermark](https://pypi.org/project/invisible-watermark/) library to help identify images that are generated. If the invisible-watermark library is installed, it is used by default. To disable the watermarker:

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(..., add_watermarker=False)
```

</Tip>

## Load single file formats

Use the [`~StableDiffusionXLPipeline.from_single_file`] method to load single file formats (`.ckpt` or `.safetensors`) into 🤗 Diffusers (otherwise you can use [`~StableDiffusionXLPipeline.from_pretrained`]):

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
    "./sd_xl_base_1.0.safetensors", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
    "./sd_xl_refiner_1.0.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")
```

## Text-to-image

For text-to-image, pass a text prompt:

```py
from diffusers import AutoPipeline
import torch

pipeline_text2image = AutoPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt=prompt).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png" alt="generated image of an astronaut in a jungle"/>
</div>

## Image-to-image

For image-to-image, SDXL works especially well with image sizes between 768x768 and 1024x1024. Pass an initial image, and a text prompt to condition the image with:

```py
from diffusers import AutoPipeline
from diffusers.utils import load_image

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = AutoPipeline.from_pipe(pipeline_text2image).to("cuda")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-img2img.png"

init_image = load_image(url).convert("RGB")
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-img2img.png" alt="generated image of a dog catching a frisbee in a jungle"/>
</div>

## Inpainting

For inpainting, you'll need the original image and a mask of what you want to replace in the original image. Create a prompt to describe what you want to replace the masked area with.

```py
from diffusers import AutoPipeline
from diffusers.utils import load_image

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = AutoPipeline.from_pipe(pipeline_text2image).to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A deep sea diver floating"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint.png" alt="generated image of a deep sea diver in a jungle"/>
</div>

## Refine image quality

SDXL includes a [refiner model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) specialized in denoising low-noise stage images to generate higher-quality images from the base model. There are two ways to use the refiner:

1. use the base and refiner model as an [*ensemble of expert denoisers*](https://research.nvidia.com/labs/dir/eDiff-I/) (❤️ thanks to the following contributors for proposing and implementing this method: [SytanSD](https://github.com/SytanSD), [bghira](https://github.com/bghira), [Birch-san](https://github.com/Birch-san), [AmericanPresidentJimmyCarter](https://github.com/AmericanPresidentJimmyCarter))
2. use the refiner with [SDEdit](https://huggingface.co/papers/2108.01073) after running the base model (this is how SDXL is originally trained)

### Ensemble of expert denoisers

The ensemble of expert denoisers approach requires less overall denoising steps versus passing the base model's output to the refiner model, so it should be significantly faster to run. However, you won't be able to inspect the base model's output because it is heavily denoised.

As an ensemble of expert denoisers, the base model serves as the expert during the high-noise diffusion stage and the refiner model serves as the expert during the low-noise diffusion stage. Load the base and refiner model:

```py
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
```

To use this approach, you need to define the number of timesteps for each model to run through their respective stages. For the base model, this is controlled by the [`denoising_end`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.denoising_end) parameter and for the refiner model, it is controlled by the [`denoising_start`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline.__call__.denoising_start) parameter.

<Tip>

The `denoising_end` and `denoising_start` parameters should be a float between 0 and 1. These parameters are represented as a proportion of discrete timesteps as defined by the scheduler. If you're also using the `strength` parameter, it'll be ignored because the number of denoising steps is determined by the discrete timesteps the model is trained on and the declared fractional cutoff.

</Tip>

Let's set `denoising_end=0.8` so the base model performs the first 80% of denoising the **high-noise** timesteps and set `denoising_start=0.8` so the refiner model performs the last 20% of denoising the **low-noise** timesteps. The base model output should be in **latent** space instead of a PIL image.

```py
prompt = "A majestic lion jumping from a big stone at night"

image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=40,
    denoising_start=0.8,
    image=image,
).images[0]
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_base.png" alt="generated image of a lion on a rock at night" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">base model</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_refined.png" alt="generated image of a lion on a rock at night in higher quality" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">ensemble of expert denoisers</figcaption>
  </div>
</div>

For inpainting, use the [`StableDiffusionXLInpaintPipeline`]:

```py
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

base = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
num_inference_steps = 75
high_noise_frac = 0.7

image = base(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
).images[0]
```

This ensemble of expert denoisers method works well for all available schedulers!

### Refine fully-denoised base image

SDXL gets a boost in image quality by using the refiner model to add additional high-quality details to the fully-denoised image from the base model, similar to image-to-image generation.

Load the base and refiner models:

```py
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
```

Generate an image from the base model, and set the model output to **latent** space:

```py
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil").images[0]
```

Pass the generated image to the refiner model:

```py
image = refiner(prompt=prompt, image=image[None, :]).images[0]
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/init_image.png" alt="generated image of an astronaut riding a green horse on Mars" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">base model</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_image.png" alt="higher quality generated image of an astronaut riding a green horse on Mars" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">base model + refiner model</figcaption>
  </div>
</div>

For inpainting, use the [`StableDiffusionXLInpaintPipeline`], remove the `denoising_end` and `denoising_start` parameters, and choose a smaller number of inference steps for the refiner.

## Use a different prompt for each text-encoder

SDXL uses two text-encoders so it is possible to pass a different prompt to each text-encoder which can [improve quality](https://github.com/huggingface/diffusers/issues/4004#issuecomment-1627764201). Pass your original prompt to `prompt` and the second prompt to `prompt_2` (use `negative_prompt` and `negative_prompt_2` if you're using a negative prompt):

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

# prompt is passed to OAI CLIP-ViT/L-14
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt_2 is passed to OpenCLIP-ViT/bigG-14
prompt_2 = "Van Gogh painting"
image = pipeline(prompt=prompt, prompt_2=prompt_2).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-double-prompt.png" alt="generated image of an astronaut in a jungle in the style of a van gogh painting"/>
</div>

## Cropped image generation

Images generated from previous Stable Diffusion models may sometimes appear to be randomly cropped due to how the model is trained. By conditioning SDXL on the cropping parameters, SDXL is able to generate images that are more centered and subjects in the images aren't randomly cut off. You can control the amount of cropping during inference with the [`crops_coords_top_left`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.crops_coords_top_left) parameter. By default, `crops_coords_top_left` is (0, 0) for a centered image.

```py
from diffusers import StableDiffusionXLPipeline
import torch


pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt=prompt, crops_coords_top_left=(256,0)).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-cropped.png" alt="generated image of an astronaut in a jungle, slightly cropped"/>
</div>

## Optimizations

SDXL is a large model, and you may need to optimize your memory to get it to run on hardware. Here are some tips to save memory and speed up inference.

1. Offload the model to the CPU with [`~StableDiffusionXLPipeline.enable_model_cpu_offload`] for out-of-memory errors:

```diff
- base.to("cuda")
- refiner.to("cuda")
+ base.enable_model_cpu_offload
+ refiner.enable_model_cpu_offload
```

2. Use `torch.compile` for ~20% speed-up (you need `torch>2.0`):

```diff
+ base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
+ refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
```

3. Enable [xFormers](/optimization/xformers) to run SDXL if `torch<2.0`:

```diff
+ base.enable_xformers_memory_efficient_attention()
+ refiner.enable_xformers_memory_efficient_attention()
```

## Other resources

If you're interested in experimenting with a minimal version of the [`UNet2DConditionModel`] used in SDXL, take a look at the [minSDXL](https://github.com/cloneofsimo/minSDXL) implementation which is written in PyTorch and directly compatible with 🤗 Diffusers.