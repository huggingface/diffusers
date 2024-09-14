<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Flux

Flux is a series of text-to-image generation models based on diffusion transformers. To know more about Flux, check out the original [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/) by the creators of Flux, Black Forest Labs.

Original model checkpoints for Flux can be found [here](https://huggingface.co/black-forest-labs). Original inference code can be found [here](https://github.com/black-forest-labs/flux).

<Tip>

Flux can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details. Additionally, Flux can benefit from quantization for memory efficiency with a trade-off in inference latency. Refer to [this blog post](https://huggingface.co/blog/quanto-diffusers) to learn more. For an exhaustive list of resources, check out [this gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c).

</Tip>

Flux comes in two variants:

- Timestep-distilled (`black-forest-labs/FLUX.1-schnell`)
- Guidance-distilled (`black-forest-labs/FLUX.1-dev`)

Both checkpoints have slightly difference usage which we detail below.

### Timestep-distilled

- `max_sequence_length` cannot be more than 256.
- `guidance_scale` needs to be 0.
- As this is a timestep-distilled model, it benefits from fewer sampling steps.

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("image.png")
```

### Guidance-distilled

- The guidance-distilled variant takes about 50 sampling steps for good-quality generation.
- It doesn't have any limitations around the `max_sequence_length`.

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "a tiny astronaut hatching from an egg on the moon"
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
).images[0]
out.save("image.png")
```

## Running FP16 inference

Flux can generate high-quality images with FP16 (i.e. to accelerate inference on Turing/Volta GPUs) but produces different outputs compared to FP32/BF16. The issue is that some activations in the text encoders have to be clipped when running in FP16, which affects the overall image. Forcing text encoders to run with FP32 inference thus removes this output difference. See [here](https://github.com/huggingface/diffusers/pull/9097#issuecomment-2272292516) for details.

FP16 inference code:

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16) # can replace schnell with dev
# to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("image.png")
```

## Single File Loading for the `FluxTransformer2DModel`

The `FluxTransformer2DModel` supports loading checkpoints in the original format shipped by Black Forest Labs. This is also useful when trying to load finetunes or quantized versions of the models that have been published by the community.

<Tip>
`FP8` inference can be brittle depending on the GPU type, CUDA version, and `torch` version that you are using. It is recommended that you use the `optimum-quanto` library in order to run FP8 inference on your machine.
</Tip>

The following example demonstrates how to run Flux with less than 16GB of VRAM.

First install `optimum-quanto`

```shell
pip install optimum-quanto
```

Then run the following example

```python
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", torch_dtype=dtype)
quantize(transformer, weights=qfloat8)
freeze(transformer)

text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=3.5,
    output_type="pil",
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("flux-fp8-dev.png")
```

## FluxPipeline

[[autodoc]] FluxPipeline - all - **call**

## FluxImg2ImgPipeline

[[autodoc]] FluxImg2ImgPipeline - all - **call**

## FluxInpaintPipeline

[[autodoc]] FluxInpaintPipeline - all - **call**

## Flux ControlNet Inpaint Pipeline

The Flux ControlNet Inpaint pipeline is designed for controllable image inpainting using the Flux architecture.

### Overview

This pipeline combines the power of Flux's transformer-based architecture with ControlNet conditioning and inpainting capabilities. It allows for guided image generation within specified masked areas of an input image.

### Usage

```python
import torch
from diffusers import FluxControlNetInpaintPipeline
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=torch.bfloat16
)

pipe = FluxControlNetInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", controlnet=controlnet, torch_dtype=torch.bfloat16
)

pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
pipe.to(device)

control_image = load_image(
    "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg"
)
init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
)
mask_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
)

prompt = "A girl holding a sign that says InstantX"
image = pipe(
    prompt,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
    controlnet_conditioning_scale=0.7,
    strength=0.7,
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]

image.save("flux_controlnet_inpaint.png")
```

## Flux ControlNet Image to Image Pipeline

The Flux ControlNet Img2Img pipeline enables controllable image-to-image translation using the Flux architecture combined with ControlNet conditioning.

### Overview

This pipeline allows for the transformation of input images based on text prompts and ControlNet conditions. It leverages the Flux transformer-based architecture to generate high-quality output images while maintaining control over the generation process.

### Usage

```python
import torch
from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=torch.bfloat16
)

pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
pipe.to(device)

control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
)

prompt = "A girl in city, 25 years old, cool, futuristic"
image = pipe(
    prompt,
    image=init_image,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    strength=0.7,
    num_inference_steps=2,
    guidance_scale=3.5,
).images[0]

image.save("flux_controlnet_img2img.png")
```
