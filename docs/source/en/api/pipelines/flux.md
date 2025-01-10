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

Flux can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details. Additionally, Flux can benefit from quantization for memory efficiency with a trade-off in inference latency. Refer to [this blog post](https://huggingface.co/blog/quanto-diffusers) to learn more.  For an exhaustive list of resources, check out [this gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c).

</Tip>

Flux comes in the following variants:

| model type | model id |
|:----------:|:--------:|
| Timestep-distilled | [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| Guidance-distilled | [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Fill Inpainting/Outpainting (Guidance-distilled) | [`black-forest-labs/FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) |
| Canny Control (Guidance-distilled) | [`black-forest-labs/FLUX.1-Canny-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) |
| Depth Control (Guidance-distilled) | [`black-forest-labs/FLUX.1-Depth-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) |
| Canny Control (LoRA) | [`black-forest-labs/FLUX.1-Canny-dev-lora`](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora) |
| Depth Control (LoRA) | [`black-forest-labs/FLUX.1-Depth-dev-lora`](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora) |
| Redux (Adapter) | [`black-forest-labs/FLUX.1-Redux-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) |

All checkpoints have different usage which we detail below.

### Timestep-distilled

* `max_sequence_length` cannot be more than 256.
* `guidance_scale` needs to be 0.
* As this is a timestep-distilled model, it benefits from fewer sampling steps.

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

* The guidance-distilled variant takes about 50 sampling steps for good-quality generation.
* It doesn't have any limitations around the `max_sequence_length`.

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

### Fill Inpainting/Outpainting

* Flux Fill pipeline does not require `strength` as an input like regular inpainting pipelines.
* It supports both inpainting and outpainting.

```python
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup_mask.png")

repo_id = "black-forest-labs/FLUX.1-Fill-dev"
pipe = FluxFillPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to("cuda")

image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"output.png")
```

### Canny Control

**Note:** `black-forest-labs/Flux.1-Canny-dev` is _not_ a [`ControlNetModel`] model. ControlNet models are a separate component from the UNet/Transformer whose residuals are added to the actual underlying model. Canny Control is an alternate architecture that achieves effectively the same results as a ControlNet model would, by using channel-wise concatenation with input control condition and ensuring the transformer learns structure control by following the condition as closely as possible. 

```python
# !pip install -U controlnet-aux
import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda")

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = CannyDetector()
control_image = processor(control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
).images[0]
image.save("output.png")
```

Canny Control is also possible with a LoRA variant of this condition. The usage is as follows:

```python
# !pip install -U controlnet-aux
import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora")

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = CannyDetector()
control_image = processor(control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
).images[0]
image.save("output.png")
```

### Depth Control

**Note:** `black-forest-labs/Flux.1-Depth-dev` is _not_ a ControlNet model. [`ControlNetModel`] models are a separate component from the UNet/Transformer whose residuals are added to the actual underlying model. Depth Control is an alternate architecture that achieves effectively the same results as a ControlNet model would, by using channel-wise concatenation with input control condition and ensuring the transformer learns structure control by following the condition as closely as possible.

```python
# !pip install git+https://github.com/huggingface/image_gen_aux
import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("output.png")
```

Depth Control is also possible with a LoRA variant of this condition. The usage is as follows:

```python
# !pip install git+https://github.com/huggingface/image_gen_aux
import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora")

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("output.png")
```

### Redux

* Flux Redux pipeline is an adapter for FLUX.1 base models. It can be used with both flux-dev and flux-schnell, for image-to-image generation.
* You can first use the `FluxPriorReduxPipeline` to get the `prompt_embeds` and `pooled_prompt_embeds`, and then feed them into the `FluxPipeline` for image-to-image generation.
* When use `FluxPriorReduxPipeline` with a base pipeline, you can set `text_encoder=None` and `text_encoder_2=None` in the base pipeline, in order to save VRAM.

```python
import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers.utils import load_image
device = "cuda"
dtype = torch.bfloat16


repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
repo_base = "black-forest-labs/FLUX.1-dev" 
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=dtype).to(device)
pipe = FluxPipeline.from_pretrained(
    repo_base, 
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=torch.bfloat16
).to(device)

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy/img5.png")
pipe_prior_output = pipe_prior_redux(image)
images = pipe(
    guidance_scale=2.5,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_prior_output,
).images
images[0].save("flux-redux.png")
```

## Combining Flux Turbo LoRAs with Flux Control, Fill, and Redux

We can combine Flux Turbo LoRAs with Flux Control and other pipelines like Fill and Redux to enable few-steps' inference. The example below shows how to do that for Flux Control LoRA for depth and turbo LoRA from [`ByteDance/Hyper-SD`](https://hf.co/ByteDance/Hyper-SD).

```py
from diffusers import FluxControlPipeline
from image_gen_aux import DepthPreprocessor
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
import torch

control_pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
control_pipe.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora", adapter_name="depth")
control_pipe.load_lora_weights(
    hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), adapter_name="hyper-sd"
)
control_pipe.set_adapters(["depth", "hyper-sd"], adapter_weights=[0.85, 0.125])
control_pipe.enable_model_cpu_offload()

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = control_pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("output.png")
```

## Note about `unload_lora_weights()` when using Flux LoRAs

When unloading the Control LoRA weights, call `pipe.unload_lora_weights(reset_to_overwritten_params=True)` to reset the `pipe.transformer` completely back to its original form. The resultant pipeline can then be used with methods like [`DiffusionPipeline.from_pipe`]. More details about this argument are available in [this PR](https://github.com/huggingface/diffusers/pull/10397).

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

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`FluxPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt, guidance_scale=3.5, height=768, width=1360, num_inference_steps=50).images[0]
image.save("flux.png")
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

[[autodoc]] FluxPipeline
	- all
	- __call__

## FluxImg2ImgPipeline

[[autodoc]] FluxImg2ImgPipeline
	- all
	- __call__

## FluxInpaintPipeline

[[autodoc]] FluxInpaintPipeline
	- all
	- __call__


## FluxControlNetInpaintPipeline

[[autodoc]] FluxControlNetInpaintPipeline
	- all
	- __call__

## FluxControlNetImg2ImgPipeline

[[autodoc]] FluxControlNetImg2ImgPipeline
	- all
	- __call__

## FluxControlPipeline

[[autodoc]] FluxControlPipeline
	- all
	- __call__

## FluxControlImg2ImgPipeline

[[autodoc]] FluxControlImg2ImgPipeline
	- all
	- __call__

## FluxPriorReduxPipeline

[[autodoc]] FluxPriorReduxPipeline
	- all
	- __call__

## FluxFillPipeline

[[autodoc]] FluxFillPipeline
	- all
	- __call__
