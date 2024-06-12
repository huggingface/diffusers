<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reduce memory usage

A barrier to using diffusion models is the large amount of memory required. To overcome this challenge, there are several memory-reducing techniques you can use to run even some of the largest models on free-tier or consumer GPUs. Some of these techniques can even be combined to further reduce memory usage.

<Tip>

In many cases, optimizing for memory or speed leads to improved performance in the other, so you should try to optimize for both whenever you can. This guide focuses on minimizing memory usage, but you can also learn more about how to [Speed up inference](fp16).

</Tip>

The results below are obtained from generating a single 512x512 image from the prompt a photo of an astronaut riding a horse on mars with 50 DDIM steps on a Nvidia Titan RTX, demonstrating the speed-up you can expect as a result of reduced memory consumption.

|                  | latency | speed-up |
| ---------------- | ------- | ------- |
| original         | 9.50s   | x1      |
| fp16             | 3.61s   | x2.63   |
| channels last    | 3.30s   | x2.88   |
| traced UNet      | 3.21s   | x2.96   |
| memory-efficient attention  | 2.63s  | x3.61   |

## Sliced VAE

Sliced VAE enables decoding large batches of images with limited VRAM or batches with 32 images or more by decoding the batches of latents one image at a time. You'll likely want to couple this with [`~ModelMixin.enable_xformers_memory_efficient_attention`] to reduce memory use further if you have xFormers installed.

To use sliced VAE, call [`~StableDiffusionPipeline.enable_vae_slicing`] on your pipeline before inference:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_vae_slicing()
#pipe.enable_xformers_memory_efficient_attention()
images = pipe([prompt] * 32).images
```

You may see a small performance boost in VAE decoding on multi-image batches, and there should be no performance impact on single-image batches.

## Tiled VAE

Tiled VAE processing also enables working with large images on limited VRAM (for example, generating 4k images on 8GB of VRAM) by splitting the image into overlapping tiles, decoding the tiles, and then blending the outputs together to compose the final image. You should also used tiled VAE with [`~ModelMixin.enable_xformers_memory_efficient_attention`] to reduce memory use further if you have xFormers installed.

To use tiled VAE processing, call [`~StableDiffusionPipeline.enable_vae_tiling`] on your pipeline before inference:

```python
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
prompt = "a beautiful landscape photograph"
pipe.enable_vae_tiling()
#pipe.enable_xformers_memory_efficient_attention()

image = pipe([prompt], width=3840, height=2224, num_inference_steps=20).images[0]
```

The output image has some tile-to-tile tone variation because the tiles are decoded separately, but you shouldn't see any sharp and obvious seams between the tiles. Tiling is turned off for images that are 512x512 or smaller.

## CPU offloading

Offloading the weights to the CPU and only loading them on the GPU when performing the forward pass can also save memory. Often, this technique can reduce memory consumption to less than 3GB.

To perform CPU offloading, call [`~StableDiffusionPipeline.enable_sequential_cpu_offload`]:

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_sequential_cpu_offload()
image = pipe(prompt).images[0]
```

CPU offloading works on submodules rather than whole models. This is the best way to minimize memory consumption, but inference is much slower due to the iterative nature of the diffusion process. The UNet component of the pipeline runs several times (as many as `num_inference_steps`); each time, the different UNet submodules are sequentially onloaded and offloaded as needed, resulting in a large number of memory transfers.

<Tip>

Consider using [model offloading](#model-offloading) if you want to optimize for speed because it is much faster. The tradeoff is your memory savings won't be as large.

</Tip>

<Tip warning={true}>

When using [`~StableDiffusionPipeline.enable_sequential_cpu_offload`], don't move the pipeline to CUDA beforehand or else the gain in memory consumption will only be minimal (see this [issue](https://github.com/huggingface/diffusers/issues/1934) for more information).

[`~StableDiffusionPipeline.enable_sequential_cpu_offload`] is a stateful operation that installs hooks on the models.

</Tip>

## Model offloading

<Tip>

Model offloading requires 🤗 Accelerate version 0.17.0 or higher.

</Tip>

[Sequential CPU offloading](#cpu-offloading) preserves a lot of memory but it makes inference slower because submodules are moved to GPU as needed, and they're immediately returned to the CPU when a new module runs.

Full-model offloading is an alternative that moves whole models to the GPU, instead of handling each model's constituent *submodules*. There is a negligible impact on inference time (compared with moving the pipeline to `cuda`), and it still provides some memory savings.

During model offloading, only one of the main components of the pipeline (typically the text encoder, UNet and VAE)
is placed on the GPU while the others wait on the CPU. Components like the UNet that run for multiple iterations stay on the GPU until they're no longer needed.

Enable model offloading by calling [`~StableDiffusionPipeline.enable_model_cpu_offload`] on the pipeline:

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_model_cpu_offload()
image = pipe(prompt).images[0]
```

<Tip warning={true}>

In order to properly offload models after they're called, it is required to run the entire pipeline and models are called in the pipeline's expected order. Exercise caution if models are reused outside the context of the pipeline after hooks have been installed. See [Removing Hooks](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.hooks.remove_hook_from_module) for more information.

[`~StableDiffusionPipeline.enable_model_cpu_offload`] is a stateful operation that installs hooks on the models and state on the pipeline.

</Tip>

## Channels-last memory format

The channels-last memory format is an alternative way of ordering NCHW tensors in memory to preserve dimension ordering. Channels-last tensors are ordered in such a way that the channels become the densest dimension (storing images pixel-per-pixel). Since not all operators currently support the channels-last format, it may result in worst performance but you should still try and see if it works for your model.

For example, to set the pipeline's UNet to use the channels-last format:

```python
print(pipe.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
print(
    pipe.unet.conv_out.state_dict()["weight"].stride()
)  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works
```

## Tracing

Tracing runs an example input tensor through the model and captures the operations that are performed on it as that input makes its way through the model's layers. The executable or `ScriptFunction` that is returned is optimized with just-in-time compilation.

To trace a UNet:

```python
import time
import torch
from diffusers import StableDiffusionPipeline
import functools

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
unet_runs_per_experiment = 50


# load inputs
def generate_inputs():
    sample = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    timestep = torch.rand(1, device="cuda", dtype=torch.float16) * 999
    encoder_hidden_states = torch.randn((2, 77, 768), device="cuda", dtype=torch.float16)
    return sample, timestep, encoder_hidden_states


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
unet = pipe.unet
unet.eval()
unet.to(memory_format=torch.channels_last)  # use channels_last memory format
unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

# warmup
for _ in range(3):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet(*inputs)

# trace
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")


# warmup and optimize graph
for _ in range(5):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet_traced(*inputs)


# benchmarking
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# save the model
unet_traced.save("unet_traced.pt")
```

Replace the `unet` attribute of the pipeline with the traced model:

```python
from diffusers import StableDiffusionPipeline
import torch
from dataclasses import dataclass


@dataclass
class UNet2DConditionOutput:
    sample: torch.Tensor


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

# use jitted unet
unet_traced = torch.jit.load("unet_traced.pt")


# del pipe.unet
class TracedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = pipe.unet.config.in_channels
        self.device = pipe.unet.device

    def forward(self, latent_model_input, t, encoder_hidden_states):
        sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


pipe.unet = TracedUNet()

with torch.inference_mode():
    image = pipe([prompt] * 1, num_inference_steps=50).images[0]
```

## Memory-efficient attention

Recent work on optimizing bandwidth in the attention block has generated huge speed-ups and reductions in GPU memory usage. The most recent type of memory-efficient attention is [Flash Attention](https://arxiv.org/abs/2205.14135) (you can check out the original code at [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)).

<Tip>

If you have PyTorch >= 2.0 installed, you should not expect a speed-up for inference when enabling `xformers`.

</Tip>

To use Flash Attention, install the following:

- PyTorch > 1.12
- CUDA available
- [xFormers](xformers)

Then call [`~ModelMixin.enable_xformers_memory_efficient_attention`] on the pipeline:

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode():
    sample = pipe("a small cat")

# optional: You can disable it via
# pipe.disable_xformers_memory_efficient_attention()
```

The iteration speed when using `xformers` should match the iteration speed of PyTorch 2.0 as described [here](torch2.0).
