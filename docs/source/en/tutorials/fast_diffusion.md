<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Accelerate inference of text-to-image diffusion models

Diffusion models are slower than their GAN counterparts because of the iterative and sequential reverse diffusion process. There are several techniques that can address this limitation such as progressive timestep distillation ([LCM LoRA](../using-diffusers/inference_with_lcm_lora)), model compression ([SSD-1B](https://huggingface.co/segmind/SSD-1B)), and reusing adjacent features of the denoiser ([DeepCache](../optimization/deepcache)).

However, you don't necessarily need to use these techniques to speed up inference. With PyTorch 2 alone, you can accelerate the inference latency of text-to-image diffusion pipelines by up to 3x. This tutorial will show you how to progressively apply the optimizations found in PyTorch 2 to reduce inference latency. You'll use the [Stable Diffusion XL (SDXL)](../using-diffusers/sdxl) pipeline in this tutorial, but these techniques are applicable to other text-to-image diffusion pipelines too.

Make sure you're using the latest version of Diffusers:

```bash
pip install -U diffusers
```

Then upgrade the other required libraries too:

```bash
pip install -U transformers accelerate peft
```

Install [PyTorch nightly](https://pytorch.org/) to benefit from the latest and fastest kernels:

```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

> [!TIP]
> The results reported below are from a 80GB 400W A100 with its clock rate set to the maximum. 
> If you're interested in the full benchmarking code, take a look at [huggingface/diffusion-fast](https://github.com/huggingface/diffusion-fast).


## Baseline

Let's start with a baseline. Disable reduced precision and the [`scaled_dot_product_attention` (SDPA)](../optimization/torch2.0#scaled-dot-product-attention) function which is automatically used by Diffusers:

```python
from diffusers import StableDiffusionXLPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
).to("cuda")

# Run the attention ops without SDPA.
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

This default setup takes 7.36 seconds.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_0.png" width=500>
</div>

## bfloat16

Enable the first optimization, reduced precision or more specifically bfloat16. There are several benefits of using reduced precision:

* Using a reduced numerical precision (such as float16 or bfloat16) for inference doesn’t affect the generation quality but significantly improves latency.
* The benefits of using bfloat16 compared to float16 are hardware dependent, but modern GPUs tend to favor bfloat16.
* bfloat16 is much more resilient when used with quantization compared to float16, but more recent versions of the quantization library ([torchao](https://github.com/pytorch-labs/ao)) we used don't have numerical issues with float16.

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# Run the attention ops without SDPA.
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

bfloat16 reduces the latency from 7.36 seconds to 4.63 seconds.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_1.png" width=500>
</div>

<Tip>

In our later experiments with float16, recent versions of torchao do not incur numerical problems from float16.

</Tip>

Take a look at the [Speed up inference](../optimization/fp16) guide to learn more about running inference with reduced precision.

## SDPA

Attention blocks are intensive to run. But with PyTorch's [`scaled_dot_product_attention`](../optimization/torch2.0#scaled-dot-product-attention) function, it is a lot more efficient. This function is used by default in Diffusers so you don't need to make any changes to the code.

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

Scaled dot product attention improves the latency from 4.63 seconds to 3.31 seconds.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_2.png" width=500>
</div>

## torch.compile

PyTorch 2 includes `torch.compile` which uses fast and optimized kernels. In Diffusers, the UNet and VAE are usually compiled because these are the most compute-intensive modules. First, configure a few compiler flags (refer to the [full list](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py) for more options):

```python
from diffusers import StableDiffusionXLPipeline
import torch

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
```

It is also important to change the UNet and VAE's memory layout to "channels_last" when compiling them to ensure maximum speed.

```python
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
```

Now compile and perform inference:

```python
# Compile the UNet and VAE.
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# First call to `pipe` is slow, subsequent ones are faster.
image = pipe(prompt, num_inference_steps=30).images[0]
```

`torch.compile` offers different backends and modes. For maximum inference speed, use "max-autotune" for the inductor backend. “max-autotune” uses CUDA graphs and optimizes the compilation graph specifically for latency. CUDA graphs greatly reduces the overhead of launching GPU operations by using a mechanism to launch multiple GPU operations through a single CPU operation.

Using SDPA attention and compiling both the UNet and VAE cuts the latency from 3.31 seconds to 2.54 seconds.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_3.png" width=500>
</div>

> [!TIP]
> From PyTorch 2.3.1, you can control the caching behavior of `torch.compile()`. This is particularly beneficial for compilation modes like `"max-autotune"` which performs a grid-search over several compilation flags to find the optimal configuration. Learn more in the [Compile Time Caching in torch.compile](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) tutorial. 

### Prevent graph breaks

Specifying `fullgraph=True` ensures there are no graph breaks in the underlying model to take full advantage of `torch.compile` without any performance degradation. For the UNet and VAE, this means changing how you access the return variables.

```diff
- latents = unet(
-   latents, timestep=timestep, encoder_hidden_states=prompt_embeds
-).sample

+ latents = unet(
+   latents, timestep=timestep, encoder_hidden_states=prompt_embeds, return_dict=False
+)[0]
```

### Remove GPU sync after compilation

During the iterative reverse diffusion process, the `step()` function is [called](https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1228) on the scheduler each time after the denoiser predicts the less noisy latent embeddings. Inside `step()`, the `sigmas` variable is [indexed](https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/schedulers/scheduling_euler_discrete.py#L476) which when placed on the GPU, causes a communication sync between the CPU and GPU. This introduces latency and it becomes more evident when the denoiser has already been compiled.

But if the `sigmas` array always [stays on the CPU](https://github.com/huggingface/diffusers/blob/35a969d297cba69110d175ee79c59312b9f49e1e/src/diffusers/schedulers/scheduling_euler_discrete.py#L240), the CPU and GPU sync doesn’t occur and you don't get any latency. In general, any CPU and GPU communication sync should be none or be kept to a bare minimum because it can impact inference latency.

## Combine the attention block's projection matrices

The UNet and VAE in SDXL use Transformer-like blocks which consists of attention blocks and feed-forward blocks.

In an attention block, the input is projected into three sub-spaces using three different projection matrices – Q, K, and V. These projections are performed separately on the input. But we can horizontally combine the projection matrices into a single matrix and perform the projection in one step. This increases the size of the matrix multiplications of the input projections and improves the impact of quantization.

You can combine the projection matrices with just a single line of code:

```python
pipe.fuse_qkv_projections()
```

This provides a minor improvement from 2.54 seconds to 2.52 seconds.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_4.png" width=500>
</div>

<Tip warning={true}>

Support for [`~StableDiffusionXLPipeline.fuse_qkv_projections`] is limited and experimental. It's not available for many non-Stable Diffusion pipelines such as [Kandinsky](../using-diffusers/kandinsky). You can refer to this [PR](https://github.com/huggingface/diffusers/pull/6179) to get an idea about how to enable this for the other pipelines.

</Tip>

## Dynamic quantization

You can also use the ultra-lightweight PyTorch quantization library, [torchao](https://github.com/pytorch-labs/ao) (commit SHA `54bcd5a10d0abbe7b0c045052029257099f83fd9`), to apply [dynamic int8 quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) to the UNet and VAE. Quantization adds additional conversion overhead to the model that is hopefully made up for by faster matmuls (dynamic quantization). If the matmuls are too small, these techniques may degrade performance.

First, configure all the compiler tags:

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Notice the two new flags at the end.
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
```

Certain linear layers in the UNet and VAE don’t benefit from dynamic int8 quantization. You can filter out those layers with the [`dynamic_quant_filter_fn`](https://github.com/huggingface/diffusion-fast/blob/0f169640b1db106fe6a479f78c1ed3bfaeba3386/utils/pipeline_utils.py#L16) shown below.

```python
def dynamic_quant_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Linear)
        and mod.in_features > 16
        and (mod.in_features, mod.out_features)
        not in [
            (1280, 640),
            (1920, 1280),
            (1920, 640),
            (2048, 1280),
            (2048, 2560),
            (2560, 1280),
            (256, 128),
            (2816, 1280),
            (320, 640),
            (512, 1536),
            (512, 256),
            (512, 512),
            (640, 1280),
            (640, 1920),
            (640, 320),
            (640, 5120),
            (640, 640),
            (960, 320),
            (960, 640),
        ]
    )


def conv_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
    )
```

Finally, apply all the optimizations discussed so far:

```python
# SDPA + bfloat16.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# Combine attention projection matrices.
pipe.fuse_qkv_projections()

# Change the memory layout.
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
```

Since dynamic quantization is only limited to the linear layers, convert the appropriate pointwise convolution layers into linear layers to maximize its benefit.

```python
from torchao import swap_conv2d_1x1_to_linear

swap_conv2d_1x1_to_linear(pipe.unet, conv_filter_fn)
swap_conv2d_1x1_to_linear(pipe.vae, conv_filter_fn)
```

Apply dynamic quantization:

```python
from torchao import apply_dynamic_quant

apply_dynamic_quant(pipe.unet, dynamic_quant_filter_fn)
apply_dynamic_quant(pipe.vae, dynamic_quant_filter_fn)
```

Finally, compile and perform inference:

```python
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

Applying dynamic quantization improves the latency from 2.52 seconds to 2.43 seconds.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_5.png" width=500>
</div>
