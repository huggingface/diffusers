<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Working with big models

A modern diffusion model, like [Stable Diffusion XL (SDXL)](../using-diffusers/sdxl), is not just a single model, but a collection of multiple models. SDXL has four different model-level components:

* A variational autoencoder (VAE)
* Two text encoders
* A UNet for denoising

Usually, the text encoders and the denoiser are much larger compared to the VAE.

As models get bigger and better, it’s possible your model is so big that even a single copy won’t fit in memory. But that doesn’t mean it can’t be loaded. If you have more than one GPU, there is more memory available to store your model. In this case, it’s better to split your model checkpoint into several smaller *checkpoint shards*.

When a text encoder checkpoint has multiple shards, like [T5-xxl for SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main/text_encoder_3), it is automatically handled by the [Transformers](https://huggingface.co/docs/transformers/index) library as it is a required dependency of Diffusers when using the [`StableDiffusion3Pipeline`]. More specifically, Transformers will automatically handle the loading of multiple shards within the requested model class and get it ready so that inference can be performed.

The denoiser checkpoint can also have multiple shards and supports inference thanks to the [Accelerate](https://huggingface.co/docs/accelerate/index) library.

> [!TIP]
> Refer to the [Handling big models for inference](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference) guide for general guidance when working with big models that are hard to fit into memory.

For example, let's save a sharded checkpoint for the [SDXL UNet](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/unet):

```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
)
unet.save_pretrained("sdxl-unet-sharded", max_shard_size="5GB")
```

The size of the fp32 variant of the SDXL UNet checkpoint is ~10.4GB. Set the `max_shard_size` parameter to 5GB to create 3 shards. After saving, you can load them in [`StableDiffusionXLPipeline`]:

```python
from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline
import torch

unet = UNet2DConditionModel.from_pretrained(
    "sayakpaul/sdxl-unet-sharded", torch_dtype=torch.float16
)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
).to("cuda")

image = pipeline("a cute dog running on the grass", num_inference_steps=30).images[0]
image.save("dog.png")
```

If placing all the model-level components on the GPU at once is not feasible, use [`~DiffusionPipeline.enable_model_cpu_offload`] to help you:

```diff
- pipeline.to("cuda")
+ pipeline.enable_model_cpu_offload()
```

In general, we recommend sharding when a checkpoint is more than 5GB (in fp32).

## Device placement

On distributed setups, you can run inference across multiple GPUs with Accelerate.

> [!WARNING]
> This feature is experimental and its APIs might change in the future.

With Accelerate, you can use the `device_map` to determine how to distribute the models of a pipeline across multiple devices. This is useful in situations where you have more than one GPU.

For example, if you have two 8GB GPUs, then using [`~DiffusionPipeline.enable_model_cpu_offload`] may not work so well because:

* it only works on a single GPU
* a single model might not fit on a single GPU ([`~DiffusionPipeline.enable_sequential_cpu_offload`] might work but it will be extremely slow and it is also limited to a single GPU)

To make use of both GPUs, you can use the "balanced" device placement strategy which splits the models across all available GPUs.

> [!WARNING]
> Only the "balanced" strategy is supported at the moment, and we plan to support additional mapping strategies in the future.

```diff
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
-    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
+    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, device_map="balanced"
)
image = pipeline("a dog").images[0]
image
```

You can also pass a dictionary to enforce the maximum GPU memory that can be used on each device:

```diff
from diffusers import DiffusionPipeline
import torch

max_memory = {0:"1GB", 1:"1GB"}
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    device_map="balanced",
+   max_memory=max_memory
)
image = pipeline("a dog").images[0]
image
```

If a device is not present in `max_memory`, then it will be completely ignored and will not participate in the device placement.

By default, Diffusers uses the maximum memory of all devices. If the models don't fit on the GPUs, they are offloaded to the CPU. If the CPU doesn't have enough memory, then you might see an error. In that case, you could defer to using [`~DiffusionPipeline.enable_sequential_cpu_offload`] and [`~DiffusionPipeline.enable_model_cpu_offload`].

Call [`~DiffusionPipeline.reset_device_map`] to reset the `device_map` of a pipeline. This is also necessary if you want to use methods like `to()`, [`~DiffusionPipeline.enable_sequential_cpu_offload`], and [`~DiffusionPipeline.enable_model_cpu_offload`] on a pipeline that was device-mapped.

```py
pipeline.reset_device_map()
```

Once a pipeline has been device-mapped, you can also access its device map via `hf_device_map`:

```py
print(pipeline.hf_device_map)
```

An example device map would look like so:


```bash
{'unet': 1, 'vae': 1, 'safety_checker': 0, 'text_encoder': 0}
```