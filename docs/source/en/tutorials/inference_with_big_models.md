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

When a text encoder checkpoint has multiple shards, like [T5-xxl for SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main/text_encoder_3), it is automatically handled by the [Transformers](https://huggingface.co/docs/transformers/index) library as it is a required dependency of Diffusers when using the [`StableDiffusion3Pipeline`]. 

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

If you want to distribute the model-level components across multiple GPUs, then using `device_map` when loading a pipeline could be also useful. Refer to [this guide](../training/distributed_inference.md#distributed-inference-with-multiple-gpus) for more details.