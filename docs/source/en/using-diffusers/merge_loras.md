<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Merge LoRAs

It can be fun and creative to use multiple [LoRAs]((https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora)) together to generate something entirely new and unique. This works by merging multiple LoRA weights together to produce images that are a blend of different styles. Diffusers provides a few methods to merge LoRAs depending on *how* you want to merge their weights, which can affect image quality.

This guide will show you how to merge LoRAs using the [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] and [`~peft.LoraModel.add_weighted_adapter`] methods. To improve inference speed and reduce memory-usage of merged LoRAs, you'll also see how to use the [`~loaders.LoraLoaderMixin.fuse_lora`] method to fuse the LoRA weights with the original weights of the underlying model.

For this guide, load a Stable Diffusion XL (SDXL) checkpoint and the [KappaNeuro/studio-ghibli-style]() and [Norod78/sdxl-chalkboarddrawing-lora]() LoRAs with the [`~loaders.LoraLoaderMixin.load_lora_weights`] method. You'll need to assign each LoRA an `adapter_name` to combine them later.

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")
```

## set_adapters

The [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] method merges LoRA adapters by concatenating their weighted matrices. Use the adapter name to specify which LoRAs to merge, and the `adapter_weights` parameter to control the scaling for each LoRA. For example, if `adapter_weights=[0.5, 0.5]`, then the merged LoRA output is an average of both LoRAs. Try adjusting the adapter weights to see how it affects the generated image!

```py
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])

generator = torch.manual_seed(0)
prompt = "A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai"
image = pipeline(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0}).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lora_merge_set_adapters.png"/>
</div>

## add_weighted_adapter

> [!WARNING]
> This is an experimental method that adds PEFTs [`~peft.LoraModel.add_weighted_adapter`] method to Diffusers to enable more efficient merging methods. Check out this [issue](https://github.com/huggingface/diffusers/issues/6892) if you're interested in learning more about the motivation and design behind this integration.

The [`~peft.LoraModel.add_weighted_adapter`] method provides access to more efficient merging method such as [TIES and DARE](https://huggingface.co/docs/peft/developer_guides/model_merging). To use these merging methods, make sure you have the latest stable version of Diffusers and PEFT installed.

```bash
pip install -U diffusers peft
```

There are three steps to merge LoRAs with the [`~peft.LoraModel.add_weighted_adapter`] method:

1. Create a [`~peft.PeftModel`] from the underlying model and LoRA checkpoint.
2. Load a base UNet model and the LoRA adapters.
3. Merge the adapters using the [`~peft.LoraModel.add_weighted_adapter`] method and the merging method of your choice.

Let's dive deeper into what these steps entail.

1. Load a UNet that corresponds to the UNet in the LoRA checkpoint. In this case, both LoRAs use the SDXL UNet as their base model.

```python
from diffusers import UNet2DConditionModel
import torch

unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")
```

Load the SDXL pipeline and the LoRA checkpoints, starting with the [ostris/ikea-instructions-lora-sdxl](https://huggingface.co/ostris/ikea-instructions-lora-sdxl) LoRA.

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    unet=unet
).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
```

Now you'll create a [`~peft.PeftModel`] from the loaded LoRA checkpoint by combining the SDXL UNet and the LoRA UNet from the pipeline.

```python
from peft import get_peft_model, LoraConfig
import copy

sdxl_unet = copy.deepcopy(unet)
ikea_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["ikea"],
    adapter_name="ikea"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipeline.unet.state_dict().items()}
ikea_peft_model.load_state_dict(original_state_dict, strict=True)
```

> [!TIP]
> You can optionally push the ikea_peft_model to the Hub by calling `ikea_peft_model.push_to_hub("ikea_peft_model", token=TOKEN)`.

Repeat this process to create a [`~peft.PeftModel`] from the [lordjia/by-feng-zikai](https://huggingface.co/lordjia/by-feng-zikai) LoRA.

```python
pipeline.delete_adapters("ikea")
sdxl_unet.delete_adapters("ikea")

pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")
pipeline.set_adapters(adapter_names="feng")

feng_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["feng"],
    adapter_name="feng"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}
feng_peft_model.load_state_dict(original_state_dict, strict=True)
```

2. Load a base UNet model and then load the adapters onto it.

```python
from peft import PeftModel

base_unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")

model = PeftModel.from_pretrained(base_unet, "stevhliu/ikea_peft_model", use_safetensors=True, subfolder="ikea", adapter_name="ikea")
model.load_adapter("stevhliu/feng_peft_model", use_safetensors=True, subfolder="feng", adapter_name="feng")
```

3. Merge the adapters using the [`~peft.LoraModel.add_weighted_adapter`] method and the merging method of your choice (learn more about other merging methods in this [blog post](https://huggingface.co/blog/peft_merging)). For this example, let's use the `"dare_linear"` method to merge the LoRAs.

> [!WARNING]
> Keep in mind the LoRAs need to have the same rank to be merged!

```python
model.add_weighted_adapter(
    adapters=["ikea", "feng"],
    weights=[1.0, 1.0],
    combination_type="dare_linear",
    adapter_name="ikea-feng"
)
model.set_adapters("ikea-feng")
```

Now you can generate an image with the merged LoRA.

```python
model = model.to(dtype=torch.float16, device="cuda")

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=model, variant="fp16", torch_dtype=torch.float16,
).to("cuda")

image = pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", generator=torch.manual_seed(0)).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ikea-feng-dare-linear.png"/>
</div>

## fuse_lora

Both the [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] and [`~peft.LoraModel.add_weighted_adapter`] methods require loading the base model and the LoRA adapters separately which incurs some overhead. The [`~loaders.LoraLoaderMixin.fuse_lora`] method allows you to fuse the LoRA weights directly with the original weights of the underlying model. This way, you're only loading the model once which can increase inference and lower memory-usage.

You can use PEFT to easily fuse/unfuse multiple adapters directly into the model weights (both UNet and text encoder) using the [`~loaders.LoraLoaderMixin.fuse_lora`] method, which can lead to a speed-up in inference and lower VRAM usage.

For example, if you have a base model and adapters loaded and set as active with the following adapter weights:

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")

pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
```

Fuse these LoRAs into the UNet with the [`~loaders.LoraLoaderMixin.fuse_lora`] method. The `lora_scale` parameter controls how much to scale the output by with the LoRA weights. It is important to make the `lora_scale` adjustments in the [`~loaders.LoraLoaderMixin.fuse_lora`] method because it won’t work if you try to pass `scale` to the `cross_attention_kwargs` in the pipeline.

```py
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
```

Then you should use [`~loaders.LoraLoaderMixin.unload_lora_weights`] to unload the LoRA weights since they've already been fused with the underlying base model. Finally, call [`~DiffusionPipeline.save_pretrained`] to save the fused pipeline locally or you could call [`~DiffusionPipeline.push_to_hub`] to push the fused pipeline to the Hub.

```py
pipeline.unload_lora_weights()
# save locally
pipeline.save_pretrained("path/to/fused-pipeline")
# save to the Hub
pipeline.push_to_hub("fused-ikea-feng")
```

Now you can quickly load the fused pipeline and use it for inference without needing to separately load the LoRA adapters.

```py
pipeline = DiffusionPipeline.from_pretrained(
    "username/fused-ikea-feng", torch_dtype=torch.float16,
).to("cuda")

image = pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", generator=torch.manual_seed(0)).images[0]
image
```

You can call [`~loaders.LoraLoaderMixin.unfuse_lora`] to restore the original model's weights (for example, if you want to use a different `lora_scale` value). However, this only works if you've only fused one LoRA adapter to the original model. If you've fused multiple LoRAs, you'll need to reload the model.

```py
pipeline.unfuse_lora()
```

### torch.compile

[torch.compile](../optimization/torch2.0#torchcompile) can speed up your pipeline even more, but the LoRA weights must be fused first and then unloaded. Typically, the UNet is compiled because it is such a computationally intensive component of the pipeline.

```py
from diffusers import DiffusionPipeline
import torch

# load base model and LoRAs
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")

# activate both LoRAs and set adapter weights
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])

# fuse LoRAs and unload weights
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
pipeline.unload_lora_weights()

# torch.compile
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

image = pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", generator=torch.manual_seed(0)).images[0]
```

Learn more about torch.compile in the [Accelerate inference of text-to-image diffusion models](../tutorials/fast_diffusion#torchcompile) guide.

## Next steps

For more conceptual details about how each merging method works, take a look at the [🤗 PEFT welcomes new merging methods](https://huggingface.co/blog/peft_merging#concatenation-cat) blog post!
