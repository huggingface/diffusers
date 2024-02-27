<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Merging LoRA adapters 

> [!WARNING]
> This is experimental and you should proceed with caution. An extended discussion on this topic can be found [here](https://github.com/huggingface/diffusers/issues/6892). 

[LoRA](../training/lora.md) allows one to fine-tune a model to adapt to a particular visual style in a cost-effective way. Merging different LoRA checkpoints can help combine different styles in a coherent way, preserving the overall aesthetics of the generated content. It should also remain faithful to the properties that the individual LoRA checkpoints bring (specific textures, for example).

This guide shows how you can merge different LoRAs using the 🤗 [PEFT](https://huggingface.co/docs/peft/index) library.

## Setup

Make sure you've installed PEFT from the source and you have the latest stable version of Diffusers installed: 

```bash
pip install git+https://github.com/huggingface/peft.git
pip install -U diffusers
```

## The general workflow

As seen in [this guide](./using_peft_for_inference.md), Diffusers [already relies on PEFT](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference) for all things LoRA, including training and inference. However, currently, it’s not possible to benefit from the new merging methods when calling [`set_adapters()`](https://huggingface.co/docs/diffusers/main/en/api/loaders/unet#diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters) on a Diffusers pipeline. 

But thanks to PEFT, there’s a way to circumvent around this. You will use the [`add_weighted_adapter()`](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter) functionality for this. Precisely, these are the steps that you will take to combine the [“toy-face” LoRA](https://huggingface.co/CiroN2022/toy-face) and the [“Pixel-Art” loRA](https://huggingface.co/nerijs/pixel-art-xl), and experiment with different merging techniques:

- Obtain `PeftModel`s from these LoRA checkpoints.
- Merge the `PeftModel`s using the `add_weighted_adapter()` method with a merging method of our choice.
- Assign the merged model to the respective component of the underlying `DiffusionPipeline`.

## Merge LoRAs

Since both the LoRA checkpoints use [SDXL](../using-diffusers/sdxl.md) UNet as the their base model, you will first load the UNet:

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

Load the actual SDXL pipeline and the LoRA checkpoints. Start with the “CiroN2022/toy-face” LoRA:

```python
from diffusers import DiffusionPipeline
import copy

sdxl_unet = copy.deepcopy(unet)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    unet=unet
).to("cuda")
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
```

Now, obtain the PeftModel from the loaded LoRA checkpoint:

```python
from peft import get_peft_model, LoraConfig

toy_peft_model = get_peft_model(
    sdxl_unet,
    pipe.unet.peft_config["toy"],
    adapter_name="toy"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}
toy_peft_model.load_state_dict(original_state_dict, strict=True)
```

> [!TIP]
> You can optionally push the toy_peft_model to the Hub using: `toy_peft_model.push_to_hub("toy_peft_model", token=TOKEN)`.

Repeat the same for the “nerijs/pixel-art-xl” LoRA:

```python
pipe.delete_adapters("toy")
sdxl_unet.delete_adapters("toy")

pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters(adapter_names="pixel")

pixel_peft_model = get_peft_model(
    sdxl_unet,
    pipe.unet.peft_config["pixel"],
    adapter_name="pixel"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}
pixel_peft_model.load_state_dict(original_state_dict, strict=True)
```

Load the adapters into the UNet:

```python
from peft import PeftModel
from diffusers import UNet2DConditionModel, DiffusionPipeline
import torch

base_unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")

toy_id = "sayakpaul/toy_peft_model"
model = PeftModel.from_pretrained(base_unet, toy_id, use_safetensors=True, subfolder="toy", adapter_name="toy")
model.load_adapter("sayakpaul/pixel_peft_model", use_safetensors=True, subfolder="pixel", adapter_name="pixel")
```

Finally, merge the adapters:

```python
model.add_weighted_adapter(
    adapters=["toy", "pixel"],
    weights=[0.7, 0.3],
    combination_type="linear",
    adapter_name="toy-pixel"
)
model.set_adapters("toy-pixel")
```

Refer to [this post](https://huggingface.co/blog/peft_merging) to know more about the different merging methods available for LoRA adapters.

## Run inference

```python
model = model.to(dtype=torch.float16, device="cuda")

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=model, variant="fp16", torch_dtype=torch.float16,
).to("cuda")

prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![toy_face_hacker](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft_merging/toy_face_hacker.png)

With a different merging method, "DARE Linear", you can get more aesthetic results:

```python
model.delete_adapter("toy-pixel")
model.add_weighted_adapter(
    adapters=["toy", "pixel"],
    weights=[1.0, 1.0],
    combination_type="dare_linear",
    adapter_name="merge",
    density=0.7,
    adapter_name="toy-pixel-dare-linear"
)
model.set_adapters("toy-pixel-dare-linear")
model = model.to(dtype=torch.float16, device="cuda")

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=model, variant="fp16", torch_dtype=torch.float16,
).to("cuda")

prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![toy_face_pixel_art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft_merging/toy_face_pixel_art.png)

There are more examples of different merging methods available in [this post](https://huggingface.co/blog/peft_merging). We encourage you to give them a try 🤗

## AnimateDiff

TODO
