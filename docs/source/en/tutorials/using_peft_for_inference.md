<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Load LoRAs for inference

There are many adapter types (with [LoRAs](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) being the most popular) trained in different styles to achieve different effects. You can even combine multiple adapters to create new and unique images.

In this tutorial, you'll learn how to easily load and manage adapters for inference with the 🤗 [PEFT](https://huggingface.co/docs/peft/index) integration in 🤗 Diffusers. You'll use LoRA as the main adapter technique, so you'll see the terms LoRA and adapter used interchangeably.

Let's first install all the required libraries.

```bash
!pip install -q transformers accelerate peft diffusers
```

Now, load a pipeline with a [Stable Diffusion XL (SDXL)](../api/pipelines/stable_diffusion/stable_diffusion_xl) checkpoint:

```python
from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
```

Next, load a [CiroN2022/toy-face](https://huggingface.co/CiroN2022/toy-face) adapter with the [`~diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] method. With the 🤗 PEFT integration, you can assign a specific `adapter_name` to the checkpoint, which let's you easily switch between different LoRA checkpoints. Let's call this adapter `"toy"`.

```python
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
```

Make sure to include the token `toy_face` in the prompt and then you can perform inference:

```python
prompt = "toy_face of a hacker with a hoodie"

lora_scale = 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_8_1.png)

With the `adapter_name` parameter, it is really easy to use another adapter for inference! Load the [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl) adapter that has been fine-tuned to generate pixel art images and call it `"pixel"`.

The pipeline automatically sets the first loaded adapter (`"toy"`) as the active adapter, but you can activate the `"pixel"` adapter with the [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] method:

```python
pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters("pixel")
```

Make sure you include the token `pixel art` in your prompt to generate a pixel art image:

```python
prompt = "a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_12_1.png)

## Merge adapters

You can also merge different adapter checkpoints for inference to blend their styles together.

Once again, use the [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] method to activate the `pixel` and `toy` adapters and specify the weights for how they should be merged.

```python
pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
```

<Tip>

LoRA checkpoints in the diffusion community are almost always obtained with [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth). DreamBooth training often relies on "trigger" words in the input text prompts in order for the generation results to look as expected. When you combine multiple LoRA checkpoints, it's important to ensure the trigger words for the corresponding LoRA checkpoints are present in the input text prompts.

</Tip>

Remember to use the trigger words for [CiroN2022/toy-face](https://hf.co/CiroN2022/toy-face) and [nerijs/pixel-art-xl](https://hf.co/nerijs/pixel-art-xl) (these are found in their repositories) in the prompt to generate an image.

```python
prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face-pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_16_1.png)

Impressive! As you can see, the model generated an image that mixed the characteristics of both adapters.

> [!TIP]
> Through its PEFT integration, Diffusers also offers more efficient merging methods which you can learn about in the [Merge LoRAs](../using-diffusers/merge_loras) guide!

To return to only using one adapter, use the [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] method to activate the `"toy"` adapter:

```python
pipe.set_adapters("toy")

prompt = "toy_face of a hacker with a hoodie"
lora_scale = 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

Or to disable all adapters entirely, use the [`~diffusers.loaders.UNet2DConditionLoadersMixin.disable_lora`] method to return the base model.

```python
pipe.disable_lora()

prompt = "toy_face of a hacker with a hoodie"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![no-lora](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_20_1.png)

### Customize adapters strength
For even more customization, you can control how strongly the adapter affects each part of the pipeline. For this, pass a dictionary with the control strengths (called "scales") to [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`].

For example, here's how you can turn on the adapter for the `down` parts, but turn it off for the `mid` and `up` parts:
```python
pipe.enable_lora()  # enable lora again, after we disabled it above
prompt = "toy_face of a hacker with a hoodie, pixel art"
adapter_weight_scales = { "unet": { "down": 1, "mid": 0, "up": 0} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-text-and-down](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_down.png)

Let's see how turning off the `down` part and turning on the `mid` and `up` part respectively changes the image.
```python
adapter_weight_scales = { "unet": { "down": 0, "mid": 1, "up": 0} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-text-and-mid](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_mid.png)

```python
adapter_weight_scales = { "unet": { "down": 0, "mid": 0, "up": 1} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-text-and-up](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_up.png)

Looks cool!

This is a really powerful feature. You can use it to control the adapter strengths down to per-transformer level. And you can even use it for multiple adapters.
```python
adapter_weight_scales_toy = 0.5
adapter_weight_scales_pixel = {
    "unet": {
        "down": 0.9,  # all transformers in the down-part will use scale 0.9
        # "mid"  # because, in this example, "mid" is not given, all transformers in the mid part will use the default scale 1.0
        "up": {
            "block_0": 0.6,  # all 3 transformers in the 0th block in the up-part will use scale 0.6
            "block_1": [0.4, 0.8, 1.0],  # the 3 transformers in the 1st block in the up-part will use scales 0.4, 0.8 and 1.0 respectively
        }
    }
}
pipe.set_adapters(["toy", "pixel"], [adapter_weight_scales_toy, adapter_weight_scales_pixel])
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-mixed](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_mixed.png)

## Manage active adapters

You have attached multiple adapters in this tutorial, and if you're feeling a bit lost on what adapters have been attached to the pipeline's components, use the [`~diffusers.loaders.LoraLoaderMixin.get_active_adapters`] method to check the list of active adapters:

```py
active_adapters = pipe.get_active_adapters()
active_adapters
["toy", "pixel"]
```

You can also get the active adapters of each pipeline component with [`~diffusers.loaders.LoraLoaderMixin.get_list_adapters`]:

```py
list_adapters_component_wise = pipe.get_list_adapters()
list_adapters_component_wise
{"text_encoder": ["toy", "pixel"], "unet": ["toy", "pixel"], "text_encoder_2": ["toy", "pixel"]}
```
