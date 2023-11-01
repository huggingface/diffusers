<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Inference with PEFT

There are many adapters trained in different styles to achieve different effects. You can even combine multiple adapters to create new and unique images. With the 🤗 [PEFT](https://huggingface.co/docs/peft/index) integration in 🤗 Diffusers, it is really easy to load and manage adapters for inference. In this guide, you'll learn how to use different adapters with [Stable Diffusion XL (SDXL)](../api/pipelines/stable_diffusion/stable_diffusion_xl) for inference.

Throughout this guide, you'll use LoRA as the main adapter technique, so we'll use the terms LoRA and adapter interchangeably. You should have some familiarity with LoRA, and if you don't, we welcome you to check out the [LoRA guide](https://huggingface.co/docs/peft/conceptual_guides/lora).

Let's first install all the required libraries.

```bash
!pip install -q transformers accelerate
# Will be updated once the stable releases are done.
!pip install -q git+https://github.com/huggingface/peft.git
!pip install -q git+https://github.com/huggingface/diffusers.git
```

Now, let's load a pipeline with a SDXL checkpoint:

```python
from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
```


Next, load a LoRA checkpoint with the [`~diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] method.

With the 🤗 PEFT integration, you can assign a specific `adapter_name` to the checkpoint, which let's you easily switch between different LoRA checkpoints. Let's call this adapter `"toy"`.

```python
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
```

And then perform inference:

```python
prompt = "toy_face of a hacker with a hoodie"

lora_scale= 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_8_1.png)
    

With the `adapter_name` parameter, it is really easy to use another adapter for inference! Load the [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl) adapter that has been fine-tuned to generate pixel art images, and let's call it `"pixel"`.

The pipeline automatically sets the first loaded adapter (`"toy"`) as the active adapter. But you can activate the `"pixel"` adapter with the [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] method as shown below:

```python
pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters("pixel")
```

Let's now generate an image with the second adapter and check the result:

```python
prompt = "a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_12_1.png)
    
## Combine multiple adapters

You can also perform multi-adapter inference where you combine different adapter checkpoints for inference.

Once again, use the [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] method to activate two LoRA checkpoints and specify the weight for how the checkpoints should be combined.

```python
pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
```

Now that we have set these two adapters, let's generate an image from the combined adapters!

<Tip>

LoRA checkpoints in the diffusion community are almost always obtained with [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth). DreamBooth training often relies on "trigger" words in the input text prompts in order for the generation results to look as expected. When you combine multiple LoRA checkpoints, it's important to ensure the trigger words for the corresponding LoRA checkpoints are present in the input text prompts.

</Tip>

The trigger words for [CiroN2022/toy-face](https://hf.co/CiroN2022/toy-face) and [nerijs/pixel-art-xl](https://hf.co/nerijs/pixel-art-xl) are found in their repositories.


```python
# Notice how the prompt is constructed.
prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face-pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_16_1.png)
    
Impressive! As you can see, the model was able to generate an image that mixes the characteristics of both adapters.

If you want to go back to using only one adapter, use the [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] method to activate the `"toy"` adapter:

```python
# First, set the adapter.
pipe.set_adapters("toy")

# Then, run inference.
prompt = "toy_face of a hacker with a hoodie"
lora_scale= 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face-again](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_18_1.png)


If you want to switch to only the base model, disable all LoRAs with the [`~diffusers.loaders.UNet2DConditionLoadersMixin.disable_lora`] method.


```python
pipe.disable_lora()

prompt = "toy_face of a hacker with a hoodie"
lora_scale= 0.9
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![no-lora](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_20_1.png)

## Monitoring active adapters

You have attached multiple adapters in this tutorial, and if you're feeling a bit lost on what adapters have been attached to the pipeline's components, you can easily check the list of active adapters using the [`~diffusers.loaders.LoraLoaderMixin.get_active_adapters`] method:

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
