<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]] 

# Using PEFT for LoRA inference in Diffusers

🤗 `peft` is an open-source library primarily maintained at Hugging Face. From the [official documentation](https://huggingface.co/docs/peft/index) of `peft`:

> 🤗 PEFT, or Parameter-Efficient Fine-Tuning (PEFT), is a library for efficiently adapting pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. PEFT methods only fine-tune a small number of (extra) model parameters, significantly decreasing computational and storage costs because fine-tuning large-scale PLMs is prohibitively costly. Recent state-of-the-art PEFT techniques achieve performance comparable to that of full fine-tuning.

But guess what? PEFT is not limited to just language models. It is modality-agnostic. This means it can be applied to pure vision models, vision-language models, speech models, and so on.

PEFT is natively integrated into Diffusers allowing users to take advantage of its support for doing efficient multi-adapter inference, swapping in and swapping out adapters, etc. In this guide, we walk through such use cases with [Stable Diffusion XL](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl), making you fully equipped with how you can take advantage of PEFT for adapter inference when using Diffusers.

Throughout this guide, we will use LoRA as the main adapter technique. As such, we will sometimes refer to LoRA as adapter and vice-versa. This guide assumes that you're familiar with LoRA. If you're not, we welcome you to check out [this guide](https://huggingface.co/docs/diffusers/main/en/training/lora).

Let's first install all the required libraries.

```bash
!pip install -q transformers accelerate
!pip install -q git+https://github.com/huggingface/peft.git
# To remove the branch once it's merged.
!pip install -q git+https://github.com/younesbelkada/diffusers.git@peft-part-2
```

## Load the SDXL pipeline

```python
from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
```

## Loading a LoRA checkpoint

We'll start by loading a LoRA checkpoint using the `load_lora_weights()` method. If you have used `diffusers` before for LoRA inference, you're probably already familiar with the method.

But with our new integration with PEFT, you can do much more with `load_lora_weights()` as you will notice in a moment.

Note that we assign a specific `adapter_name` to the checkpoint, so that we can easily switch between different LoRA checkpoints. Let's call it `"toy"`.

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
    
## Loading another adapter

Next, we load the adapter from `nerijs/pixel-art-xl` that has been fine-tuned to generate pixel art images. Let's call this one `"pixel"`!

The pipeline will automatically set the first loaded adapter as the active adapter. But you can activate the adapter with which you want to run inference with by using the `set_adapters()` method as shown below:

```python
pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters("pixel")
```

Let's now generate the image with the second adapter and check the result:

```python
prompt = "a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_12_1.png)
    
## Combining multiple adapters!

With the PEFT integration, it's easy to perform multi-adapter inference wherein you can combine different adapter checkpoints and perform inference. We discuss that in this section.

We use the `set_adapters()` method to activate two LoRA checkpoints specifying weight coefficients with which the checkpoints should be combined.

```python
# Change the argument name to `adapter_weights`.
pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
```

Now that we have set these two adapters, let's generate an image by combining the adapters!

<Tip>

LoRA checkpoints in the diffusion community are almost always obtained with [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth). DreamBooth training often relies on "trigger" words to be present in the input text prompts in order for the generation results to look as expected. So, when combining multiple LoRA checkpoints, it's important to keep this mind and ensure that the trigger words for the corresponding LoRA checkpoints are present in the input text prompts.

</Tip>

We can know about trigger words of the LoRA checkpoints being used from their repositories:

* [CiroN2022/toy-face](https://hf.co/CiroN2022/toy-face)
* [nerijs/pixel-art-xl](https://hf.co/nerijs/pixel-art-xl)


```python
# Notice how the promopt is constructed.
prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face-pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_16_1.png)
    
Impressive! As you can see, the model was able to generate an image that mixes the characteritics of both adapters.

After performing multi-adapter inference, it's possible to again go back to single-adapter inference. With the `set_adapters()` method, it's easy:

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

## Disabling all adapters

If you want to switch to the base model, you can disable all LoRAs with the `disable_lora()` method.


```python
pipe.disable_lora()

prompt = "toy_face of a hacker with a hoodie"
lora_scale= 0.9
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![no-lora](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_20_1.png)