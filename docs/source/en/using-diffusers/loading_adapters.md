<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load adapters

[[open-in-colab]]

There are several [training](../training/overview) techniques for personalizing diffusion models to generate images of a specific subject or images in certain styles. Each of these training methods produces a different type of adapter. Some of the adapters generate an entirely new model, while other adapters only modify a smaller set of embeddings or weights. This means the loading process for each adapter is also different.

This guide will show you how to load DreamBooth, textual inversion, and LoRA weights.

<Tip>

Feel free to browse the [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer), [LoRA the Explorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer), and the [Diffusers Models Gallery](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery) for checkpoints and embeddings to use.

</Tip>

## DreamBooth

[DreamBooth](https://dreambooth.github.io/) finetunes an *entire diffusion model* on just several images of a subject to generate images of that subject in new styles and settings. This method works by using a special word in the prompt that the model learns to associate with the subject image. Of all the training methods, DreamBooth produces the largest file size (usually a few GBs) because it is a full checkpoint model.

Let's load the [herge_style](https://huggingface.co/sd-dreambooth-library/herge-style) checkpoint, which is trained on just 10 images drawn by Hergé, to generate images in that style. For it to work, you need to include the special word `herge_style` in your prompt to trigger the checkpoint:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("sd-dreambooth-library/herge-style", torch_dtype=torch.float16).to("cuda")
prompt = "A cute herge_style brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_dreambooth.png" />
</div>

## Textual inversion

[Textual inversion](https://textual-inversion.github.io/) is very similar to DreamBooth and it can also personalize a diffusion model to generate certain concepts (styles, objects) from just a few images. This method works by training and finding new embeddings that represent the images you provide with a special word in the prompt. As a result, the diffusion model weights stay the same and the training process produces a relatively tiny (a few KBs) file.

Because textual inversion creates embeddings, it cannot be used on its own like DreamBooth and requires another model.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
```

Now you can load the textual inversion embeddings with the [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] method and generate some images. Let's load the [sd-concepts-library/gta5-artwork](https://huggingface.co/sd-concepts-library/gta5-artwork) embeddings and you'll need to include the special word `<gta5-artwork>` in your prompt to trigger it:

```py
pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, <gta5-artwork> style"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_txt_embed.png" />
</div>

Textual inversion can also be trained on undesirable things to create *negative embeddings* to discourage a model from generating images with those undesirable things like blurry images or extra fingers on a hand. This can be an easy way to quickly improve your prompt. You'll also load the embeddings with [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`], but this time, you'll need two more parameters:

- `weight_name`: specifies the weight file to load if the file was saved in the 🤗 Diffusers format with a specific name or if the file is stored in the A1111 format
- `token`: specifies the special word to use in the prompt to trigger the embeddings

Let's load the [sayakpaul/EasyNegative-test](https://huggingface.co/sayakpaul/EasyNegative-test) embeddings:

```py
pipeline.load_textual_inversion(
    "sayakpaul/EasyNegative-test", weight_name="EasyNegative.safetensors", token="EasyNegative"
)
```

Now you can use the `token` to generate an image with the negative embeddings:

```py
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, EasyNegative"
negative_prompt = "EasyNegative"

image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png" />
</div>

## LoRA

[Low-Rank Adaptation (LoRA)](https://huggingface.co/papers/2106.09685) is a popular training technique because it is fast and generates smaller file sizes (a couple hundred MBs). Like the other methods in this guide, LoRA can train a model to learn new styles from just a few images. It works by inserting new weights into the diffusion model and then only the new weights are trained instead of the entire model. This makes LoRAs faster to train and easier to store.

<Tip>

LoRA is a very general training technique that can be used with other training methods. For example, it is common to train a model with DreamBooth and LoRA.

</Tip>

LoRAs also need to be used with another model:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
```

Then use the [`~loaders.LoraLoaderMixin.load_lora_weights`] method to load the [ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora) weights and specify the weights filename from the repository:

```py
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors")
prompt = "bears, pizza bites"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_lora.png" />
</div>

The [`~loaders.LoraLoaderMixin.load_lora_weights`] method loads LoRA weights into both the UNet and text encoder. It is the preferred way for loading LoRAs because it can handle cases where:

- the LoRA weights don't have separate identifiers for the UNet and text encoder
- the LoRA weights have separate identifiers for the UNet and text encoder

But if you only need to load LoRA weights into the UNet, then you can use the [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method. Let's load the [jbilcke-hf/sdxl-cinematic-1](https://huggingface.co/jbilcke-hf/sdxl-cinematic-1) LoRA:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors")

# use cnmt in the prompt to trigger the LoRA
prompt = "A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_attn_proc.png" />
</div>

<Tip>

For both [`~loaders.LoraLoaderMixin.load_lora_weights`] and [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`], you can pass the `cross_attention_kwargs={"scale": 0.5}` parameter to adjust how much of the LoRA weights to use. A value of `0` is the same as only using the base model weights, and a value of `1` is equivalent to using the fully finetuned LoRA.

</Tip>

To unload the LoRA weights, use the [`~loaders.LoraLoaderMixin.unload_lora_weights`] method to discard the LoRA weights and restore the model to its original weights:

```py
pipeline.unload_lora_weights()
```

### Load multiple LoRAs

It can be fun to use multiple LoRAs together to create something entirely new and unique. The [`~loaders.LoraLoaderMixin.fuse_lora`] method allows you to fuse the LoRA weights with the original weights of the underlying model.

<Tip>

Fusing the weights can lead to a speedup in inference latency because you don't need to separately load the base model and LoRA! You can save your fused pipeline with [`~DiffusionPipeline.save_pretrained`] to avoid loading and fusing the weights every time you want to use the model.

</Tip>

Load an initial model:

```py
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")
```

Next, load the LoRA checkpoint and fuse it with the original weights. The `lora_scale` parameter controls how much to scale the output by with the LoRA weights. It is important to make the `lora_scale` adjustments in the [`~loaders.LoraLoaderMixin.fuse_lora`] method because it won't work if you try to pass `scale` to the `cross_attention_kwargs` in the pipeline.

If you need to reset the original model weights for any reason (use a different `lora_scale`), you should use the [`~loaders.LoraLoaderMixin.unfuse_lora`] method.

```py
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl")
pipeline.fuse_lora(lora_scale=0.7)

# to unfuse the LoRA weights
pipeline.unfuse_lora()
```

Then fuse this pipeline with the next set of LoRA weights:

```py
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora")
pipeline.fuse_lora(lora_scale=0.7)
```

<Tip warning={true}>

You can't unfuse multiple LoRA checkpoints, so if you need to reset the model to its original weights, you'll need to reload it.

</Tip>

Now you can generate an image that uses the weights from both LoRAs:

```py
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

### 🤗 PEFT

<Tip>

Read the [Inference with 🤗 PEFT](../tutorials/using_peft_for_inference) tutorial to learn more about its integration with 🤗 Diffusers and how you can easily work with and juggle multiple adapters. You'll need to install 🤗 Diffusers and PEFT from source to run the example in this section.

</Tip>

Another way you can load and use multiple LoRAs is to specify the `adapter_name` parameter in [`~loaders.LoraLoaderMixin.load_lora_weights`]. This method takes advantage of the 🤗 PEFT integration. For example, load and name both LoRA weights:

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors", adapter_name="cereal")
```

Now use the [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] to activate both LoRAs, and you can configure how much weight each LoRA should have on the output:

```py
pipeline.set_adapters(["ikea", "cereal"], adapter_weights=[0.7, 0.5])
```

Then, generate an image:

```py
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}).images[0]
image
```

### Kohya and TheLastBen

Other popular LoRA trainers from the community include those by [Kohya](https://github.com/kohya-ss/sd-scripts/) and [TheLastBen](https://github.com/TheLastBen/fast-stable-diffusion). These trainers create different LoRA checkpoints than those trained by 🤗 Diffusers, but they can still be loaded in the same way.

Let's download the [Blueprintify SD XL 1.0](https://civitai.com/models/150986/blueprintify-sd-xl-10) checkpoint from [Civitai](https://civitai.com/):

```sh
!wget https://civitai.com/api/download/models/168776 -O blueprintify-sd-xl-10.safetensors
```

Load the LoRA checkpoint with the [`~loaders.LoraLoaderMixin.load_lora_weights`] method, and specify the filename in the `weight_name` parameter:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("path/to/weights", weight_name="blueprintify-sd-xl-10.safetensors")
```

Generate an image:

```py
# use bl3uprint in the prompt to trigger the LoRA
prompt = "bl3uprint, a highly detailed blueprint of the eiffel tower, explaining how to build all parts, many txt, blueprint grid backdrop"
image = pipeline(prompt).images[0]
image
```

<Tip warning={true}>

Some limitations of using Kohya LoRAs with 🤗 Diffusers include:

- Images may not look like those generated by UIs - like ComfyUI - for multiple reasons, which are explained [here](https://github.com/huggingface/diffusers/pull/4287/#issuecomment-1655110736).
- [LyCORIS checkpoints](https://github.com/KohakuBlueleaf/LyCORIS) aren't fully supported. The [`~loaders.LoraLoaderMixin.load_lora_weights`] method loads LyCORIS checkpoints with LoRA and LoCon modules, but Hada and LoKR are not supported.

</Tip>

Loading a checkpoint from TheLastBen is very similar. For example, to load the [TheLastBen/William_Eggleston_Style_SDXL](https://huggingface.co/TheLastBen/William_Eggleston_Style_SDXL) checkpoint:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("TheLastBen/William_Eggleston_Style_SDXL", weight_name="wegg.safetensors")

# use by william eggleston in the prompt to trigger the LoRA
prompt = "a house by william eggleston, sunrays, beautiful, sunlight, sunrays, beautiful"
image = pipeline(prompt=prompt).images[0]
image
```

## IP-Adapter 

[IP-Adapter](https://ip-adapter.github.io/) is an effective and lightweight adapter that adds image prompting capabilities to a diffusion model. This adapter works by decoupling the cross-attention layers of the image and text features. All the other model components are frozen and only the embedded image features in the UNet are trained. As a result, IP-Adapter files are typically only ~100MBs.

IP-Adapter works with most of our pipelines, including Stable Diffusion, Stable Diffusion XL (SDXL), ControlNet, T2I-Adapter, AnimateDiff.  And you can use any custom models finetuned from the same base models. It also works with LCM-Lora out of box.


<Tip>

You can find official IP-Adapter checkpoints in [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter).

IP-Adapter was contributed by [okotaku](https://github.com/okotaku).

</Tip>

Let's first create a Stable Diffusion Pipeline.

```py
from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image


pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
```

Now load the [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) weights with the [`~loaders.IPAdapterMixin.load_ip_adapter`] method. 

```py
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
```

<Tip>
IP-Adapter relies on an image encoder to generate the image features, if your IP-Adapter weights folder contains a "image_encoder" subfolder, the image encoder will be automatically loaded and registered to the pipeline. Otherwise you can so load a [`~transformers.CLIPVisionModelWithProjection`] model and  pass it to a Stable Diffusion pipeline when you create it.

```py
from diffusers import AutoPipelineForText2Image, CLIPVisionModelWithProjection
import torch

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", 
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
).to("cuda")

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, torch_dtype=torch.float16).to("cuda")
```
</Tip>

IP-Adapter allows you to use both image and text to condition the image generation process. For example, let's use the bear image from the [Textual Inversion](#textual-inversion) section as the image prompt (`ip_adapter_image`) along with a text prompt to add "sunglasses". 😎

```py
pipeline.set_ip_adapter_scale(0.6)
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt='best quality, high quality, wearing sunglasses', 
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=50,
    generator=generator,
).images
images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip-bear.png" />
</div>

<Tip>

You can use the [`~loaders.IPAdapterMixin.set_ip_adapter_scale`] method to adjust the text prompt and image prompt condition ratio.  If you're only using the image prompt, you should set the scale to `1.0`. You can lower the scale to get more generation diversity, but it'll be less aligned with the prompt.
`scale=0.5` can achieve good results in most cases when you use both text and image prompts.
</Tip>

IP-Adapter also works great with Image-to-Image and Inpainting pipelines. See below examples of how you can use it with Image-to-Image and Inpaint.

<hfoptions id="tasks">
<hfoption id="image-to-image">

```py
from diffusers import AutoPipelineForImage2Image
import torch
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/vermeer.jpg")
ip_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/river.png")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt='best quality, high quality', 
    image = image,
    ip_adapter_image=ip_image,
    num_inference_steps=50,
    generator=generator,
    strength=0.6,
).images
images[0]
```

</hfoption>
<hfoption id="inpaint">

```py
from diffusers import AutoPipelineForInpaint
import torch
from diffusers.utils import load_image

pipeline = AutoPipelineForInpaint.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float).to("cuda")

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/inpaint_image.png")
mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/mask.png")
ip_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/girl.png")

image = image.resize((512, 768))
mask = mask.resize((512, 768))

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt='best quality, high quality', 
    image = image,
    mask_image = mask,
    ip_adapter_image=ip_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=50,
    generator=generator,
    strength=0.5,
).images
images[0]
```
</hfoption>
</hfoptions>


IP-Adapters can also be used with [SDXL](../api/pipelines/stable_diffusion/stable_diffusion_xl.md)

```python
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

image = load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/watercolor_painting.jpeg")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

generator = torch.Generator(device="cpu").manual_seed(33)
image = pipeline(
    prompt="best quality, high quality", 
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=25,
    generator=generator,
).images[0]
image.save("sdxl_t2i.png")
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/watercolor_painting.jpeg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">input image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/sdxl_t2i.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">adapted image</figcaption>
  </div>
</div>

You can use the IP-Adapter face model to apply specific faces to your images.  It is an effective way to maintain consistent characters in your image generations.
Weights are loaded with the same method used for the other IP-Adapters.  

```python
# Load ip-adapter-full-face_sd15.bin
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
```

<Tip>

It is recommended to use `DDIMScheduler` and `EulerDiscreteScheduler` for face model. 


</Tip>

```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1
)

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")

pipeline.set_ip_adapter_scale(0.7)

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ai_face2.png")

generator = torch.Generator(device="cpu").manual_seed(33)

image = pipeline(
    prompt="A photo of a girl wearing a black dress, holding red roses in hand, upper body, behind is the Eiffel Tower",
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=50, num_images_per_prompt=1, width=512, height=704,
    generator=generator,
).images[0]
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ai_face2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">input image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ipadapter_full_face_output.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">output image</figcaption>
  </div>
</div>

### LCM-Lora

You can use IP-Adapter with LCM-Lora to achieve "instant fine-tune" with custom images. Note that you need to load IP-Adapter weights before loading the LCM-Lora weights.

```py
from diffusers import DiffusionPipeline, LCMScheduler
import torch
from diffusers.utils import load_image

model_id =  "sd-dreambooth-library/herge-style"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "best quality, high quality"
image = load_image("https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
images = pipe(
    prompt=prompt,
    ip_adapter_image=image,
    num_inference_steps=4,
    guidance_scale=1,
).images[0]
```

### Other pipelines

IP-Adapter is compatible with any pipeline that (1) uses a text prompt and (2) uses Stable Diffusion or Stable Diffusion XL checkpoint. To use IP-Adapter with a different pipeline, all you need to do is to run `load_ip_adapter()` method after you create the pipeline, and then pass your image to the pipeline as `ip_adapter_image`

<Tip>

🤗 Diffusers currently only supports using IP-Adapter with some of the most popular pipelines, feel free to open a [feature request](https://github.com/huggingface/diffusers/issues/new/choose) if you have a cool use-case and require integrating IP-adapters with a pipeline that does not support it yet!

</Tip>

You can find below examples on how to use IP-Adapter with ControlNet and AnimateDiff. 

<hfoptions id="model">
<hfoption id="ControlNet">

```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers.utils import load_image

controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
pipeline.to("cuda")

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png")
depth_map = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/depth.png")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt='best quality, high quality', 
    image=depth_map,
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=50,
    generator=generator,
).images
images[0]
```
<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">input image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ipa-controlnet-out.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">adapted image</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="AnimateDiff">

```py
# animate diff + ip adapter
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif, load_image

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "Lykon/DreamShaper"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)

# scheduler
scheduler = DDIMScheduler(
    clip_sample=False,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="linear",
    timestep_spacing="trailing",
    steps_offset=1
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# load ip_adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

# load motion adapters
pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-up", adapter_name="tilt-up")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-pan-left", adapter_name="pan-left")

seed = 42
image = load_image("https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
images = [image] * 3
prompts = ["best quality, high quality"] * 3
negative_prompt = "bad quality, worst quality"
adapter_weights = [[0.75, 0.0, 0.0], [0.0, 0.0, 0.75], [0.0, 0.75, 0.75]]

# generate
output_frames = []
for prompt, image, adapter_weight in zip(prompts, images, adapter_weights):
    pipe.set_adapters(["zoom-out", "tilt-up", "pan-left"], adapter_weights=adapter_weight)
    output = pipe(
      prompt= prompt,
      num_frames=16,
      guidance_scale=7.5,
      num_inference_steps=30,
      ip_adapter_image = image,
      generator=torch.Generator("cpu").manual_seed(seed),
    )
    frames = output.frames[0]
    output_frames.extend(frames)

export_to_gif(output_frames, "test_out_animation.gif") 
```

</hfoption>
</hfoptions>

