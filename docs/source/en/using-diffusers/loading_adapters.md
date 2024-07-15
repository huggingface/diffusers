<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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

LoRA is a very general training technique that can be used with other training methods. For example, it is common to train a model with DreamBooth and LoRA. It is also increasingly common to load and merge multiple LoRAs to create new and unique images. You can learn more about it in the in-depth [Merge LoRAs](merge_loras) guide since merging is outside the scope of this loading guide.

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

To unload the LoRA weights, use the [`~loaders.LoraLoaderMixin.unload_lora_weights`] method to discard the LoRA weights and restore the model to its original weights:

```py
pipeline.unload_lora_weights()
```

### Adjust LoRA weight scale

For both [`~loaders.LoraLoaderMixin.load_lora_weights`] and [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`], you can pass the `cross_attention_kwargs={"scale": 0.5}` parameter to adjust how much of the LoRA weights to use. A value of `0` is the same as only using the base model weights, and a value of `1` is equivalent to using the fully finetuned LoRA.

For more granular control on the amount of LoRA weights used per layer, you can use [`~loaders.LoraLoaderMixin.set_adapters`] and pass a dictionary specifying by how much to scale the weights in each layer by.
```python
pipe = ... # create pipeline
pipe.load_lora_weights(..., adapter_name="my_adapter")
scales = {
    "text_encoder": 0.5,
    "text_encoder_2": 0.5,  # only usable if pipe has a 2nd text encoder
    "unet": {
        "down": 0.9,  # all transformers in the down-part will use scale 0.9
        # "mid"  # in this example "mid" is not given, therefore all transformers in the mid part will use the default scale 1.0
        "up": {
            "block_0": 0.6,  # all 3 transformers in the 0th block in the up-part will use scale 0.6
            "block_1": [0.4, 0.8, 1.0],  # the 3 transformers in the 1st block in the up-part will use scales 0.4, 0.8 and 1.0 respectively
        }
    }
}
pipe.set_adapters("my_adapter", scales)
```

This also works with multiple adapters - see [this guide](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference#customize-adapters-strength) for how to do it.

<Tip warning={true}>

Currently, [`~loaders.LoraLoaderMixin.set_adapters`] only supports scaling attention weights. If a LoRA has other parts (e.g., resnets or down-/upsamplers), they will keep a scale of 1.0.

</Tip>

### Kohya and TheLastBen

Other popular LoRA trainers from the community include those by [Kohya](https://github.com/kohya-ss/sd-scripts/) and [TheLastBen](https://github.com/TheLastBen/fast-stable-diffusion). These trainers create different LoRA checkpoints than those trained by 🤗 Diffusers, but they can still be loaded in the same way.

<hfoptions id="other-trainers">
<hfoption id="Kohya">

To load a Kohya LoRA, let's download the [Blueprintify SD XL 1.0](https://civitai.com/models/150986/blueprintify-sd-xl-10) checkpoint from [Civitai](https://civitai.com/) as an example:

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

</hfoption>
<hfoption id="TheLastBen">

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

</hfoption>
</hfoptions>

## IP-Adapter

[IP-Adapter](https://ip-adapter.github.io/) is a lightweight adapter that enables image prompting for any diffusion model. This adapter works by decoupling the cross-attention layers of the image and text features. All the other model components are frozen and only the embedded image features in the UNet are trained. As a result, IP-Adapter files are typically only ~100MBs.

You can learn more about how to use IP-Adapter for different tasks and specific use cases in the [IP-Adapter](../using-diffusers/ip_adapter) guide.

> [!TIP]
> Diffusers currently only supports IP-Adapter for some of the most popular pipelines. Feel free to open a feature request if you have a cool use case and want to integrate IP-Adapter with an unsupported pipeline!
> Official IP-Adapter checkpoints are available from [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter).

To start, load a Stable Diffusion checkpoint.

```py
from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
```

Then load the IP-Adapter weights and add it to the pipeline with the [`~loaders.IPAdapterMixin.load_ip_adapter`] method.

```py
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
```

Once loaded, you can use the pipeline with an image and text prompt to guide the image generation process.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt='best quality, high quality, wearing sunglasses',
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=50,
    generator=generator,
).images[0]
images
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip-bear.png" />
</div>

### IP-Adapter Plus

IP-Adapter relies on an image encoder to generate image features. If the IP-Adapter repository contains an `image_encoder` subfolder, the image encoder is automatically loaded and registered to the pipeline. Otherwise, you'll need to explicitly load the image encoder with a [`~transformers.CLIPVisionModelWithProjection`] model and pass it to the pipeline.

This is the case for *IP-Adapter Plus* checkpoints which use the ViT-H image encoder.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
```

### IP-Adapter Face ID models

The IP-Adapter FaceID models are experimental IP Adapters that use image embeddings generated by `insightface` instead of CLIP image embeddings. Some of these models also use LoRA to improve ID consistency.
You need to install `insightface` and all its requirements to use these models.

<Tip warning={true}>
As InsightFace pretrained models are available for non-commercial research purposes, IP-Adapter-FaceID models are released exclusively for research purposes and are not intended for commercial use.
</Tip>

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None)
```

If you want to use one of the two IP-Adapter FaceID Plus models, you must also load the CLIP image encoder, as this models use both `insightface` and CLIP image embeddings to achieve better photorealism.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    image_encoder=image_encoder,
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plus_sd15.bin")
```
