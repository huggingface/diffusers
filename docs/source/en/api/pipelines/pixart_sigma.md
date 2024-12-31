<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PixArt-Σ

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pixart/header_collage_sigma.jpg)

[PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://huggingface.co/papers/2403.04692) is Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li.

The abstract from the paper is:

*In this paper, we introduce PixArt-Σ, a Diffusion Transformer model (DiT) capable of directly generating images at 4K resolution. PixArt-Σ represents a significant advancement over its predecessor, PixArt-α, offering images of markedly higher fidelity and improved alignment with text prompts. A key feature of PixArt-Σ is its training efficiency. Leveraging the foundational pre-training of PixArt-α, it evolves from the ‘weaker’ baseline to a ‘stronger’ model via incorporating higher quality data, a process we term “weak-to-strong training”. The advancements in PixArt-Σ are twofold: (1) High-Quality Training Data: PixArt-Σ incorporates superior-quality image data, paired with more precise and detailed image captions. (2) Efficient Token Compression: we propose a novel attention module within the DiT framework that compresses both keys and values, significantly improving efficiency and facilitating ultra-high-resolution image generation. Thanks to these improvements, PixArt-Σ achieves superior image quality and user prompt adherence capabilities with significantly smaller model size (0.6B parameters) than existing text-to-image diffusion models, such as SDXL (2.6B parameters) and SD Cascade (5.1B parameters). Moreover, PixArt-Σ’s capability to generate 4K images supports the creation of high-resolution posters and wallpapers, efficiently bolstering the production of highquality visual content in industries such as film and gaming.*

You can find the original codebase at [PixArt-alpha/PixArt-sigma](https://github.com/PixArt-alpha/PixArt-sigma) and all the available checkpoints at [PixArt-alpha](https://huggingface.co/PixArt-alpha).

Some notes about this pipeline:

* It uses a Transformer backbone (instead of a UNet) for denoising. As such it has a similar architecture as [DiT](https://hf.co/docs/transformers/model_doc/dit).
* It was trained using text conditions computed from T5. This aspect makes the pipeline better at following complex text prompts with intricate details.
* It is good at producing high-resolution images at different aspect ratios. To get the best results, the authors recommend some size brackets which can be found [here](https://github.com/PixArt-alpha/PixArt-sigma/blob/master/diffusion/data/datasets/utils.py).
* It rivals the quality of state-of-the-art text-to-image generation systems (as of this writing) such as PixArt-α, Stable Diffusion XL, Playground V2.0 and DALL-E 3, while being more efficient than them.
* It shows the ability of generating super high resolution images, such as 2048px or even 4K.
* It shows that text-to-image models can grow from a weak model to a stronger one through several improvements (VAEs, datasets, and so on.)

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

<Tip>

You can further improve generation quality by passing the generated image from [`PixArtSigmaPipeline`] to the [SDXL refiner](../../using-diffusers/sdxl#base-to-refiner-model) model.

</Tip>

## Inference with under 8GB GPU VRAM

Run the [`PixArtSigmaPipeline`] with under 8GB GPU VRAM by loading the text encoder in 8-bit precision. Let's walk through a full-fledged example.

First, install the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library:

```bash
pip install -U bitsandbytes
```

Then load the text encoder in 8-bit:

```python
from transformers import T5EncoderModel
from diffusers import PixArtSigmaPipeline
import torch

text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    subfolder="text_encoder",
    load_in_8bit=True,
    device_map="auto",
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    text_encoder=text_encoder,
    transformer=None,
    device_map="balanced"
)
```

Now, use the `pipe` to encode a prompt:

```python
with torch.no_grad():
    prompt = "cute cat"
    prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)
```

Since text embeddings have been computed, remove the `text_encoder` and `pipe` from the memory, and free up some GPU VRAM:

```python
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()

del text_encoder
del pipe
flush()
```

Then compute the latents with the prompt embeddings as inputs:

```python
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    text_encoder=None,
    torch_dtype=torch.float16,
).to("cuda")

latents = pipe(
    negative_prompt=None,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    num_images_per_prompt=1,
    output_type="latent",
).images

del pipe.transformer
flush()
```

<Tip>

Notice that while initializing `pipe`, you're setting `text_encoder` to `None` so that it's not loaded.

</Tip>

Once the latents are computed, pass it off to the VAE to decode into a real image:

```python
with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="pil")[0]
image.save("cat.png")
```

By deleting components you aren't using and flushing the GPU VRAM, you should be able to run [`PixArtSigmaPipeline`] with under 8GB GPU VRAM.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pixart/8bits_cat.png)

If you want a report of your memory-usage, run this [script](https://gist.github.com/sayakpaul/3ae0f847001d342af27018a96f467e4e).

<Tip warning={true}>

Text embeddings computed in 8-bit can impact the quality of the generated images because of the information loss in the representation space caused by the reduced precision. It's recommended to compare the outputs with and without 8-bit.

</Tip>

While loading the `text_encoder`, you set `load_in_8bit` to `True`. You could also specify `load_in_4bit` to bring your memory requirements down even further to under 7GB.

## PixArtSigmaPipeline

[[autodoc]] PixArtSigmaPipeline
	- all
	- __call__
