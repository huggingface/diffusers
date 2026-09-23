<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Prompting

Prompts describe what a model should generate. Good prompts are detailed, specific, structured, and they generate better images and videos.

This guide shows you how to write effective prompts and introduces techniques that make them stronger.

## Writing good prompts

Every effective prompt needs three core elements.

1. Subject - what you want to generate. Start your prompt here.
2. Style - the medium or aesthetic. How should it look?
3. Context - details about actions, setting, and mood.

Use these elements as a structured narrative, not a keyword list. Modern models understand language better than keyword matching. Start simple, then add details.

Context is especially important for creating better prompts. Try adding lighting, artistic details, and mood.

<div class="flex gap-4">
  <div class="flex-1 text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ok-prompt.png" class="w-full h-auto object-cover rounded-lg">
    <figcaption>cute cat lounges on a leaf in a pool during a peaceful summer afternoon, in lofi art style, illustration.</figcaption>
  </div>
  <div class="flex-1 text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/better-prompt.png" class="w-full h-auto object-cover rounded-lg"/>
    <figcaption>A cute cat lounges on a floating leaf in a sparkling pool during a peaceful summer afternoon. Clear reflections ripple across the water, with sunlight casting soft, smooth highlights. The illustration is detailed and polished, with elegant lines and harmonious colors, evoking a relaxing, serene, and whimsical lofi mood, anime-inspired and visually comforting.</figcaption>
  </div>
</div>

Be specific and add context. Use photography terms like lens type, focal length, camera angles, and depth of field.

> [!TIP]
> Try a [prompt enhancer](https://huggingface.co/models?sort=downloads&search=prompt+enhancer) to help improve your prompt structure.


## Guidance and negatives

Most Diffusers pipelines still steer sampling with classifier-free guidance and an optional `negative_prompt`.

On Stable Diffusion–family pipelines, pass `guidance_scale` and `negative_prompt`. Higher `guidance_scale` follows the prompt more closely. Values that are too high can look unnatural or oversaturated.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
image = pipeline(
    prompt="a cozy reading nook with afternoon light",
    negative_prompt="blurry, low quality, distorted",
    guidance_scale=7.5,
).images[0]
```

On newer checkpoints such as Qwen-Image and Flux, classic CFG is usually `true_cfg_scale` together with `negative_prompt`. Negatives still apply when true CFG is enabled (`true_cfg_scale > 1` and a `negative_prompt`). Their `guidance_scale` argument is distilled or embedded guidance when the transformer supports it, which is not the same. Check the pipeline API for the model you load.

Modular Diffusers can replace the guidance algorithm with a guider. See [Guiders](./guiders).

## Prompt weighting

Prompt weighting makes some words stronger and others weaker. It scales attention scores so you control how much influence each concept has.

Diffusers handles this through `prompt_embeds` and `pooled_prompt_embeds` arguments which take scaled text embedding vectors. Use the [sd_embed](https://github.com/xhinker/sd_embed) library to generate these embeddings. It also supports longer prompts. For Stable Diffusion-family weighting with a simpler syntax, you can also use [Compel](https://github.com/damian0815/compel).

> [!NOTE]
> The sd_embed library only supports Stable Diffusion, Stable Diffusion XL, Stable Diffusion 3, Stable Cascade, and Flux. Prompt weighting does not always help on Flux-class models, which already follow prompts closely.

```shell
uv pip install git+https://github.com/xhinker/sd_embed.git@main
```

Format weighted text with numerical multipliers or parentheses. More parentheses mean stronger weighting.

| format | multiplier |
|---|---|
| `(cat)` | increase by 1.1x |
| `((cat))` | increase by 1.21x |
| `(cat:1.5)` | increase by 1.5x |
| `(cat:0.5)` | set weight to 0.5× |

Create a weighted prompt and pass it to [get_weighted_text_embeddings_sdxl](https://github.com/xhinker/sd_embed/blob/4a47f71150a22942fa606fb741a1c971d95ba56f/src/sd_embed/embedding_funcs.py#L405) to generate embeddings.

> [!TIP]
> You can also pass negative prompts to `negative_prompt_embeds` and `negative_pooled_prompt_embeds`.

```py
import torch
from diffusers import DiffusionPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl

pipeline = DiffusionPipeline.from_pretrained(
    "Lykon/dreamshaper-xl-1-0", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

prompt = """
A (cute cat:1.4) lounges on a (floating leaf:1.2) in a (sparkling pool:1.1) during a peaceful summer afternoon.
Gentle ripples reflect pastel skies, while (sunlight:1.1) casts soft highlights. The illustration is smooth and polished
with elegant, sketchy lines and subtle gradients, evoking a ((whimsical, nostalgic, dreamy lofi atmosphere:2.0)), 
(anime-inspired:1.6), calming, comforting, and visually serene.
"""

prompt_embeds, _, pooled_prompt_embeds, *_ = get_weighted_text_embeddings_sdxl(pipeline, prompt=prompt)
```

Pass the embeddings to `prompt_embeds` and `pooled_prompt_embeds` to generate your image.

```py
image = pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds).images[0]
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/prompt-embed-sdxl.png"/>
</div>

Prompt weighting works with [Textual inversion](./textual_inversion_inference) and [DreamBooth](./dreambooth) adapters too.