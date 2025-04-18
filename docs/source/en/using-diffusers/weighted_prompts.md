<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Prompt techniques

[[open-in-colab]]

Prompts are important because they describe what you want a diffusion model to generate. The best prompts are detailed, specific, and well-structured to help the model realize your vision. But crafting a great prompt takes time and effort and sometimes it may not be enough because language and words can be imprecise. This is where you need to boost your prompt with other techniques, such as prompt enhancing and prompt weighting, to get the results you want.

This guide will show you how you can use these prompt techniques to generate high-quality images with lower effort and adjust the weight of certain keywords in a prompt.

## Prompt engineering

> [!TIP]
> This is not an exhaustive guide on prompt engineering, but it will help you understand the necessary parts of a good prompt. We encourage you to continue experimenting with different prompts and combine them in new ways to see what works best. As you write more prompts, you'll develop an intuition for what works and what doesn't!

New diffusion models do a pretty good job of generating high-quality images from a basic prompt, but it is still important to create a well-written prompt to get the best results. Here are a few tips for writing a good prompt:

1. What is the image *medium*? Is it a photo, a painting, a 3D illustration, or something else?
2. What is the image *subject*? Is it a person, animal, object, or scene?
3. What *details* would you like to see in the image? This is where you can get really creative and have a lot of fun experimenting with different words to bring your image to life. For example, what is the lighting like? What is the vibe and aesthetic? What kind of art or illustration style are you looking for? The more specific and precise words you use, the better the model will understand what you want to generate.

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/plain-prompt.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"A photo of a banana-shaped couch in a living room"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/detail-prompt.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"A vibrant yellow banana-shaped couch sits in a cozy living room, its curve cradling a pile of colorful cushions. on the wooden floor, a patterned rug adds a touch of eclectic charm, and a potted plant sits in the corner, reaching towards the sunlight filtering through the windows"</figcaption>
  </div>
</div>

## Prompt enhancing with GPT2

Prompt enhancing is a technique for quickly improving prompt quality without spending too much effort constructing one. It uses a model like GPT2 pretrained on Stable Diffusion text prompts to automatically enrich a prompt with additional important keywords to generate high-quality images.

The technique works by curating a list of specific keywords and forcing the model to generate those words to enhance the original prompt. This way, your prompt can be "a cat" and GPT2 can enhance the prompt to "cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain quality sharp focus beautiful detailed intricate stunning amazing epic".

> [!TIP]
> You should also use a [*offset noise*](https://www.crosslabs.org//blog/diffusion-with-offset-noise) LoRA to improve the contrast in bright and dark images and create better lighting overall. This [LoRA](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_offset_example-lora_1.0.safetensors) is available from [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0).

Start by defining certain styles and a list of words (you can check out a more comprehensive list of [words](https://hf.co/LykosAI/GPT-Prompt-Expansion-Fooocus-v2/blob/main/positive.txt) and [styles](https://github.com/lllyasviel/Fooocus/tree/main/sdxl_styles) used by Fooocus) to enhance a prompt with.

```py
import torch
from transformers import GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList
from diffusers import StableDiffusionXLPipeline

styles = {
    "cinematic": "cinematic film still of {prompt}, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
    "anime": "anime artwork of {prompt}, anime style, key visual, vibrant, studio anime, highly detailed",
    "photographic": "cinematic photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed",
    "comic": "comic of {prompt}, graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "lineart": "line art drawing {prompt}, professional, sleek, modern, minimalist, graphic, line art, vector graphics",
    "pixelart": " pixel-art {prompt}, low-res, blocky, pixel art style, 8-bit graphics",
}

words = [
    "aesthetic", "astonishing", "beautiful", "breathtaking", "composition", "contrasted", "epic", "moody", "enhanced",
    "exceptional", "fascinating", "flawless", "glamorous", "glorious", "illumination", "impressive", "improved",
    "inspirational", "magnificent", "majestic", "hyperrealistic", "smooth", "sharp", "focus", "stunning", "detailed",
    "intricate", "dramatic", "high", "quality", "perfect", "light", "ultra", "highly", "radiant", "satisfying",
    "soothing", "sophisticated", "stylish", "sublime", "terrific", "touching", "timeless", "wonderful", "unbelievable",
    "elegant", "awesome", "amazing", "dynamic", "trendy",
]
```

You may have noticed in the `words` list, there are certain words that can be paired together to create something more meaningful. For example, the words "high" and "quality" can be combined to create "high quality". Let's pair these words together and remove the words that can't be paired.

```py
word_pairs = ["highly detailed", "high quality", "enhanced quality", "perfect composition", "dynamic light"]

def find_and_order_pairs(s, pairs):
    words = s.split()
    found_pairs = []
    for pair in pairs:
        pair_words = pair.split()
        if pair_words[0] in words and pair_words[1] in words:
            found_pairs.append(pair)
            words.remove(pair_words[0])
            words.remove(pair_words[1])

    for word in words[:]:
        for pair in pairs:
            if word in pair.split():
                words.remove(word)
                break
    ordered_pairs = ", ".join(found_pairs)
    remaining_s = ", ".join(words)
    return ordered_pairs, remaining_s
```

Next, implement a custom [`~transformers.LogitsProcessor`] class that assigns tokens in the `words` list a value of 0 and assigns tokens not in the `words` list a negative value so they aren't picked during generation. This way, generation is biased towards words in the `words` list. After a word from the list is used, it is also assigned a negative value so it isn't picked again.

```py
class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def __call__(self, input_ids, scores):
        if len(input_ids.shape) == 2:
            last_token_id = input_ids[0, -1]
            self.bias[last_token_id] = -1e10
        return scores + self.bias

word_ids = [tokenizer.encode(word, add_prefix_space=True)[0] for word in words]
bias = torch.full((tokenizer.vocab_size,), -float("Inf")).to("cuda")
bias[word_ids] = 0
processor = CustomLogitsProcessor(bias)
processor_list = LogitsProcessorList([processor])
```

Combine the prompt and the `cinematic` style prompt defined in the `styles` dictionary earlier.

```py
prompt = "a cat basking in the sun on a roof in Turkey"
style = "cinematic"

prompt = styles[style].format(prompt=prompt)
prompt
"cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
```

Load a GPT2 tokenizer and model from the [Gustavosta/MagicPrompt-Stable-Diffusion](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion) checkpoint (this specific checkpoint is trained to generate prompts) to enhance the prompt.

```py
tokenizer = GPT2Tokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
model = GPT2LMHeadModel.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion", torch_dtype=torch.float16).to(
    "cuda"
)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
token_count = inputs["input_ids"].shape[1]
max_new_tokens = 50 - token_count

generation_config = GenerationConfig(
    penalty_alpha=0.7,
    top_k=50,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
    do_sample=True,
)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        generation_config=generation_config,
        logits_processor=proccesor_list,
    )
```

Then you can combine the input prompt and the generated prompt. Feel free to take a look at what the generated prompt (`generated_part`) is, the word pairs that were found (`pairs`), and the remaining words (`words`). This is all packed together in the `enhanced_prompt`.

```py
output_tokens = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
input_part, generated_part = output_tokens[0][: len(prompt)], output_tokens[0][len(prompt) :]
pairs, words = find_and_order_pairs(generated_part, word_pairs)
formatted_generated_part = pairs + ", " + words
enhanced_prompt = input_part + ", " + formatted_generated_part
enhanced_prompt
["cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain quality sharp focus beautiful detailed intricate stunning amazing epic"]
```

Finally, load a pipeline and the offset noise LoRA with a *low weight* to generate an image with the enhanced prompt.

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pipeline.load_lora_weights(
    "stabilityai/stable-diffusion-xl-base-1.0",
    weight_name="sd_xl_offset_example-lora_1.0.safetensors",
    adapter_name="offset",
)
pipeline.set_adapters(["offset"], adapter_weights=[0.2])

image = pipeline(
    enhanced_prompt,
    width=1152,
    height=896,
    guidance_scale=7.5,
    num_inference_steps=25,
).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"a cat basking in the sun on a roof in Turkey"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/enhanced-prompt.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"</figcaption>
  </div>
</div>

## Prompt weighting

Prompt weighting provides a way to emphasize or de-emphasize certain parts of a prompt, allowing for more control over the generated image. A prompt can include several concepts, which gets turned into contextualized text embeddings. The embeddings are used by the model to condition its cross-attention layers to generate an image (read the Stable Diffusion [blog post](https://huggingface.co/blog/stable_diffusion) to learn more about how it works).

Prompt weighting works by increasing or decreasing the scale of the text embedding vector that corresponds to its concept in the prompt because you may not necessarily want the model to focus on all concepts equally. The easiest way to prepare the prompt embeddings is to use [Stable Diffusion Long Prompt Weighted Embedding](https://github.com/xhinker/sd_embed) (sd_embed). Once you have the prompt-weighted embeddings, you can pass them to any pipeline that has a [prompt_embeds](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.prompt_embeds) (and optionally [negative_prompt_embeds](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.negative_prompt_embeds)) parameter, such as [`StableDiffusionPipeline`], [`StableDiffusionControlNetPipeline`], and [`StableDiffusionXLPipeline`].

<Tip>

If your favorite pipeline doesn't have a `prompt_embeds` parameter, please open an [issue](https://github.com/huggingface/diffusers/issues/new/choose) so we can add it!

</Tip>

This guide will show you how to weight your prompts with sd_embed.

Before you begin, make sure you have the latest version of sd_embed installed:

```bash
pip install git+https://github.com/xhinker/sd_embed.git@main
```

For this example, let's use [`StableDiffusionXLPipeline`].

```py
from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("Lykon/dreamshaper-xl-1-0", torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
```

To upweight or downweight a concept, surround the text with parentheses. More parentheses applies a heavier weight on the text. You can also append a numerical multiplier to the text to indicate how much you want to increase or decrease its weights by.

| format | multiplier |
|---|---|
| `(hippo)` | increase by 1.1x |
| `((hippo))` | increase by 1.21x |
| `(hippo:1.5)` | increase by 1.5x |
| `(hippo:0.5)` | decrease by 4x |

Create a prompt and use a combination of parentheses and numerical multipliers to upweight various text.

```py
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl

prompt = """A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus. 
This imaginative creature features the distinctive, bulky body of a hippo, 
but with a texture and appearance resembling a golden-brown, crispy waffle. 
The creature might have elements like waffle squares across its skin and a syrup-like sheen. 
It's set in a surreal environment that playfully combines a natural water habitat of a hippo with elements of a breakfast table setting, 
possibly including oversized utensils or plates in the background. 
The image should evoke a sense of playful absurdity and culinary fantasy.
"""

neg_prompt = """\
skin spots,acnes,skin blemishes,age spot,(ugly:1.2),(duplicate:1.2),(morbid:1.21),(mutilated:1.2),\
(tranny:1.2),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.2),(bad proportions:1.3),\
extra limbs,(disfigured:1.2),(missing arms:1.2),(extra legs:1.2),(fused fingers:1.5),\
(too many fingers:1.5),(unclear eyes:1.2),lowers,bad hands,missing fingers,extra digit,\
bad hands,missing fingers,(extra arms and legs),(worst quality:2),(low quality:2),\
(normal quality:2),lowres,((monochrome)),((grayscale))
"""
```

Use the `get_weighted_text_embeddings_sdxl` function to generate the prompt embeddings and the negative prompt embeddings. It'll also generated the pooled and negative pooled prompt embeddings since you're using the SDXL model.

> [!TIP]
> You can safely ignore the error message below about the token index length exceeding the models maximum sequence length. All your tokens will be used in the embedding process.
>
> ```
> Token indices sequence length is longer than the specified maximum sequence length for this model
> ```

```py
( 
  prompt_embeds,
  prompt_neg_embeds,
  pooled_prompt_embeds,
  negative_pooled_prompt_embeds
) = get_weighted_text_embeddings_sdxl(
    pipe,
    prompt=prompt,
    neg_prompt=neg_prompt
)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=prompt_neg_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    num_inference_steps=30,
    height=1024,
    width=1024 + 512,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(2)
).images[0]
image
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sd_embed_sdxl.png"/>
</div>

> [!TIP]
> Refer to the [sd_embed](https://github.com/xhinker/sd_embed) repository for additional details about long prompt weighting for FLUX.1, Stable Cascade, and Stable Diffusion 1.5.

### Textual inversion

[Textual inversion](../training/text_inversion) is a technique for learning a specific concept from some images which you can use to generate new images conditioned on that concept.

Create a pipeline and use the [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] function to load the textual inversion embeddings (feel free to browse the [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer) for 100+ trained concepts):

```py
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
  "stable-diffusion-v1-5/stable-diffusion-v1-5",
  torch_dtype=torch.float16,
).to("cuda")
pipe.load_textual_inversion("sd-concepts-library/midjourney-style")
```

Add the `<midjourney-style>` text to the prompt to trigger the textual inversion.

```py
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15

prompt = """<midjourney-style> A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus. 
This imaginative creature features the distinctive, bulky body of a hippo, 
but with a texture and appearance resembling a golden-brown, crispy waffle. 
The creature might have elements like waffle squares across its skin and a syrup-like sheen. 
It's set in a surreal environment that playfully combines a natural water habitat of a hippo with elements of a breakfast table setting, 
possibly including oversized utensils or plates in the background. 
The image should evoke a sense of playful absurdity and culinary fantasy.
"""

neg_prompt = """\
skin spots,acnes,skin blemishes,age spot,(ugly:1.2),(duplicate:1.2),(morbid:1.21),(mutilated:1.2),\
(tranny:1.2),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.2),(bad proportions:1.3),\
extra limbs,(disfigured:1.2),(missing arms:1.2),(extra legs:1.2),(fused fingers:1.5),\
(too many fingers:1.5),(unclear eyes:1.2),lowers,bad hands,missing fingers,extra digit,\
bad hands,missing fingers,(extra arms and legs),(worst quality:2),(low quality:2),\
(normal quality:2),lowres,((monochrome)),((grayscale))
"""
```

Use the `get_weighted_text_embeddings_sd15` function to generate the prompt embeddings and the negative prompt embeddings.

```py
( 
  prompt_embeds,
  prompt_neg_embeds,
) = get_weighted_text_embeddings_sd15(
    pipe,
    prompt=prompt,
    neg_prompt=neg_prompt
)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=prompt_neg_embeds,
    height=768,
    width=896,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(2)
).images[0]
image
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sd_embed_textual_inversion.png"/>
</div>

### DreamBooth

[DreamBooth](../training/dreambooth) is a technique for generating contextualized images of a subject given just a few images of the subject to train on. It is similar to textual inversion, but DreamBooth trains the full model whereas textual inversion only fine-tunes the text embeddings. This means you should use [`~DiffusionPipeline.from_pretrained`] to load the DreamBooth model (feel free to browse the [Stable Diffusion Dreambooth Concepts Library](https://huggingface.co/sd-dreambooth-library) for 100+ trained models):

```py
import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

pipe = DiffusionPipeline.from_pretrained("sd-dreambooth-library/dndcoverart-v1", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```

Depending on the model you use, you'll need to incorporate the model's unique identifier into your prompt. For example, the `dndcoverart-v1` model uses the identifier `dndcoverart`:

```py
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15

prompt = """dndcoverart of A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus. 
This imaginative creature features the distinctive, bulky body of a hippo, 
but with a texture and appearance resembling a golden-brown, crispy waffle. 
The creature might have elements like waffle squares across its skin and a syrup-like sheen. 
It's set in a surreal environment that playfully combines a natural water habitat of a hippo with elements of a breakfast table setting, 
possibly including oversized utensils or plates in the background. 
The image should evoke a sense of playful absurdity and culinary fantasy.
"""

neg_prompt = """\
skin spots,acnes,skin blemishes,age spot,(ugly:1.2),(duplicate:1.2),(morbid:1.21),(mutilated:1.2),\
(tranny:1.2),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.2),(bad proportions:1.3),\
extra limbs,(disfigured:1.2),(missing arms:1.2),(extra legs:1.2),(fused fingers:1.5),\
(too many fingers:1.5),(unclear eyes:1.2),lowers,bad hands,missing fingers,extra digit,\
bad hands,missing fingers,(extra arms and legs),(worst quality:2),(low quality:2),\
(normal quality:2),lowres,((monochrome)),((grayscale))
"""

(
    prompt_embeds
    , prompt_neg_embeds
) = get_weighted_text_embeddings_sd15(
    pipe
    , prompt = prompt
    , neg_prompt = neg_prompt
)
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sd_embed_dreambooth.png"/>
</div>
