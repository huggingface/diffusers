<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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

Prompt weighting works by increasing or decreasing the scale of the text embedding vector that corresponds to its concept in the prompt because you may not necessarily want the model to focus on all concepts equally. The easiest way to prepare the prompt-weighted embeddings is to use [Compel](https://github.com/damian0815/compel), a text prompt-weighting and blending library. Once you have the prompt-weighted embeddings, you can pass them to any pipeline that has a [`prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.prompt_embeds) (and optionally [`negative_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.negative_prompt_embeds)) parameter, such as [`StableDiffusionPipeline`], [`StableDiffusionControlNetPipeline`], and [`StableDiffusionXLPipeline`].

<Tip>

If your favorite pipeline doesn't have a `prompt_embeds` parameter, please open an [issue](https://github.com/huggingface/diffusers/issues/new/choose) so we can add it!

</Tip>

This guide will show you how to weight and blend your prompts with Compel in 🤗 Diffusers.

Before you begin, make sure you have the latest version of Compel installed:

```py
# uncomment to install in Colab
#!pip install compel --upgrade
```

For this guide, let's generate an image with the prompt `"a red cat playing with a ball"` using the [`StableDiffusionPipeline`]:

```py
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_safetensors=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "a red cat playing with a ball"

generator = torch.Generator(device="cpu").manual_seed(33)

image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_0.png"/>
</div>

### Weighting

You'll notice there is no "ball" in the image! Let's use compel to upweight the concept of "ball" in the prompt. Create a [`Compel`](https://github.com/damian0815/compel/blob/main/doc/compel.md#compel-objects) object, and pass it a tokenizer and text encoder:

```py
from compel import Compel

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
```

compel uses `+` or `-` to increase or decrease the weight of a word in the prompt. To increase the weight of "ball":

<Tip>

`+` corresponds to the value `1.1`, `++` corresponds to `1.1^2`, and so on. Similarly, `-` corresponds to `0.9` and `--` corresponds to `0.9^2`. Feel free to experiment with adding more `+` or `-` in your prompt!

</Tip>

```py
prompt = "a red cat playing with a ball++"
```

Pass the prompt to `compel_proc` to create the new prompt embeddings which are passed to the pipeline:

```py
prompt_embeds = compel_proc(prompt)
generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_1.png"/>
</div>

To downweight parts of the prompt, use the `-` suffix:

```py
prompt = "a red------- cat playing with a ball"
prompt_embeds = compel_proc(prompt)

generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"/>
</div>

You can even up or downweight multiple concepts in the same prompt:

```py
prompt = "a red cat++ playing with a ball----"
prompt_embeds = compel_proc(prompt)

generator = torch.manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-pos-neg.png"/>
</div>

### Blending

You can also create a weighted *blend* of prompts by adding `.blend()` to a list of prompts and passing it some weights. Your blend may not always produce the result you expect because it breaks some assumptions about how the text encoder functions, so just have fun and experiment with it!

```py
prompt_embeds = compel_proc('("a red cat playing with a ball", "jungle").blend(0.7, 0.8)')
generator = torch.Generator(device="cuda").manual_seed(33)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-blend.png"/>
</div>

### Conjunction

A conjunction diffuses each prompt independently and concatenates their results by their weighted sum. Add `.and()` to the end of a list of prompts to create a conjunction:

```py
prompt_embeds = compel_proc('["a red cat", "playing with a", "ball"].and()')
generator = torch.Generator(device="cuda").manual_seed(55)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-conj.png"/>
</div>

### Textual inversion

[Textual inversion](../training/text_inversion) is a technique for learning a specific concept from some images which you can use to generate new images conditioned on that concept.

Create a pipeline and use the [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] function to load the textual inversion embeddings (feel free to browse the [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer) for 100+ trained concepts):

```py
import torch
from diffusers import StableDiffusionPipeline
from compel import Compel, DiffusersTextualInversionManager

pipe = StableDiffusionPipeline.from_pretrained(
  "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16,
  use_safetensors=True, variant="fp16").to("cuda")
pipe.load_textual_inversion("sd-concepts-library/midjourney-style")
```

Compel provides a `DiffusersTextualInversionManager` class to simplify prompt weighting with textual inversion. Instantiate `DiffusersTextualInversionManager` and pass it to the `Compel` class:

```py
textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel_proc = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    textual_inversion_manager=textual_inversion_manager)
```

Incorporate the concept to condition a prompt with using the `<concept>` syntax:

```py
prompt_embeds = compel_proc('("A red cat++ playing with a ball <midjourney-style>")')

image = pipe(prompt_embeds=prompt_embeds).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-text-inversion.png"/>
</div>

### DreamBooth

[DreamBooth](../training/dreambooth) is a technique for generating contextualized images of a subject given just a few images of the subject to train on. It is similar to textual inversion, but DreamBooth trains the full model whereas textual inversion only fine-tunes the text embeddings. This means you should use [`~DiffusionPipeline.from_pretrained`] to load the DreamBooth model (feel free to browse the [Stable Diffusion Dreambooth Concepts Library](https://huggingface.co/sd-dreambooth-library) for 100+ trained models):

```py
import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from compel import Compel

pipe = DiffusionPipeline.from_pretrained("sd-dreambooth-library/dndcoverart-v1", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```

Create a `Compel` class with a tokenizer and text encoder, and pass your prompt to it. Depending on the model you use, you'll need to incorporate the model's unique identifier into your prompt. For example, the `dndcoverart-v1` model uses the identifier `dndcoverart`:

```py
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt_embeds = compel_proc('("magazine cover of a dndcoverart dragon, high quality, intricate details, larry elmore art style").and()')
image = pipe(prompt_embeds=prompt_embeds).images[0]
image
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-dreambooth.png"/>
</div>

### Stable Diffusion XL

Stable Diffusion XL (SDXL) has two tokenizers and text encoders so it's usage is a bit different. To address this, you should pass both tokenizers and encoders to the `Compel` class:

```py
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
import torch

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  variant="fp16",
  use_safetensors=True,
  torch_dtype=torch.float16
).to("cuda")

compel = Compel(
  tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
  text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  requires_pooled=[False, True]
)
```

This time, let's upweight "ball" by a factor of 1.5 for the first prompt, and downweight "ball" by 0.6 for the second prompt. The [`StableDiffusionXLPipeline`] also requires [`pooled_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__.pooled_prompt_embeds) (and optionally [`negative_pooled_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__.negative_pooled_prompt_embeds)) so you should pass those to the pipeline along with the conditioning tensors:

```py
# apply weights
prompt = ["a red cat playing with a (ball)1.5", "a red cat playing with a (ball)0.6"]
conditioning, pooled = compel(prompt)

# generate image
generator = [torch.Generator().manual_seed(33) for _ in range(len(prompt))]
images = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=30).images
make_image_grid(images, rows=1, cols=2)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"a red cat playing with a (ball)1.5"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"a red cat playing with a (ball)0.6"</figcaption>
  </div>
</div>
