<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Prompting

Prompting describes what a model should generate. A good prompt should be detailed, specific, and structured in order to generate better images or videos.

This guide covers general best practices for writing prompts and introduce a few prompt-enhancing techniques.

## Writing good prompts

A good prompt foundation should include the following elements.

1. <span class="underline decoration-wavy decoration-blue-500 decoration-2 underline-offset-4">Subject</span> is what you want to generate an image or video of. It is the main focus and you should generally begin your prompt with the subject.
2. <span class="underline decoration-wavy decoration-purple-500 decoration-2 underline-offset-4">Style</span> describes the medium or aesthetic of the image or video. What do you want it to look like?
3. <span class="underline decoration-wavy decoration-green-500 decoration-2 underline-offset-4">Context</span> adds details to the image or video. For example, what is the subject doing and what is the setting and mood?

Combine these elements into a structured narrative instead of a list of keywords. Modern models have powerful text encoders that have better language understanding. Start with a short prompt, and then iterate on it.

To generate an even better image, enhance the prompt with additional details such as the vibe, specific visual details like lighting, and artistic details.

<div class="flex gap-4">
  <div class="flex-1 text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ok-prompt.png" class="w-full h-auto object-cover rounded-lg">
    <figcaption class="mt-2 text-sm text-gray-500">A <span class="underline decoration-wavy decoration-blue-500 decoration-2 underline-offset-1">cute cat</span> <span class="underline decoration-wavy decoration-green-500 decoration-2 underline-offset-1">lounges on a leaf in a pool during a peaceful summer afternoon</span>, in <span class="underline decoration-wavy decoration-purple-500 decoration-2 underline-offset-1">lofi art style, illustration</span>.</figcaption>
  </div>
  <div class="flex-1 text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/better-prompt.png" class="w-full h-auto object-cover rounded-lg"/>
    <figcaption class="mt-2 text-sm text-gray-500">A cute cat lounges on a floating leaf in a sparkling pool during a peaceful summer afternoon. Clear reflections ripple across the water, with sunlight casting soft, smooth highlights. The illustration is detailed and polished, with elegant lines and harmonious colors, evoking a relaxing, serene, and whimsical lofi mood, anime-inspired and visually comforting.</figcaption>
  </div>
</div>

Try to be as specific and precise as possible and provide as much context as you can to help a model understand what to generate. For example, consider using specific photography or cinematographic terms such as lens type, focal lengths, camera angles and scene depths.

## Prompt weighting

Prompt weighting emphasizes and de-emphasizes certain words or concepts in a prompt to help a model focus. It works by scaling the attention score for a token by some value, allowing you to fine-tune how much influence a word or concept has.

Diffusers supports prompt weighting through the `prompt_embeds` and `pooled_prompt_embeds` arguments. These arguments accept the scaled text embedding vector. To get these embeddings, we recommend using the [sd_embed](https://github.com/xhinker/sd_embed) library, which also supports longer prompts. Run the command below to install the sd_embed library.

```py
!uv pip install git+https://github.com/xhinker/sd_embed.git@main
```

To apply prompt weighting, format the text a numerical multiplier or parentheses (more parentheses increases weighting). Refer to the table below for some examples.

| format | multiplier |
|---|---|
| `(cat)` | increase by 1.1x |
| `((cat))` | increase by 1.21x |
| `(cat:1.5)` | increase by 1.5x |
| `(cat:0.5)` | decrease by 4x |

Create a weighed prompt and pass it to the [get_weighted_text_embeddings_sdxl](https://github.com/xhinker/sd_embed/blob/4a47f71150a22942fa606fb741a1c971d95ba56f/src/sd_embed/embedding_funcs.py#L405) function to generate the embeddings.

> [!TIP]
> You could also pass negative prompts to `negative_prompt_embeds` and `negative_pooled_prompt_embeds`.

```py
import torch
from diffusers import DiffusionPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl

pipeline = DiffusionPipeline.from_pretrained(
  "Lykon/dreamshaper-xl-1-0", torch_dtype=torch.bfloat16, device_map="cuda"
)
prompt="""
A (cute cat:1.4) lounges on a (floating leaf:1.2) in a (sparkling pool:1.1) during a peaceful summer afternoon.
Gentle ripples reflect pastel skies, while (sunlight:1.1) casts soft highlights. The illustration is smooth and polished
with elegant, sketchy lines and subtle gradients, evoking a ((whimsical, nostalgic, dreamy lofi atmosphere:2.0)), 
(anime-inspired:1.6), calming, comforting, and visually serene.
"""
(prompt_embeds, _, pooled_prompt_embeds, *_) = get_weighted_text_embeddings_sdxl(pipeline, prompt=prompt)
```

Pass the embeddings to the `prompt_embeds` and `pooled_prompt_embeds` arguments to generate an image.

```py
pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds).images[0]
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/prompt-embed-sdxl.png"/>
</div>

Prompt weighting is also supported for adapters like [Textual inversion](./textual_inversion_inference) and [DreamBooth](./dreambooth).

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