<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

> [!WARNING]
> This pipeline is deprecated but it can still be used. However, we won't test the pipeline anymore and won't accept any changes to it. If you run into any issues, reinstall the last Diffusers version that supported this model.

# DiffEdit

[DiffEdit: Diffusion-based semantic image editing with mask guidance](https://huggingface.co/papers/2210.11427) is by Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord.

The abstract from the paper is:

*Image generation has recently seen tremendous advances, with diffusion models allowing to synthesize convincing images for a large variety of text prompts. In this article, we propose DiffEdit, a method to take advantage of text-conditioned diffusion models for the task of semantic image editing, where the goal is to edit an image based on a text query. Semantic image editing is an extension of image generation, with the additional constraint that the generated image should be as similar as possible to a given input image. Current editing methods based on diffusion models usually require to provide a mask, making the task much easier by treating it as a conditional inpainting task. In contrast, our main contribution is able to automatically generate a mask highlighting regions of the input image that need to be edited, by contrasting predictions of a diffusion model conditioned on different text prompts. Moreover, we rely on latent inference to preserve content in those regions of interest and show excellent synergies with mask-based diffusion. DiffEdit achieves state-of-the-art editing performance on ImageNet. In addition, we evaluate semantic image editing in more challenging settings, using images from the COCO dataset as well as text-based generated images.*

The original codebase can be found at [Xiang-cd/DiffEdit-stable-diffusion](https://github.com/Xiang-cd/DiffEdit-stable-diffusion), and you can try it out in this [demo](https://blog.problemsolversguild.com/technical/research/2022/11/02/DiffEdit-Implementation.html).

This pipeline was contributed by [clarencechen](https://github.com/clarencechen). ❤️

The [`StableDiffusionDiffEditPipeline`] requires an image mask and a set of partially inverted latents. The image mask is generated from the [`~StableDiffusionDiffEditPipeline.generate_mask`] function, and includes two parameters, `source_prompt` and `target_prompt`. These parameters determine what to edit in the image. For example, if you want to change a bowl of *fruits* to a bowl of *pears*, then:

```py
source_prompt = "a bowl of fruits"
target_prompt = "a bowl of pears"
```

The partially inverted latents are generated from the [`~StableDiffusionDiffEditPipeline.invert`] function, and it is generally a good idea to include a `prompt` or *caption* describing the image to help guide the inverse latent sampling process. The caption can often be your `source_prompt`, but feel free to experiment with other text descriptions!

Let's load the pipeline, scheduler, inverse scheduler, and enable some optimizations to reduce memory usage:

```py
import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()
```

Load the image to edit:

```py
from diffusers.utils import load_image, make_image_grid

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).resize((768, 768))
raw_image
```

Use the [`~StableDiffusionDiffEditPipeline.generate_mask`] function to generate the image mask. You'll need to pass it the `source_prompt` and `target_prompt` to specify what to edit in the image:

```py
from PIL import Image

source_prompt = "a bowl of fruits"
target_prompt = "a basket of pears"
mask_image = pipeline.generate_mask(
    image=raw_image,
    source_prompt=source_prompt,
    target_prompt=target_prompt,
)
Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
```

Next, create the inverted latents and pass it a caption describing the image:

```py
inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents
```

Finally, pass the image mask and inverted latents to the pipeline. The `target_prompt` becomes the `prompt` now, and the `source_prompt` is used as the `negative_prompt`:

```py
output_image = pipeline(
    prompt=target_prompt,
    mask_image=mask_image,
    image_latents=inv_latents,
    negative_prompt=source_prompt,
).images[0]
mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
make_image_grid([raw_image, mask_image, output_image], rows=1, cols=3)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/blob/main/assets/target.png?raw=true"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">edited image</figcaption>
  </div>
</div>

## Generate source and target embeddings

The source and target embeddings can be automatically generated with the [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model instead of creating them manually.

Load the Flan-T5 model and tokenizer from the 🤗 Transformers library:

```py
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", torch_dtype=torch.float16)
```

Provide some initial text to prompt the model to generate the source and target prompts.

```py
source_concept = "bowl"
target_concept = "basket"

source_text = f"Provide a caption for images containing a {source_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {target_concept}. "
"The captions should be in English and should be no longer than 150 characters."
```

Next, create a utility function to generate the prompts:

```py
@torch.no_grad()
def generate_prompts(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

source_prompts = generate_prompts(source_text)
target_prompts = generate_prompts(target_text)
print(source_prompts)
print(target_prompts)
```

<Tip>

Check out the [generation strategy](https://huggingface.co/docs/transformers/main/en/generation_strategies) guide if you're interested in learning more about strategies for generating different quality text.

</Tip>

Load the text encoder model used by the [`StableDiffusionDiffEditPipeline`] to encode the text. You'll use the text encoder to compute the text embeddings:

```py
import torch
from diffusers import StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

@torch.no_grad()
def embed_prompts(sentences, tokenizer, text_encoder, device="cuda"):
    embeddings = []
    for sent in sentences:
        text_inputs = tokenizer(
            sent,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
        embeddings.append(prompt_embeds)
    return torch.concatenate(embeddings, dim=0).mean(dim=0).unsqueeze(0)

source_embeds = embed_prompts(source_prompts, pipeline.tokenizer, pipeline.text_encoder)
target_embeds = embed_prompts(target_prompts, pipeline.tokenizer, pipeline.text_encoder)
```

Finally, pass the embeddings to the [`~StableDiffusionDiffEditPipeline.generate_mask`] and [`~StableDiffusionDiffEditPipeline.invert`] functions, and pipeline to generate the image:

```diff
  from diffusers import DDIMInverseScheduler, DDIMScheduler
  from diffusers.utils import load_image, make_image_grid
  from PIL import Image

  pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
  pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)

  img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
  raw_image = load_image(img_url).resize((768, 768))

  mask_image = pipeline.generate_mask(
      image=raw_image,
-     source_prompt=source_prompt,
-     target_prompt=target_prompt,
+     source_prompt_embeds=source_embeds,
+     target_prompt_embeds=target_embeds,
  )

  inv_latents = pipeline.invert(
-     prompt=source_prompt,
+     prompt_embeds=source_embeds,
      image=raw_image,
  ).latents

  output_image = pipeline(
      mask_image=mask_image,
      image_latents=inv_latents,
-     prompt=target_prompt,
-     negative_prompt=source_prompt,
+     prompt_embeds=target_embeds,
+     negative_prompt_embeds=source_embeds,
  ).images[0]
  mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L")
  make_image_grid([raw_image, mask_image, output_image], rows=1, cols=3)
```

## Generate a caption for inversion

While you can use the `source_prompt` as a caption to help generate the partially inverted latents, you can also use the [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) model to automatically generate a caption.

Load the BLIP model and processor from the 🤗 Transformers library:

```py
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16, low_cpu_mem_usage=True)
```

Create a utility function to generate a caption from the input image:

```py
@torch.no_grad()
def generate_caption(images, caption_generator, caption_processor):
    text = "a photograph of"

    inputs = caption_processor(images, text, return_tensors="pt").to(device="cuda", dtype=caption_generator.dtype)
    caption_generator.to("cuda")
    outputs = caption_generator.generate(**inputs, max_new_tokens=128)

    # offload caption generator
    caption_generator.to("cpu")

    caption = caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption
```

Load an input image and generate a caption for it using the `generate_caption` function:

```py
from diffusers.utils import load_image

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
raw_image = load_image(img_url).resize((768, 768))
caption = generate_caption(raw_image, model, processor)
```

<div class="flex justify-center">
    <figure>
        <img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"/>
        <figcaption class="text-center">generated caption: "a photograph of a bowl of fruit on a table"</figcaption>
    </figure>
</div>

Now you can drop the caption into the [`~StableDiffusionDiffEditPipeline.invert`] function to generate the partially inverted latents!


## Tips

* The pipeline can generate masks that can be fed into other inpainting pipelines.
* In order to generate an image using this pipeline, both an image mask (source and target prompts can be manually specified or generated, and passed to [`~StableDiffusionDiffEditPipeline.generate_mask`])
and a set of partially inverted latents (generated using [`~StableDiffusionDiffEditPipeline.invert`]) _must_ be provided as arguments when calling the pipeline to generate the final edited image.
* The function [`~StableDiffusionDiffEditPipeline.generate_mask`] exposes two prompt arguments, `source_prompt` and `target_prompt`
that let you control the locations of the semantic edits in the final image to be generated. Let's say,
you wanted to translate from "cat" to "dog". In this case, the edit direction will be "cat -> dog". To reflect
this in the generated mask, you simply have to set the embeddings related to the phrases including "cat" to
`source_prompt` and "dog" to `target_prompt`.
* When generating partially inverted latents using `invert`, assign a caption or text embedding describing the
overall image to the `prompt` argument to help guide the inverse latent sampling process. In most cases, the
source concept is sufficiently descriptive to yield good results, but feel free to explore alternatives.
* When calling the pipeline to generate the final edited image, assign the source concept to `negative_prompt`
and the target concept to `prompt`. Taking the above example, you simply have to set the embeddings related to
the phrases including "cat" to `negative_prompt` and "dog" to `prompt`.
* If you wanted to reverse the direction in the example above, i.e., "dog -> cat", then it's recommended to:
    * Swap the `source_prompt` and `target_prompt` in the arguments to `generate_mask`.
    * Change the input prompt in [`~StableDiffusionDiffEditPipeline.invert`] to include "dog".
    * Swap the `prompt` and `negative_prompt` in the arguments to call the pipeline to generate the final edited image.
* The source and target prompts, or their corresponding embeddings, can also be automatically generated. Please refer to the [DiffEdit](../../using-diffusers/diffedit) guide for more details.

## StableDiffusionDiffEditPipeline
[[autodoc]] StableDiffusionDiffEditPipeline
    - all
    - generate_mask
    - invert
    - __call__

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
