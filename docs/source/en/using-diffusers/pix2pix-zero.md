# Pix2Pix Zero

Editing real images is challenging because it can be difficult to accurately describe with text the changes you want to make, and because text prompts do not preserve information so you may lose some features in an image you wanted to keep. Pix2Pix Zero addresses these issues by:

1. automatically detecting the *edit direction* based on the CLIP embedding difference between the two groups of generated sentences containing the original and edit word; this eliminates the need for input text prompting
2. a cross-attention map of the original image is used to guide editing to preserve the structure of the original image in the edited images

Pix2Pix Zero is considered a zero-shot image editing piepeline because it does not require any training. You can use it immediately without any input text prompting.

This guide will show you how to use Pix2Pix Zero to edit images.

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries in Colab
#!pip install diffusers transformers accelerate safetensors
```

There are two main parameters in the [`StableDiffusionPix2PixZeroPipeline`] that controls the edit direction, [`source_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/pix2pix_zero#diffusers.StableDiffusionPix2PixZeroPipeline.__call__.source_embeds) and [`target_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/pix2pix_zero#diffusers.StableDiffusionPix2PixZeroPipeline.__call__.target_embeds). To edit an image containing a "cat" to a "dog", you'd set `source_emebds` to the embeddings including "cat" and set `target_emebds` to the embeddings including "dog".

Let's see how this works in practice.

## Text-to-image

Load the pipeline and download the "cat" and "dog" embeddings:

```py
import requests
import torch
from diffusers import DDIMScheduler, StableDiffusionPix2PixZeroPipeline

pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", conditions_input_image=False, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

def download(embedding_url, local_filepath):
    r = requests.get(embedding_url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)

src_embs_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/embeddings_sd_1.4/cat.pt"
target_embs_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/embeddings_sd_1.4/dog.pt"

for url in [src_embs_url, target_embs_url]:
    download(url, url.split("/")[-1])

src_embeds = torch.load(src_embs_url.split("/")[-1])
target_embeds = torch.load(target_embs_url.split("/")[-1])
```

Add an input prompt to generate an image to be edited (notice how the source concept "cat" is used in the prompt instead of describing a target); pass the prompt and embeddings to the pipeline:

<Tip>

The `cross_attention_guidance_amount` controls how much of the structure is preserved in the edited image from the input. A higher value preserves more of the structure. Feel free to experiment with different values!

</Tip>

```py
prompt = "a high resolution painting of a cat in the style of van gogh"

images = pipeline(
    prompt,
    source_embeds=src_embeds,
    target_embeds=target_embeds,
    num_inference_steps=50,
    cross_attention_guidance_amount=0.15,
).images
images[0].save("edited_image_dog.png")
```

<div class="flex justify-center">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pix2pix-van-gogh-dog.png"/>
</div>

## Image-to-image

You can also condition Pix2Pix Zero with an input image. Load the pipeline and the [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) model (you'll see why in a second):

```py
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16, low_cpu_mem_usage=True)

pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    caption_generator=model,
    caption_processor=processor,
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
```

Now load an input image to create the inverted latents from with the [`~StableDiffusionPix2PixZeroPipeline.invert`] function. The inverted latents are the starting point of the generation process. To guide this process, it also helps to add a caption describing the image. You can use the [`~StableDiffusionPix2PixZeroPipeline.generate_caption`] function to create a caption:

```py
import requests
from PIL import Image

img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
caption = pipeline.generate_caption(raw_image)
print(caption)
"a photography of a black and white kitten in a field of daies"
```

Generate the inverted latents:

```py
generator = torch.manual_seed(0)
inv_latents = pipeline.invert(caption, image=raw_image, generator=generator).latents
```

Create several source and target prompts and retrieve their embeddings with the [`~StableDiffusionPix2PixZeroPipeline.get_embeds`] function:

```py
source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]
target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]

source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)
```

Finally, pass the caption, source and target embeddings, and inverted latents to the pipeline to generate an image:

```py
image = pipeline(
    caption,
    source_embeds=source_embeds,
    target_embeds=target_embeds,
    num_inference_steps=50,
    cross_attention_guidance_amount=0.15,
    generator=generator,
    latents=inv_latents,
    negative_prompt=caption,
).images[0]
image.save("edited_image.png")
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pix2pix-dog-daisy.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">edited image</figcaption>
  </div>
</div>

## Generate source and target embeddings

You can also automatically generate your own source and target sentences by using a model like [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5), and then use [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) to compute the text embeddings.

Load the Flan-T5 model and tokenizer from the 🤗 Transformers library:

```py
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
```

Define your source and target concept to prompt the model to generate some captions:

```py
source_concept = "cat"
target_concept = "dog"

source_text = f"Provide a caption for images containing a {source_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {target_concept}. "
"The captions should be in English and should be no longer than 150 characters."
```

Now you can create a function to generate the captions. It is encouraged to play around with the different parameters supported by the [`~transformers.GenerationMixin.generate`] function, like trying different [text generation strategies](https://huggingface.co/docs/transformers/generation_strategies), to produce a caption you're happy with.

```py
def generate_captions(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

source_captions = generate_captions(source_text)
target_captions = generate_captions(target_concept)
print(source_captions)
print(target_captions)
['A white cat posing for a picture with a camera', 'cat sitting next to its owner', 'cat with a kitten in a trough', 'A cat and a dog outside', 'A cat sits outside of a small tree', 'cat is playing with a kitten in the park', 'cats are the most popular animal', 'cat in a kennel', 'A cat is being taken away from the owner.', 'A cat with a blue t-shirt and shorts', 'a cat with a dog and a cat', 'A cat with a small paw in the grass.', 'A black and white cat sitting alone in a cage.', 'Cat in blue coat lying down in a garden', 'A cat with a stuffed toy is inside a cat carrier', 'Two cats are inside a barn']
['Dogs', 'dog', "- the dog's owner", 'pet', 'dogs are a very common pet.', 'terrier', ': "dog"', 'dog, dog breed and dog coat', "i'm trying to get my dog to stay sane", "a lil' dog", 'i like to have fun,', 'dog is a dog breed', 'Dog: Dog  Dog:  Dog: ', 'dog', 'dog and a cat', 'sat']
```

Next, load the text encoder from the Stable Diffusion model to compute the text embedding:

```py
from diffusers import StableDiffusionPix2PixZeroPipeline 

pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder

import torch 

def embed_captions(sentences, tokenizer, text_encoder, device="cuda"):
    with torch.no_grad():
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

source_embeddings = embed_captions(source_captions, tokenizer, text_encoder)
target_embeddings = embed_captions(target_captions, tokenizer, text_encoder)
```

With the embeddings in hand, you can drop them into the pipeline to edit a given image!