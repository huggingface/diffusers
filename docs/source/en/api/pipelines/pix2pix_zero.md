<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pix2Pix Zero

[Zero-shot Image-to-Image Translation](https://huggingface.co/papers/2302.03027) is by Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu.

The abstract from the paper is:

*Large-scale text-to-image generative models have shown their remarkable ability to synthesize diverse and high-quality images. However, it is still challenging to directly apply these models for editing real images for two reasons. First, it is hard for users to come up with a perfect text prompt that accurately describes every visual detail in the input image. Second, while existing models can introduce desirable changes in certain regions, they often dramatically alter the input content and introduce unexpected changes in unwanted regions. In this work, we propose pix2pix-zero, an image-to-image translation method that can preserve the content of the original image without manual prompting. We first automatically discover editing directions that reflect desired edits in the text embedding space. To preserve the general content structure after editing, we further propose cross-attention guidance, which aims to retain the cross-attention maps of the input image throughout the diffusion process. In addition, our method does not need additional training for these edits and can directly use the existing pre-trained text-to-image diffusion model. We conduct extensive experiments and show that our method outperforms existing and concurrent works for both real and synthetic image editing.*

You can find additional information about Pix2Pix Zero on the [project page](https://pix2pixzero.github.io/),  [original codebase](https://github.com/pix2pixzero/pix2pix-zero), and try it out in a [demo](https://huggingface.co/spaces/pix2pix-zero-library/pix2pix-zero-demo).

## Tips 

* The pipeline can be conditioned on real input images. Check out the code examples below to know more.
* The pipeline exposes two arguments namely `source_embeds` and `target_embeds`
that let you control the direction of the semantic edits in the final image to be generated. Let's say,
you wanted to translate from "cat" to "dog". In this case, the edit direction will be "cat -> dog". To reflect
this in the pipeline, you simply have to set the embeddings related to the phrases including "cat" to
`source_embeds` and "dog" to `target_embeds`. Refer to the code example below for more details.
* When you're using this pipeline from a prompt, specify the _source_ concept in the prompt. Taking
the above example, a valid input prompt would be: "a high resolution painting of a **cat** in the style of van gough".
* If you wanted to reverse the direction in the example above, i.e., "dog -> cat", then it's recommended to:
    * Swap the `source_embeds` and `target_embeds`.
    * Change the input prompt to include "dog".  
* To learn more about how the source and target embeddings are generated, refer to the [original 
paper](https://arxiv.org/abs/2302.03027). Below, we also provide some directions on how to generate the embeddings.
* Note that the quality of the outputs generated with this pipeline is dependent on how good the `source_embeds` and `target_embeds` are. Please, refer to [this discussion](#generating-source-and-target-embeddings) for some suggestions on the topic.

## Available Pipelines:

| Pipeline | Tasks | Demo
|---|---|:---:|
| [StableDiffusionPix2PixZeroPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py) | *Text-Based Image Editing* | [🤗 Space](https://huggingface.co/spaces/pix2pix-zero-library/pix2pix-zero-demo) |

<!-- TODO: add Colab -->

## Usage example

### Based on an image generated with the input prompt

```python
import requests
import torch

from diffusers import DDIMScheduler, StableDiffusionPix2PixZeroPipeline


def download(embedding_url, local_filepath):
    r = requests.get(embedding_url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)


model_ckpt = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    model_ckpt, conditions_input_image=False, torch_dtype=torch.float16
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")

prompt = "a high resolution painting of a cat in the style of van gogh"
src_embs_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/embeddings_sd_1.4/cat.pt"
target_embs_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/embeddings_sd_1.4/dog.pt"

for url in [src_embs_url, target_embs_url]:
    download(url, url.split("/")[-1])

src_embeds = torch.load(src_embs_url.split("/")[-1])
target_embeds = torch.load(target_embs_url.split("/")[-1])

images = pipeline(
    prompt,
    source_embeds=src_embeds,
    target_embeds=target_embeds,
    num_inference_steps=50,
    cross_attention_guidance_amount=0.15,
).images
images[0].save("edited_image_dog.png")
```

### Based on an input image

When the pipeline is conditioned on an input image, we first obtain an inverted
noise from it using a `DDIMInverseScheduler` with the help of a generated caption. Then 
the inverted noise is used to start the generation process. 

First, let's load our pipeline: 

```py
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline

captioner_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(captioner_id)
model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    sd_model_ckpt,
    caption_generator=model,
    caption_processor=processor,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
```

Then, we load an input image for conditioning and obtain a suitable caption for it: 

```py
import requests
from PIL import Image

img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
caption = pipeline.generate_caption(raw_image)
```

Then we employ the generated caption and the input image to get the inverted noise: 

```py 
generator = torch.manual_seed(0)
inv_latents = pipeline.invert(caption, image=raw_image, generator=generator).latents
```

Now, generate the image with edit directions: 

```py
# See the "Generating source and target embeddings" section below to
# automate the generation of these captions with a pre-trained model like Flan-T5 as explained below.
source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]
target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]

source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)


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

## Generating source and target embeddings 

The authors originally used the [GPT-3 API](https://openai.com/api/) to generate the source and target captions for discovering
edit directions. However, we can also leverage open source and public models for the same purpose.
Below, we provide an end-to-end example with the [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model
for generating captions and [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) for
computing embeddings on the generated captions.  

**1. Load the generation model**:

```py
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
```

**2. Construct a starting prompt**: 

```py
source_concept = "cat"
target_concept = "dog"

source_text = f"Provide a caption for images containing a {source_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {target_concept}. "
"The captions should be in English and should be no longer than 150 characters."
```

Here, we're interested in the "cat -> dog" direction. 

**3. Generate captions**:

We can use a utility like so for this purpose. 

```py
def generate_captions(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

And then we just call it to generate our captions:

```py
source_captions = generate_captions(source_text)
target_captions = generate_captions(target_concept)
```

We encourage you to play around with the different parameters supported by the
`generate()` method ([documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_tf_utils.TFGenerationMixin.generate)) for the generation quality you are looking for.

**4. Load the embedding model**: 

Here, we need to use the same text encoder model used by the subsequent Stable Diffusion model.

```py 
from diffusers import StableDiffusionPix2PixZeroPipeline 

pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
```

**5. Compute embeddings**:

```py 
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

And you're done! [Here](https://colab.research.google.com/drive/1tz2C1EdfZYAPlzXXbTnf-5PRBiR8_R1F?usp=sharing) is a Colab Notebook that you can use to interact with the entire process.

Now, you can use these embeddings directly while calling the pipeline: 

```py
from diffusers import DDIMScheduler

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

images = pipeline(
    prompt,
    source_embeds=source_embeddings,
    target_embeds=target_embeddings,
    num_inference_steps=50,
    cross_attention_guidance_amount=0.15,
).images
images[0].save("edited_image_dog.png")
```

## StableDiffusionPix2PixZeroPipeline
[[autodoc]] StableDiffusionPix2PixZeroPipeline
	- __call__
	- all
