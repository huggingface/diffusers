<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Textual inversion

[[open-in-colab]]

The [`StableDiffusionPipeline`] supports textual inversion, a technique that enables a model like Stable Diffusion to learn a new concept from just a few sample images. This gives you more control over the generated images and allows you to tailor the model towards specific concepts. You can get started quickly with a collection of community created concepts in the [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer).

This guide will show you how to run inference with textual inversion using a pre-learned concept from the Stable Diffusion Conceptualizer. If you're interested in teaching a model new concepts with textual inversion, take a look at the [Textual Inversion](../training/text_inversion) training guide.

Import the necessary libraries:

```py
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid
```

## Stable Diffusion 1 and 2

Pick a Stable Diffusion checkpoint and a pre-learned concept from the [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer):

```py
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
repo_id_embeds = "sd-concepts-library/cat-toy"
```

Now you can load a pipeline, and pass the pre-learned concept to it:

```py
pipeline = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipeline.load_textual_inversion(repo_id_embeds)
```

Create a prompt with the pre-learned concept by using the special placeholder token `<cat-toy>`, and choose the number of samples and rows of images you'd like to generate:

```py
prompt = "a grafitti in a favela wall with a <cat-toy> on it"

num_samples_per_row = 2
num_rows = 2
```

Then run the pipeline (feel free to adjust the parameters like `num_inference_steps` and `guidance_scale` to see how they affect image quality), save the generated images and visualize them with the helper function you created at the beginning:

```py
all_images = []
for _ in range(num_rows):
    images = pipeline(prompt, num_images_per_prompt=num_samples_per_row, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = make_image_grid(all_images, num_rows, num_samples_per_row)
grid
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/textual_inversion_inference.png">
</div>

## Stable Diffusion XL

Stable Diffusion XL (SDXL) can also use textual inversion vectors for inference. In contrast to Stable Diffusion 1 and 2, SDXL has two text encoders so you'll need two textual inversion embeddings - one for each text encoder model.

Let's download the SDXL textual inversion embeddings and have a closer look at it's structure:

```py
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

file = hf_hub_download("dn118/unaestheticXL", filename="unaestheticXLv31.safetensors")
state_dict = load_file(file)
state_dict
```

```
{'clip_g': tensor([[ 0.0077, -0.0112,  0.0065,  ...,  0.0195,  0.0159,  0.0275],
         ...,
         [-0.0170,  0.0213,  0.0143,  ..., -0.0302, -0.0240, -0.0362]],
 'clip_l': tensor([[ 0.0023,  0.0192,  0.0213,  ..., -0.0385,  0.0048, -0.0011],
         ...,
         [ 0.0475, -0.0508, -0.0145,  ...,  0.0070, -0.0089, -0.0163]],
```

There are two tensors, `"clip_g"` and `"clip_l"`.
`"clip_g"` corresponds to the bigger text encoder in SDXL and refers to 
`pipe.text_encoder_2` and `"clip_l"` refers to `pipe.text_encoder`.

Now you can load each tensor separately by passing them along with the correct text encoder and tokenizer
to [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

pipe.load_textual_inversion(state_dict["clip_g"], token="unaestheticXLv31", text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
pipe.load_textual_inversion(state_dict["clip_l"], token="unaestheticXLv31", text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

# the embedding should be used as a negative embedding, so we pass it as a negative prompt
generator = torch.Generator().manual_seed(33)
image = pipe("a woman standing in front of a mountain", negative_prompt="unaestheticXLv31", generator=generator).images[0]
image
```
