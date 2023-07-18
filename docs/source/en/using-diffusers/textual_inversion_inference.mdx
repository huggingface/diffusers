# Textual inversion

[[open-in-colab]]

The [`StableDiffusionPipeline`] supports textual inversion, a technique that enables a model like Stable Diffusion to learn a new concept from just a few sample images. This gives you more control over the generated images and allows you to tailor the model towards specific concepts. You can get started quickly with a collection of community created concepts in the [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer).

This guide will show you how to run inference with textual inversion using a pre-learned concept from the Stable Diffusion Conceptualizer. If you're interested in teaching a model new concepts with textual inversion, take a look at the [Textual Inversion](./training/text_inversion) training guide.

Login to your Hugging Face account:

```py
from huggingface_hub import notebook_login

notebook_login()
```

Import the necessary libraries, and create a helper function to visualize the generated images:

```py
import os
import torch

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
```

Pick a Stable Diffusion checkpoint and a pre-learned concept from the [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer):

```py
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
repo_id_embeds = "sd-concepts-library/cat-toy"
```

Now you can load a pipeline, and pass the pre-learned concept to it:

```py
pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")

pipeline.load_textual_inversion(repo_id_embeds)
```

Create a prompt with the pre-learned concept by using the special placeholder token `<cat-toy>`, and choose the number of samples and rows of images you'd like to generate:

```py
prompt = "a grafitti in a favela wall with a <cat-toy> on it"

num_samples = 2
num_rows = 2
```

Then run the pipeline (feel free to adjust the parameters like `num_inference_steps` and `guidance_scale` to see how they affect image quality), save the generated images and visualize them with the helper function you created at the beginning:

```py
all_images = []
for _ in range(num_rows):
    images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/textual_inversion_inference.png">
</div>
