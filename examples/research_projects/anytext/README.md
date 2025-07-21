# AnyTextPipeline

Project page: https://aigcdesigngroup.github.io/homepage_anytext

"AnyText comprises a diffusion pipeline with two primary elements: an auxiliary latent module and a text embedding module. The former uses inputs like text glyph, position, and masked image to generate latent features for text generation or editing. The latter employs an OCR model for encoding stroke data as embeddings, which blend with image caption embeddings from the tokenizer to generate texts that seamlessly integrate with the background. We employed text-control diffusion loss and text perceptual loss for training to further enhance writing accuracy."

> **Note:** Each text line that needs to be generated should be enclosed in double quotes.

For any usage questions, please refer to the [paper](https://huggingface.co/papers/2311.03054).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/tolgacangoz/b87ec9d2f265b448dd947c9d4a0da389/anytext.ipynb)

```py
# This example requires the `anytext_controlnet.py` file:
# !git clone --depth 1 https://github.com/huggingface/diffusers.git
# %cd diffusers/examples/research_projects/anytext
# Let's choose a font file shared by an HF staff:
# !wget https://huggingface.co/spaces/ysharma/TranslateQuotesInImageForwards/resolve/main/arial-unicode-ms.ttf

import torch
from diffusers import DiffusionPipeline
from anytext_controlnet import AnyTextControlNetModel
from diffusers.utils import load_image


anytext_controlnet = AnyTextControlNetModel.from_pretrained("tolgacangoz/anytext-controlnet", torch_dtype=torch.float16,
                                                            variant="fp16",)
pipe = DiffusionPipeline.from_pretrained("tolgacangoz/anytext", font_path="arial-unicode-ms.ttf",
                                          controlnet=anytext_controlnet, torch_dtype=torch.float16,
                                          trust_remote_code=False,  # One needs to give permission to run this pipeline's code
                                          ).to("cuda")

# generate image
prompt = 'photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream'
draw_pos = load_image("https://raw.githubusercontent.com/tyxsspa/AnyText/refs/heads/main/example_images/gen9.png")
# There are two modes: "generate" and "edit". "edit" mode requires `ori_image` parameter for the image to be edited.
image = pipe(prompt, num_inference_steps=20, mode="generate", draw_pos=draw_pos,
             ).images[0]
image
```
