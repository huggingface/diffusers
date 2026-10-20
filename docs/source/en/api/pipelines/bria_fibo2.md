<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Bria Fibo 2

fibo-2 is an 8.1B parameter flow-matching transformer from [Bria](https://huggingface.co/briaai) that generates images from structured JSON captions. Its text encoder is [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct): a Perceiver condenses Qwen3-VL's hidden states into gist tokens that join the image tokens in one stream, and five of the transformer's blocks also read Qwen3-VL directly through gated cross-attention.

There are two checkpoints:

- [`briaai/Fibo-2.0-dev`](https://huggingface.co/briaai/Fibo-2.0-dev), the full model for text-to-image and editing, with classifier-free guidance. The pipeline's defaults, 30 steps with guidance 5, are for this checkpoint.
- [`briaai/Fibo-2.0-edit-turbo`](https://huggingface.co/briaai/Fibo-2.0-edit-turbo), distilled for editing in 4 steps without guidance (`guidance_scale=1.0`).

With guidance on, the default negative prompt is the null caption the model was trained with: the structured caption with every value left empty.

fibo-2 is trained on structured JSON captions and will not work well with freeform text. Pass the caption serialized as JSON:

```py
import json

prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
```

The pipeline cleans the caption the way training did: it drops empty values and the `aesthetic_score` and `preference_score` fields, which no training caption had, and serializes the caption compactly. Two short forms were trained too and are kept as they are: the caption's `short_description` alone, as plain text, and a caption of exactly a `"short_description"` and a `"text_render"` list, which keeps the list even when it's empty to say the image has no text.

To turn freeform text into a structured caption, see the prompt-to-JSON models on the [Bria Fibo](https://huggingface.co/briaai/FIBO) page.

## Editing

The same pipeline edits images. Pass the image as `image` and the edit instruction as the prompt:

```py
import torch
from diffusers import BriaFibo2Pipeline
from diffusers.utils import load_image

pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-dev", dtype=torch.bfloat16).to("cuda")
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
)
edited = pipe("Make the background a snowy forest", image=image).images[0]
```

For faster edits, load the turbo checkpoint and turn guidance off:

```py
pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-edit-turbo", dtype=torch.bfloat16).to("cuda")
edited = pipe("Make the background a snowy forest", image=image, num_inference_steps=4, guidance_scale=1.0).images[0]
```

- The prompt can also be a JSON object with an `"edit_instruction"` key and a caption of the result, in any of the forms fibo-2 was trained to edit with. The pipeline turns it into the prompt fibo-2 was trained to edit with.
  - **A full caption:** the caption's keys beside `"edit_instruction"`, as Bria's prompt-to-JSON models write for edits. The pipeline drops empty values and the `aesthetic_score` and `preference_score` fields, which training removed from every caption.
  - **A short description:** a string in a `"structured_caption"` key, for example `{"structured_caption": "A woman in a red coat on a snowy street.", "edit_instruction": "Make the coat green."}`.
  - **A short description and its text:** exactly a `"short_description"` and a `"text_render"` list. The list is kept even when it's empty, which tells fibo-2 the result has no text.

  A `"structured_caption"` key can hold either JSON form too.
- Up to 5 images can be edited together, for example to put an object from one into another. The instruction refers to them in order as `<image_1>`, `<image_2>` and so on.
- Unless `height` and `width` are given, the result takes the size fibo-2 was trained at for the aspect ratio of the first image, about one megapixel. A single image is cropped to the size of the result. Several images are each cropped to the trained size for their own aspect ratio.
- `mask` marks the region of a single image to regenerate, white on black. fibo-2 was trained on edits whose image has that region greyed out, and the pipeline greys it out the same way before the image is encoded: flat (128, 128, 128) grey, through the mask made binary at 128.

## Text in images

When the caption asks for text, in its `text_render` entries, fibo-2 tends to render it better if Qwen3-VL also sees each text drawn on its own. The pipeline draws these images by default, the way training did, and they can be turned off:

```py
image = pipe(prompt, render_text_images=False).images[0]
```

Training showed fibo-2 these images for some captions with text and not for others, so it works either way.

- Each `text_render` entry that has `text` gets an image: the text in one line, black on white, in Pillow's built-in font at size 20. A text longer than 256 characters is split into lines of 256.
- Characters that font lacks are drawn with the first of `fonts` that has them. Training used Aileron, then DejaVu Sans, which are in the `fonts` folder of [`briaai/Fibo-2.0-dev`](https://huggingface.co/briaai/Fibo-2.0-dev). Without `fonts`, such characters are drawn as Pillow's built-in font draws them, and the pipeline warns:

  ```py
  from huggingface_hub import hf_hub_download

  fonts = [
      hf_hub_download("briaai/Fibo-2.0-dev", name, subfolder="fonts")
      for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")
  ]
  image = pipe(prompt, fonts=fonts).images[0]
  ```

- The pipeline marks each entry with an `image_tokens` field for its image, and Qwen3-VL sees the images after any images to edit.
- A generation runs its first steps, while the noise level is above `pipe.text_images_max_noise_level` (0.9), on the caption without the images. An edit uses them at every step.
- A caption without text gets no images, nor does a short description, as in training.

To draw the images yourself, pass them as `text_images`, one per `text_render` entry that has `text`, in order, for a single prompt. They take precedence over the pipeline's own. Images drawn differently from training's are out of distribution.

## Attention backends

The transformer runs on any of the diffusers [attention backends](../../optimization/attention_backends), for example FlashAttention-3 on Hopper GPUs such as the H100:

```py
pipe.transformer.set_attention_backend("_flash_3_varlen_hub")
```

`_flash_3_hub` also works for a single prompt, but it doesn't take masks, so it can't run a batch of prompts of different lengths (see below).

## Batching

Several prompts can be generated in one call. Prompts of different lengths are padded, and the padding is masked out of every attention layer, so such a batch needs an attention backend that takes a mask: the default one, or a varlen backend such as `flash_varlen`. In float32, a batch reproduces the images of the separate calls; in bfloat16 the attention kernels differ between padded and unpadded inputs, so batched images can differ from single calls in fine details.

## BriaFibo2Pipeline

[[autodoc]] BriaFibo2Pipeline
	- all
	- __call__
