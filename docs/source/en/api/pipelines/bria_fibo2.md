<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Bria Fibo 2.0

Fibo 2.0 is an 8.1B parameter flow-matching transformer from [Bria](https://huggingface.co/briaai) that generates images from structured JSON captions. Its text encoder is [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct): a Perceiver condenses Qwen3-VL's hidden states into gist tokens that join the image tokens in one stream, and five of the transformer's blocks also read Qwen3-VL directly through gated cross-attention.

There are two checkpoints:

- [`briaai/Fibo-2.0-dev`](https://huggingface.co/briaai/Fibo-2.0-dev), the full model for text-to-image and editing, with classifier-free guidance. The pipeline's defaults, 30 steps with guidance 5, are for this checkpoint.
- [`briaai/Fibo-2.0-edit-turbo`](https://huggingface.co/briaai/Fibo-2.0-edit-turbo), distilled for editing in 4 steps without guidance (`guidance_scale=1.0`).

With guidance on, the default negative prompt is the null caption the model was trained with: the structured caption with every value left empty.

## Generating

Fibo 2.0 is trained on structured JSON captions and will not work well with freeform text. Pass the caption as a JSON string:

```py
import json

import torch
from diffusers import BriaFibo2Pipeline

pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-dev", dtype=torch.bfloat16).to("cuda")

caption = {
    "short_description": "A red vintage bicycle leaning against a whitewashed wall on a narrow Mediterranean street.",
    "objects": [
        {
            "description": "A red vintage bicycle with a wicker basket of lemons on its handlebars.",
            "location": "center",
            "relative_size": "large within frame",
        }
    ],
    "background_setting": "A whitewashed wall with a blue wooden door and potted red geraniums.",
    "lighting": {"conditions": "bright morning sun", "direction": "from the right", "shadows": "long and crisp"},
    "aesthetics": {"composition": "centered", "color_scheme": "red, white and blue", "mood_atmosphere": "calm"},
    "photographic_characteristics": {"depth_of_field": "deep", "focus": "sharp on the bicycle", "camera_angle": "eye level"},
    "style_medium": "photograph",
    "context": "A travel photograph.",
    "artistic_style": "realistic",
}
image = pipe(json.dumps(caption)).images[0]
image.save("bicycle.png")
```

The pipeline cleans the caption the way training did: it drops empty values and the `aesthetic_score` and `preference_score` fields, which no training caption had, and serializes the caption compactly. Two short forms were trained too and are kept as they are:

```py
# The caption's short description alone, as plain text
image = pipe("A red vintage bicycle leaning against a whitewashed wall on a narrow Mediterranean street.").images[0]

# Exactly a "short_description" and a "text_render" list. The list is kept even when it's empty, to say the image
# has no text
caption = {"short_description": "A bakery storefront at dawn with a painted sign.", "text_render": [{"text": "FRESH BREAD"}]}
image = pipe(json.dumps(caption)).images[0]
```

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
edited.save("pikachu_snow.png")
```

For faster edits, load the turbo checkpoint and turn guidance off:

```py
import torch
from diffusers import BriaFibo2Pipeline
from diffusers.utils import load_image

pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-edit-turbo", dtype=torch.bfloat16).to("cuda")
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
)
edited = pipe("Make the background a snowy forest", image=image, num_inference_steps=4, guidance_scale=1.0).images[0]
edited.save("pikachu_snow.png")
```

The examples below go on from either one, with its `pipe` and `image`. With the turbo checkpoint, pass `num_inference_steps=4` and `guidance_scale=1.0` to each call.

### The edit prompt

The prompt can also be a JSON object with an `"edit_instruction"` key and a caption of the result, in any of the forms Fibo 2.0 was trained to edit with. The pipeline turns it into the prompt Fibo 2.0 was trained to edit with.

A full caption: the caption's keys beside `"edit_instruction"`, as Bria's prompt-to-JSON models write for edits. The pipeline drops empty values and the `aesthetic_score` and `preference_score` fields, which training removed from every caption.

```py
import json

prompt = {
    "short_description": "A crocheted yellow Pikachu doll sitting in a snowy pine forest.",
    "objects": [{"description": "A crocheted yellow Pikachu doll with red cheeks.", "location": "center"}],
    "background_setting": "A snowy pine forest with falling snow.",
    "edit_instruction": "Make the background a snowy forest",
}
edited = pipe(json.dumps(prompt), image=image).images[0]
```

A short description: a string in a `"structured_caption"` key.

```py
prompt = {
    "structured_caption": "A crocheted yellow Pikachu doll sitting in a snowy pine forest.",
    "edit_instruction": "Make the background a snowy forest",
}
edited = pipe(json.dumps(prompt), image=image).images[0]
```

A short description and its text: exactly a `"short_description"` and a `"text_render"` list. The list is kept even when it's empty, which tells Fibo 2.0 the result has no text. A `"structured_caption"` key can hold this form, or a full caption, too.

```py
prompt = {
    "short_description": "A crocheted yellow Pikachu doll holding a small wooden sign.",
    "text_render": [{"text": "HELLO"}],
    "edit_instruction": "Give the doll a small wooden sign that reads HELLO",
}
edited = pipe(json.dumps(prompt), image=image).images[0]
```

### Several images

Up to 5 images can be edited together, for example to put an object from one into another. The instruction refers to them in order as `<image_1>`, `<image_2>` and so on.

```py
cat = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
edited = pipe("Put the cat from <image_2> next to the doll in <image_1>", image=[image, cat]).images[0]
```

### Output size

Unless `height` and `width` are given, the result takes the size Fibo 2.0 was trained at for the aspect ratio of the first image, about one megapixel. A single image is cropped to the size of the result. Several images are each cropped to the trained size for their own aspect ratio.

```py
edited = pipe("Make the background a snowy forest", image=image, width=1424, height=800).images[0]
```

### Masked edits

`mask` marks the region of a single image to regenerate, white on black, at the image's size. Fibo 2.0 was trained on edits whose image has that region greyed out, and the pipeline greys it out the same way before the image is encoded: flat (128, 128, 128) grey, through the mask made binary at 128.

```py
from PIL import Image, ImageDraw

mask = Image.new("L", image.size, 0)
ImageDraw.Draw(mask).rectangle((image.width // 2, 0, image.width, image.height // 2), fill=255)
edited = pipe("Add a red balloon floating in the sky", image=image, mask=mask).images[0]
```

## Text in images

When the caption asks for text, in its `text_render` entries, Fibo 2.0 tends to render it better if Qwen3-VL also sees each text drawn on its own. The pipeline draws these images by default, the way training did, and they can be turned off. For a caption with text, such as the bakery's above:

```py
prompt = json.dumps(caption)
image = pipe(prompt, render_text_images=False).images[0]
```

Training showed Fibo 2.0 these images for some captions with text and not for others, so it works either way.

- Each `text_render` entry that has `text` gets an image: the text in one line, black on white, in Pillow's built-in font at size 20. A text longer than 256 characters is split into lines of 256.
- Characters that font lacks are drawn with the first of `fonts` that has them. Training used Aileron, then DejaVu Sans, which are in the `fonts` folder of [`briaai/Fibo-2.0-dev`](https://huggingface.co/briaai/Fibo-2.0-dev). Without `fonts`, such characters are drawn as Pillow's built-in font draws them, and the pipeline warns.
- The pipeline marks each entry with an `image_tokens` field for its image, and Qwen3-VL sees the images after any images to edit.
- A generation runs its first steps, while the noise level is above `pipe.text_images_max_noise_level` (0.9), on the caption without the images. An edit uses them at every step.
- A caption without text gets no images, nor does a short description, as in training.

To draw them with the fonts training used:

```py
from huggingface_hub import hf_hub_download

fonts = [
    hf_hub_download("briaai/Fibo-2.0-dev", name, subfolder="fonts")
    for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")
]
image = pipe(prompt, fonts=fonts).images[0]
```

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
