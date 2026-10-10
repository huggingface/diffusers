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

Two checkpoints share [`BriaFibo2Pipeline`] and differ only in how they are called:

| Checkpoint | Steps | Guidance |
|---|---|---|
| turbo, distilled | 4 | 1.0 (no guidance) |
| mopd | 30 | 5.0 |

With guidance on, the default negative prompt is the null caption the model was trained with: the structured caption with every value left empty.

fibo-2 is trained on structured JSON captions and will not work well with freeform text. Serialize the caption compactly, the way it was trained:

```py
import json

prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
```

To turn freeform text into a structured caption, see the prompt-to-JSON models on the [Bria Fibo](https://huggingface.co/briaai/FIBO) page.

## Editing

The same pipeline edits images. Pass the image as `image` and the edit instruction as the prompt:

```py
import torch
from diffusers import BriaFibo2Pipeline
from diffusers.utils import load_image

pipe = BriaFibo2Pipeline.from_pretrained("briaai/fibo-2-turbo-merge", dtype=torch.bfloat16).to("cuda")
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
)
edited = pipe("Make the background a snowy forest", image=image, num_inference_steps=4, guidance_scale=1.0).images[0]
```

- The prompt can also be a JSON caption of the result with an `"edit_instruction"` key, as Bria's prompt-to-JSON models write for edits. The pipeline turns it into the prompt fibo-2 was trained to edit with. It drops empty values and the `aesthetic_score` and `preference_score` fields, which training removed from every caption.
- Up to 5 images can be edited together, for example to put an object from one into another. The instruction refers to them in order as `<image_1>`, `<image_2>` and so on.
- Unless `height` and `width` are given, the result takes the size fibo-2 was trained at for the aspect ratio of the first image, about one megapixel. A single image is cropped to the size of the result. Several images are each cropped to the trained size for their own aspect ratio.
- `mask` marks the region of a single image to regenerate, white on black. That region is greyed out before the image is encoded. fibo-2 was not trained with masks, so masked edits are less reliable than instructions alone.

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
