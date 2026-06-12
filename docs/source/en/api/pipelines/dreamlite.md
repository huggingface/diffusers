<!--Copyright 2026 The ByteDance Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DreamLite

DreamLite is a text-to-image and image-editing model from ByteDance. It pairs a custom 2D U-Net
(`DreamLiteUNetModel`) with the `Qwen3-VL` multimodal encoder as its prompt / image-instruction encoder,
and uses an `AutoencoderTiny` (TAESD-style) VAE for fast latent encode/decode.

Two pipelines are exposed:

| Pipeline | Modes | CFG | Use case |
|---|---|---|---|
| [`DreamLitePipeline`] | text-to-image **and** image-editing (auto-selected by whether `image` is `None`) | 3-branch dual CFG (`guidance_scale` on text branch, `image_guidance_scale` on image branch, à la InstructPix2Pix) | Highest quality |
| [`DreamLiteMobilePipeline`] | text-to-image **and** image-editing (auto-selected by whether `image` is `None`) | None — distilled, single UNet forward per step | On-device / low-latency |

Official checkpoints:

* Base model: [carlofkl/DreamLite-base](https://huggingface.co/carlofkl/DreamLite-base)
* Distilled mobile model: [carlofkl/DreamLite-mobile](https://huggingface.co/carlofkl/DreamLite-mobile)

> [!TIP]
> Both pipelines auto-detect text-to-image vs. image-editing mode from whether the `image` argument is
> provided. There is no separate `Img2Img` class.

> [!TIP]
> When loading an input image for editing, prefer `diffusers.utils.load_image(...)` over raw `PIL.Image.open(...)`.
> `load_image` enforces an RGB conversion and applies EXIF orientation, both of which the pipeline assumes.
> A plain `Image.open` of an RGBA / palette / EXIF-rotated source will silently produce a different latent
> conditioning and degrade output quality.

## Text-to-image (Base)

```python
import torch
from diffusers import DreamLitePipeline

pipe = DreamLitePipeline.from_pretrained("carlofkl/DreamLite-base", revision="diffusers", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    prompt="a dog running on the grass",
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_t2i.png")
```

## Image editing (Base)

Pass an `image` to enter edit mode. Both `guidance_scale` (text branch) and `image_guidance_scale`
(image branch) are active here.

```python
import torch
from diffusers import DreamLitePipeline
from diffusers.utils import load_image

pipe = DreamLitePipeline.from_pretrained("carlofkl/DreamLite-base", revision="diffusers", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

source = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
    prompt="turn the cat into a corgi",
    image=source,
    height=1024,
    width=1024,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_edit.png")
```

## Text-to-image (Mobile)

The mobile pipeline is distilled and skips CFG entirely — a single UNet forward per step. It accepts the
same `prompt` / `height` / `width` / `num_inference_steps` arguments, but **ignores** `guidance_scale` and
`image_guidance_scale` if passed (a warning is logged).

```python
import torch
from diffusers import DreamLiteMobilePipeline

pipe = DreamLiteMobilePipeline.from_pretrained("carlofkl/DreamLite-mobile", revision="diffusers", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    prompt="a dog running on the grass",
    height=1024,
    width=1024,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_mobile_t2i.png")
```

## Image editing (Mobile)

```python
import torch
from diffusers import DreamLiteMobilePipeline
from diffusers.utils import load_image

pipe = DreamLiteMobilePipeline.from_pretrained("carlofkl/DreamLite-mobile", revision="diffusers", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

source = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
    prompt="turn the cat into a corgi",
    image=source,
    height=1024,
    width=1024,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_mobile_edit.png")
```

## Notes and limitations

* Both pipelines force `batch_size = 1` internally; `num_images_per_prompt` controls how many samples
  are drawn from the same prompt rather than parallel batching.
* The prompt encoder is `Qwen3-VL`, which is a multimodal model. Loading the full pipeline therefore
  requires sufficient GPU memory for both the U-Net and the Qwen3-VL text encoder (~4 GB + ~0.7 GB
  in bf16 for the base release).
* The VAE is `AutoencoderTiny` and exposes `encoder_block_out_channels`; `vae_scale_factor` is derived
  from it at pipeline init time.

## DreamLitePipeline

[[autodoc]] DreamLitePipeline
    - all
    - __call__

## DreamLiteMobilePipeline

[[autodoc]] DreamLiteMobilePipeline
    - all
    - __call__

## DreamLitePipelineOutput

[[autodoc]] pipelines.dreamlite.pipeline_output.DreamLitePipelineOutput
