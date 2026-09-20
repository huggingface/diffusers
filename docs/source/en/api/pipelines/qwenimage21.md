<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Qwen-Image 2.1

Qwen-Image 2.1 encodes the prompt and any condition images together with a Qwen3-VL model, then denoises the target
image with a single-stream block-causal transformer. See
[`QwenImage21Transformer2DModel`](../models/qwenimage21_transformer2d) for block-causal attention, the attention
processors, and `causal_condition`.

The defaults are the values Qwen recommends: 40 steps and no guidance. Pass a `negative_prompt` together with
`true_cfg_scale > 1` to turn classifier-free guidance on, which doubles the work per step.

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16).to("cuda")

# Text-to-image
image = pipe("A capybara wearing a wizard hat, oil painting").images[0]
image.save("t2i.png")

# Image-conditioned editing
edited = pipe("Move it to a snowy mountain top", image=image).images[0]
edited.save("edit.png")
```

## Multiple condition images

Pass a list to `image` and every entry becomes its own block in the joint sequence: the Qwen3-VL encoder sees them as
vision context and the VAE contributes their latent tokens. Block-causal attention keeps each block internally
bidirectional while letting later blocks and the target image attend to the earlier ones, so the order you pass them
in is the order the model reads them.

```python
edited = pipe("Put the flowers from the first image into the second scene", image=[flowers, scene]).images[0]
```

## Faster attention with flex_attention

The default `QwenImage21AttnProcessor` runs the block-causal prefill as one attention call per prefix segment. It
needs no compilation and works on any PyTorch build. `QwenImage21FlexAttnProcessor` expresses the same mask as a
single `flex_attention` call, which is faster once the model is **_compiled_**.

> [!TIP]
> Compile the model when you switch to the flex processor. An uncompiled `flex_attention` materializes the full
> attention score matrix in fp32, which is much slower and runs out of memory at high resolution.

```python
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
pipe.transformer.compile()
```

## Modular generation and inpainting

[`QwenImage21ModularPipeline`] selects text-to-image, image-conditioned generation, or inpainting from the inputs.
It reuses the QwenImage21 transformer, RGBA VAE, and Qwen3-VL processor from the standard checkpoint.

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipe = ModularPipeline.from_pretrained("Qwen/Qwen-Image-2.1")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

images = pipe(prompt="A capybara wearing a wizard hat", output="images")

source = load_image("source.png")
mask = load_image("mask.png").convert("L")
images = pipe(
    prompt="Give the capybara a red scarf",
    image=source,
    mask_image=mask,
    strength=1.0,
    generator=torch.Generator("cpu").manual_seed(0),
    output="images",
)
images[0].save("inpaint.png")
```

White mask pixels are repainted; black pixels restore the source latents after each denoising step. Preservation
is in latent space, so decoding can change unmasked pixels through VAE reconstruction. `strength` must be in
`(0, 1]` and leave at least one denoising step. Lower values start from a less noisy source image.

For multiple-reference inpainting, keep one source in `image` and pass the extra images as `reference_images`:

```python
images = pipe(
    prompt="Use the scarf from image 2 and the fabric pattern from image 3 in the masked area of image 1",
    image=source,
    mask_image=mask,
    reference_images=[load_image("scarf.png"), load_image("fabric.png")],
    output="images",
)
```

The source is the first condition image, followed by references in the supplied order. All condition images feed
both Qwen3-VL and the VAE. References can have different aspect ratios; only the source has a repaint mask. The
source, mask, and references are shared across a batch of prompts. Nested per-prompt reference lists are unsupported.
Without a mask, `image` can be a single image or a flat list for image-conditioned generation.

`height` and `width` must be multiples of 32. `output_resolution` defaults to 1024 and controls the area used to
resize condition images, independently of explicit output dimensions. Inpainting derives omitted dimensions from
the source; ordinary image-conditioned generation derives them from the last condition image.

The default guider disables classifier-free guidance. Configure guidance through the component rather than a
pipeline argument:

```python
from diffusers import ClassifierFreeGuidance

pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=3.0))
images = pipe(prompt="A capybara", negative_prompt="blurry", output="images")
```

`use_kv_cache=True` caches step-independent condition tokens when the transformer has `causal_condition` enabled.
Each generation and guidance branch has its own cache. As with the standard pipeline, reduced-precision cached
and uncached outputs can differ because the attention layouts differ.

For standalone encoding or denoising, obtain a workflow with
`pipe.blocks.get_workflow("text2image")`, `"image_conditioned"`, or `"inpainting"`, and initialize a pipeline from
its individual blocks using `init_pipeline()`. Encoders return unexpanded embeddings; the denoise input block
applies `num_images_per_prompt`, so encoded features can be reused for different output counts.

## QwenImage21ModularPipeline

[[autodoc]] QwenImage21ModularPipeline

## QwenImage21Pipeline

[[autodoc]] QwenImage21Pipeline
    - all
    - __call__

## QwenImagePipelineOutput

[[autodoc]] pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput
