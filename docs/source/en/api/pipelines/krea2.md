<!--Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Krea 2

Krea 2 (K2) is a flow-matching text-to-image model built around a single-stream MMDiT with grouped-query attention. A
Qwen3-VL text encoder provides the conditioning: instead of the last hidden state, hidden states from twelve decoder
layers are tapped per token and fused inside the transformer by a small text-fusion stage. Images are decoded with the
Qwen-Image VAE.

Two checkpoints are released, sharing the same architecture but with different recommended sampler settings:

- **Base (midtrain)** — use the full sampler with classifier-free guidance: `num_inference_steps=28`,
  `guidance_scale=4.5`.
- **TDM (distilled)** — distilled for few-step sampling, run with `num_inference_steps=8` and guidance disabled
  (`guidance_scale=0.0`).

`guidance_scale` follows the Krea 2 convention: the velocity is computed as `cond + guidance_scale * (cond - uncond)`
and guidance is enabled whenever `guidance_scale > 0` (this equals the usual CFG formulation with scale
`1 + guidance_scale`).

## Text-to-image

```python
import torch
from diffusers import Krea2Pipeline

# Load from a local directory produced by the Krea 2 conversion (no hub repo yet).
pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Raw", dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "a fox in the snow"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=4.5,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2.png")
```

We additionally provide an example for using Krea2 Turbo :

```python
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    "a fox in the snow",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2_turbo.png")
```


## Krea2Pipeline

[[autodoc]] Krea2Pipeline
  - all
  - __call__

## Krea2PipelineOutput

[[autodoc]] pipelines.krea2.pipeline_output.Krea2PipelineOutput

## Modular

Krea 2 is also available as a [modular pipeline](../../modular_diffusers/overview). Classifier-free guidance is
configured through the `guider` component rather than a `guidance_scale` call argument. Krea 2 uses cond-anchored CFG,
which is [`ClassifierFreeGuidance`] with `use_original_formulation=True`.

```python
import torch
from diffusers import ClassifierFreeGuidance, ModularPipeline

pipe = ModularPipeline.from_pretrained("krea/Krea-2-Raw")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")


image = pipe(
    prompt="a fox in the snow",
    height=1024,
    width=1024,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2.png")
```

The same modular pipeline automatically selects image-to-image generation when `image` is provided. `strength`
controls how strongly the result can depart from the source image.

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipe = ModularPipeline.from_pretrained("krea/Krea-2-Raw")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
image = pipe(
    prompt="a cat wearing a knitted wizard hat",
    image=init_image,
    height=init_image.height,
    width=init_image.width,
    strength=0.8,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2_img2img.png")
```

Provide both `image` and `mask_image` to select inpainting. White mask pixels are regenerated and black mask pixels
are preserved.

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipe = ModularPipeline.from_pretrained("krea/Krea-2-Raw")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")
image = pipe(
    prompt="a small red fox sitting on a park bench",
    image=init_image,
    mask_image=mask_image,
    height=init_image.height,
    width=init_image.width,
    strength=0.9,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2_inpaint.png")
```

### Reference-conditioned generation

Pass `reference_image` to condition generation on clean reference-image tokens and an image-grounded Qwen3-VL prompt
encoding. Unlike conventional image-to-image generation, the target starts from pure noise, so this workflow does not
use `strength`. It is intended for LoRAs trained with the same reference-conditioning layout and is not tied to one
specific identity or editing adapter.

The following example uses the community [Krea 2 Identity Edit](https://huggingface.co/conradlocke/krea2-identity-edit)
LoRA:

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

pipe = ModularPipeline.from_pretrained("krea/Krea-2-Turbo")
pipe.load_components(dtype=torch.bfloat16)
pipe.load_lora_weights(
    "conradlocke/krea2-identity-edit",
    weight_name="krea2_identity_edit_v1_2_r64.safetensors",
    adapter_name="krea2_edit",
)
pipe.to("cuda")

scene_image = load_image(
    "https://raw.githubusercontent.com/lucasruan1618/Image_storage/main/Input/cute_dog.png"
)
subject_image = load_image(
    "https://raw.githubusercontent.com/lucasruan1618/Image_storage/main/Input/cute_cat.png"
)
image = pipe(
    prompt="place the wizard cat from the second image sitting on the bench beside the dog from the first image",
    reference_image=scene_image,
    reference_image_2=subject_image,
    height=1024,
    width=1024,
    reference_image_encoder_resolution=768,
    reference_attention_scale=[1.0, 4.0],
    num_inference_steps=10,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2_reference.png")
```

For two-reference generation, `reference_image` is the scene and `reference_image_2` is the subject, matching the
adapter's training order. `reference_image_encoder_resolution` controls the maximum reference-image side length passed
to Qwen3-VL. `reference_attention_scale` accepts either one value for all references or one value per reference; the
example leaves scene attention unchanged and boosts subject fidelity. The adapter's recommended LoRA scale is `1.0`.
References are resized to the requested output dimensions before VAE encoding, so use similar aspect ratios to avoid
distortion.

We additionally provide an example for using Krea2 Turbo. The distilled checkpoint maps to its own set of blocks
([`Krea2TurboAutoBlocks`]): it runs guidance-free (no `guider`), takes no negative prompt, and samples in a few steps.
`ModularPipeline.from_pretrained` picks the turbo blocks automatically from the checkpoint's `is_distilled` config, so
no guidance configuration is needed:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("krea/Krea-2-Turbo")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    prompt="a fox in the snow",
    height=1024,
    width=1024,
    num_inference_steps=8,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("krea2_turbo.png")
```

## Krea2ModularPipeline

[[autodoc]] Krea2ModularPipeline

## Krea2AutoBlocks

[[autodoc]] Krea2AutoBlocks

## Krea2TurboModularPipeline

[[autodoc]] Krea2TurboModularPipeline

## Krea2TurboAutoBlocks

[[autodoc]] Krea2TurboAutoBlocks
