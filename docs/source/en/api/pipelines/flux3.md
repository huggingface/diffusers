<!-- Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# FLUX.3

FLUX.3 is an image generation model from Black Forest Labs: a single transformer ([`Flux3Transformer2DModel`]) generates images from a text prompt, alone or together with one or more reference images. Text conditioning comes from Qwen3-VL, and images use the FLUX.2 packed autoencoder ([`AutoencoderKLFlux2`]).

FLUX.3 ships as a modular pipeline. Prompting matters: the model responds best to full scene captions that describe the subject, composition, lighting, and atmosphere — see the [FLUX.3 prompting guide](https://docs.bfl.ai) for details. Get started with a text-to-image generation:

<!-- TODO: replace the prompting guide link with the final FLUX.3 guide URL and the repo id with the release repo -->
```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("black-forest-labs/FLUX.3-image-small")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    prompt=(
        "A wide-angle wildlife photograph of a red fox standing alert in a snow-covered forest "
        "clearing at golden hour. The fox is centered in the frame, its thick orange winter coat "
        "contrasting with the fresh white snow, breath visible in the cold air. Snow-laden pine "
        "trees fill the softly blurred background, and low warm sunlight rakes across the scene "
        "from the left, casting long blue shadows."
    ),
    generator=torch.Generator().manual_seed(0),
    output="images",
)[0]
image.save("fox.png")
```

`prompt` is the only required input: the pipeline carries the recommended sampling settings (50 steps, guidance scale 3.5, scheduler shift 6.93) and generates 1024 x 1024 images by default. It takes one prompt per call; use `num_images_per_prompt` for several images of the same prompt.

## Image-to-image

Pass one or more reference images with `image`. Every reference keeps its own resolution (center-cropped to the 16-pixel grid; references above 16 megapixels are downscaled first), and the generation size defaults to one megapixel at the first reference's aspect ratio. Describe the *target* image as a complete caption of the reference with your edit folded in, rather than a one-line instruction:

```python
night = pipe(
    prompt=(
        "The same red fox in the same snow-covered forest clearing, now at night under a full moon: "
        "cool blue moonlight, deep shadows, stars visible through the pine branches, the fox's eyes "
        "catching the light."
    ),
    image=image,
    generator=torch.Generator().manual_seed(0),
    output="images",
)[0]
```

With several references, the prompt can refer to them in order:

```python
combined = pipe(
    prompt="The fox from the first image sitting on the wooden bench from the second image, in the same snowy clearing.",
    image=[image, bench],
    height=768,
    width=1360,
    output="images",
)[0]
```

Pass `height` and/or `width` to override the derived size.

## Guidance scale

The pipeline applies classifier-free guidance through its `guider` component and does not take a `guidance_scale` argument at call time. The default guider is [`ClassifierFreeGuidance`] with `guidance_scale=3.5`, the recommended setting for both workflows. To change it, create an updated copy of the guider:

```python
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

See the [guiders guide](../../using-diffusers/guiders) to switch to a different guidance method.

## Workflows

The pipeline serves two workflows, `text2image` and `image_conditioned`; the one that runs is picked by the inputs (`image` selects image-to-image). [`~ModularPipeline.from_pretrained`] can also build a pipeline for one workflow with the `workflow` argument, which then only expects and loads the components that workflow needs:

```python
from diffusers.modular_pipelines import Flux3AutoBlocks

print(Flux3AutoBlocks().available_workflows)
# ["text2image", "image_conditioned"]

t2i = ModularPipeline.from_pretrained("black-forest-labs/FLUX.3-image-small", workflow="text2image")
print(t2i.blocks.doc)
```

## Memory

Loading everything in bf16 takes about 20 GB of weights (the transformer is 5.5B parameters, the Qwen3-VL-4B text encoder 4B), with the image VAE in fp32. For GPUs with less memory, enable component offloading:

```python
pipe = ModularPipeline.from_pretrained("black-forest-labs/FLUX.3-image-small")
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_auto_cpu_offload(device="cuda")
```

## Single-file loading

[`Flux3Transformer2DModel`] loads the transformer straight from a Black Forest Labs model package with [`~loaders.FromOriginalModelMixin.from_single_file`]. The package `config.json` is not read; pass the diffusers config with `config` and `subfolder`.

```python
import torch
from diffusers import Flux3Transformer2DModel, ModularPipeline

transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/FLUX.3-image-small/blob/main/model.safetensors",
    config="black-forest-labs/FLUX.3-image-small",
    subfolder="transformer",
    dtype=torch.bfloat16,
)

pipe = ModularPipeline.from_pretrained("black-forest-labs/FLUX.3-image-small")
pipe.load_components(names=["text_encoder", "tokenizer", "vae"], dtype=torch.bfloat16)
pipe.update_components(transformer=transformer)
pipe.to("cuda")
```

Load the other components by name first: [`~ModularPipeline.load_components`] without `names` would also load the transformer from the pipeline repo.

## Flux3ModularPipeline

[[autodoc]] Flux3ModularPipeline

## Flux3AutoBlocks

[[autodoc]] Flux3AutoBlocks
