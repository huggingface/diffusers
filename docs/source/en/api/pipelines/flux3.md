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

# FLUX 3 Image Klein

FLUX 3 Image Klein is a 5.6B image generation model from Black Forest Labs: a single transformer ([`Flux3Transformer2DModel`]) generates images from a text prompt, alone or together with one or more reference images. Text conditioning comes from Qwen3-VL, and images use the FLUX.2 packed autoencoder ([`AutoencoderKLFlux2`]).

FLUX 3 Image Klein ships as a [modular pipeline](../../modular_diffusers/overview). Get started with a text-to-image generation:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.to("cuda")

image = pipe(
    prompt="A red fox in a snow-covered forest clearing at golden hour",
    generator=torch.Generator().manual_seed(0),
    output="images",
)[0]
image.save("fox.png")
```

`prompt` is the only required input: the pipeline carries the recommended sampling settings (50 steps, guidance scale 3.5, scheduler shift 6.93) and generates 1024 x 1024 images by default. It takes one prompt per call; use `num_images_per_prompt` for several images of the same prompt.

> [!TIP]
> There are a few things `ModularPipeline` working different from a `DiffusionPipeline`; please read the [quickstart](../../modular_diffusers/quickstart) if you're working with ModularPipeline for the first time.
> - Loading is lazy: [`~ModularPipeline.from_pretrained`] reads the repository's `modular_model_index.json` and builds the pipeline without loading any weights, and [`~ModularPipeline.load_components`] then loads them. You don't have to load all the components: [`~ModularPipeline.load_components`] takes `names` to load a subset, or `workflow` to load one workflow's components. You can also use [`~ModularPipeline.update_components`] reuse component you already have, or one that differs from the default (e.g. another guider, a quantized transformer). See [Loading components](../../modular_diffusers/modular_pipeline#loading-components) and [Updating components](../../modular_diffusers/modular_pipeline#updating-components).
> - `__call__` returns the pipeline state, which holds the intermediate values too (for example, `prompt_embeds`, `latents`, ...). Ask for what you need with `output` argument, as above: `output="images"`, `output=["images", "prompt"]`, or `output="latents"`.
> - Classifier-free guidance is supported through the `guider` component, not a call argument `guidance_scale`. See [Guidance scale](#guidance-scale).
> - There is no `callback_on_step_end`; You can insert a block in the denoising loop to support callbakcs. See [callbacks](#per-step-callbacks).
> - The pipeline is a graph of blocks that you can change (pop, insert, replace) before building it, as [Prompt upsampling](#prompt-upsampling) does to drop the upsampler step. See [Compose new workflows](../../modular_diffusers/quickstart#compose-new-workflows).
> - Offloading is handled by a [`ComponentsManager`] to be able to handle multipe workflows with shared components. See [Memory](#memory) and the [ComponentsManager guide](../../modular_diffusers/components_manager#offloading).
> - `DiffusionPipeline.from_pretrained` and the `AutoPipeline` classes don't read `modular_model_index.json`; load with [`ModularPipeline`] as above.

## Prompt upsampling

Prompting matters: FLUX 3 Image Klein takes full scene captions that describe the subject, composition, lighting, and atmosphere, not short prompts; a request sent as typed gives generic output. That is why the pipeline runs the prompt upsampler first on every request, text-to-image and editing alike. It is a separate 9B model from Black Forest Labs ([black-forest-labs/flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)) that expands `prompt` into such a caption before the text encoder, and a short prompt is all it takes.

The prompt upsampler turns a short request, together with the reference images when editing, into the full scene caption the model takes: medium and style, scene, lighting and camera, each subject with its colors, materials and position in the frame. One model covers text-to-image, editing with reference images and layout with boxes. The caption it wrote comes back as the `prompt` output: in the example above, pass `output=["images", "prompt"]` and read `out["prompt"]` to see the caption the image model received. `max_new_tokens` caps its length (3584 by default).

you can print out the `doc` to see what it takes and what it writes:

```python
print(pipe.blocks.sub_blocks["prompt_upsampler"].doc)
```

```
  Inputs:
      prompt (`str`):
          The edit instruction to expand into the full caption (one per call).
      image (`Image | list`, *optional*):
          The reference image(s) of the edit, at most 10; the same ones the pipeline receives.
      boxes (`list`, *optional*):
          Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as [y_min, x_min, y_max, x_max] in
          0-1: ...
      max_new_tokens (`int`, *optional*, defaults to 3584):
          The most tokens the upsampler may generate for the caption.

  Outputs:
      prompt (`str`):
          The full caption, replacing the request.
```

Like other blocks in the pipeline, the `prompt_upsampler` block also runs on its own. The recommended way is to pop it out of the blocks: the popped block becomes a standalone upsampler, and the remaining blocks become a pipeline that takes the expanded caption as `prompt`:

```python
blocks = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base").blocks
pup_block = blocks.sub_blocks.pop("prompt_upsampler")

pup = pup_block.init_pipeline("black-forest-labs/flux-3-image-klein-base")
pup.load_components(dtype=torch.bfloat16)
pup.to("cuda")
caption = pup(prompt="A red fox in a snow-covered forest clearing at golden hour", output="prompt")

pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.to("cuda")
image = pipe(prompt=caption, generator=torch.Generator().manual_seed(0), output="images")[0]
```

This way it can be served as a separate stage, on its own GPU or behind its own endpoint, and a UI can show the caption it wrote and let the user edit it before generating. Any other LLM can take the upsampler's place, with the same system prompts: give it `system_prompts["t2i"]` from the upsampler repository (or `"i2i"`, with the references first in the user message) as the system prompt and the request as the user message, and pass the caption it returns as `prompt` to `pipe`.

## Image-to-image

Pass one or more reference images with `image`. Every reference keeps its own resolution (center-cropped to the 16-pixel grid; references above 16 megapixels are downscaled first), and the generation size defaults to one megapixel at the first reference's aspect ratio. The prompt the image model takes is the *target* image as a complete caption of the reference with the edit folded in, not a one-line instruction; the upsampler sees the references too and writes it from your instruction: what to keep, what to change.

```python
night = pipe(
    prompt="the same scene at night under a full moon",
    image=image,
    generator=torch.Generator().manual_seed(0),
    output=["images", "prompt"],
)
```

With several references, refer to them as the first image, the second image, ... in the order they are passed to `image`; the caption names them `<ref_image_0>`, `<ref_image_1>`, ... for the image model:

```python
combined = pipe(
    prompt="the fox from the first image sitting on the wooden bench from the second image",
    image=[image, bench],
    height=768,
    width=1360,
    output="images",
)[0]
```

Pass `height` and/or `width` to override the derived size.

## Layout control

To place content explicitly, pass `boxes`: for each element an `id`, a short `desc`, and a `bbox` of `[y_min, x_min, y_max, x_max]` in 0-1 (origin top-left), so it is independent of the output size. The upsampler writes the caption around them, citing each element as `<id>`, and appends the table of boxes, yours kept exactly:

```python
boxes = [
    {"id": "balloon_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]},
    {"id": "balloon_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]},
]
out = pipe(prompt="hot-air balloons over a green valley at sunrise", boxes=boxes, output=["images", "prompt"])
```

When editing, a box names its source with `from`: `"target"` with a `tgt_bbox` adds a new element there, `"ref_image_0"` with a `src_bbox` and a `tgt_bbox` moves an element of that reference (equal boxes keep it in place):

```python
boxes = [{"id": "cactus", "desc": "a small potted cactus", "from": "target", "src_bbox": None, "tgt_bbox": [0.55, 0.62, 0.85, 0.85]}]
out = pipe(prompt="add a small potted cactus in the marked spot", image=mug, boxes=boxes, output=["images", "prompt"])

boxes = [{"id": "knight", "desc": "the knight figure", "from": "ref_image_0", "src_bbox": [0.50, 0.15, 0.85, 0.35], "tgt_bbox": [0.19, 0.06, 0.54, 0.26]}]
out = pipe(prompt="move the knight up and to the left, keep the rest unchanged", image=knight, boxes=boxes, output=["images", "prompt"])
```

The caption the image model reads has the elements tagged `<id>` in the text and the table appended as JSON, with the boxes on a 0-1000 canvas; print the `prompt` output to see it. A caption written in that form by hand goes to the pipeline without the step, as in [Prompt upsampling](#prompt-upsampling).

## Guidance scale

The pipeline applies classifier-free guidance through its `guider` component and does not take a `guidance_scale` argument at call time. The default guider is [`ClassifierFreeGuidance`] with `guidance_scale=3.5`, the recommended setting for both workflows. To change it, create an updated copy of the guider:

```python
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

See the [guiders guide](../../using-diffusers/guiders) to switch to a different guidance method.

## Step-distilled model

The step-distilled checkpoint generates in 16 steps without classifier-free guidance. Its transformer is a flow map: at every step it is told the current timestep and the one the step jumps to. It loads like the base model (the repo's `modular_model_index.json` resolves to [`Flux3DistilledModularPipeline`] and [`Flux3DistilledAutoBlocks`]); the pipeline has no `guider` and does not take `negative_prompt`:

```python
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="images")[0]
```

It serves the same two workflows as the base model. `num_inference_steps` defaults to 16; the model is trained for 4 to 16 steps.

## Reproducible generation

A `generator` fixes the noise the pipeline draws (see [Reproducibility](../../using-diffusers/reusing_seeds)). To reuse noise drawn elsewhere, pass it as `latents` with shape `(batch_size, 128, height // 16, width // 16)`. For example, the [BFL implementation](https://github.com/black-forest-labs/flux-3) draws its noise as image tokens on the CPU and casts it to bf16; this reproduces its sample for a given seed:

```python
height, width = 1024, 1024
h, w = height // 16, width // 16

noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
latents = noise.T.reshape(1, 128, h, w)

image = pipe(prompt=prompt, latents=latents, height=height, width=width, output="images")[0]
```

## Workflows

The pipeline serves two workflows, `text2image` and `image_conditioned`; the one that runs is picked by the inputs (`image` selects image-to-image). [`~ModularPipeline.from_pretrained`] can also build a pipeline for one workflow with the `workflow` argument, which then only expects and loads the components that workflow needs:

```python
from diffusers.modular_pipelines import Flux3AutoBlocks

print(Flux3AutoBlocks().available_workflows)
# ["text2image", "image_conditioned"]

t2i = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base", workflow="text2image")
print(t2i.blocks.doc)
```

## Per-step callbacks

Modular pipelines don't carry the `callback_on_step_end` and `interrupt` mechanism of `DiffusionPipeline`. It will come with first-class streaming support instead: [huggingface/diffusers#14159](https://github.com/huggingface/diffusers/pull/14159). For now, to reuse your existing callback functions, you can insert a block into the denoising loop: it receives the loop's `BlockState` with the step index and timestep, and can read or edit `latents`. 

> [!TIP]
> Edit the blocks *before* [`~ModularPipelineBlocks.init_pipeline`]: `pipe.blocks` returns a copy, so changing it after the pipeline exists has no effect.

The block below takes the callback as a pipeline input and calls it with the same signature as the [callbacks guide](../../using-diffusers/callback): the pipeline, the step index, the timestep and a dict of tensors, whose returned values are written back.

```python
from typing import Callable

from diffusers.modular_pipelines import Flux3AutoBlocks, InputParam, ModularPipelineBlocks


class CallbackOnStepEnd(ModularPipelineBlocks):
    @property
    def inputs(self):
        return [
            InputParam(
                "callback_on_step_end",
                type_hint=Callable,
                description="Called after each scheduler step as `callback(pipe, i, t, {'latents': ...})`; the dict it returns is written back.",
            ),
            InputParam("latents", required=True),
        ]

    def __call__(self, components, block_state, i, t):
        if block_state.callback_on_step_end is not None:
            outputs = block_state.callback_on_step_end(components, i, t, {"latents": block_state.latents})
            block_state.latents = outputs.get("latents", block_state.latents)
        return components, block_state


blocks = Flux3AutoBlocks()
for workflow in ("text2image", "image_conditioned"):
    loop = blocks.sub_blocks["denoise"].sub_blocks[workflow].sub_blocks["denoise"]
    loop.sub_blocks.insert("callback", CallbackOnStepEnd(), 2)  # after the scheduler step
pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein-base")
```

`callback_on_step_end` is now an input of the pipeline, and a callback written for a `DiffusionPipeline` works as is:

```python
def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    ...  # preview, log or edit the latents
    return callback_kwargs


image = pipe(prompt=caption, callback_on_step_end=decode_tensors, output="images")[0]
```

[Early stopping](../../using-diffusers/callback#early-stopping) works differently: a `DiffusionPipeline` sets `pipeline._interrupt = True` and its loop skips the remaining steps, but a block cannot skip the loop it runs in. Raise from the callback instead: the exception propagates out of `pipe(...)` (no image is decoded), and the pipeline stays usable afterwards.

## Memory

Loading everything in bf16 takes about 39 GB of weights: the transformer is 5.5B parameters, the Qwen3-VL-4B text encoder 4B and the [prompt upsampler](#prompt-upsampling) 9B (about 19 GB on its own), with the image VAE in fp32. A pipeline built without the upsampler step (see there) loads about 20 GB. For GPUs with less memory, enable component offloading; the manager moves components between the CPU and the GPU as each step needs them, the upsampler included:

```python
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base", components_manager=manager)
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
manager.enable_auto_cpu_offload(device="cuda")
```

## Single-file loading

[`Flux3Transformer2DModel`] loads the transformer straight from a Black Forest Labs model package with [`~loaders.FromOriginalModelMixin.from_single_file`]. The package `config.json` is not read; pass the diffusers config with `config` and `subfolder`.

```python
import torch
from diffusers import Flux3Transformer2DModel, ModularPipeline

transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/model.safetensors",
    config="black-forest-labs/flux-3-image-klein-base",
    subfolder="transformer",
    dtype=torch.bfloat16,
)

pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.update_components(transformer=transformer)
pipe.to("cuda")
```

Load the other components by name first: [`~ModularPipeline.load_components`] without `names` would also load the transformer from the pipeline repo.

Loading the FP8r and NVFP4 single-file checkpoints are also supported. Both load through the same call with a [`BFLQuantizationConfig`]. See the [BFL quantization guide](../../quantization/bfl) for the kernel requirements of each scheme.

```python
from diffusers import BFLQuantizationConfig

transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/variants/nvfp4/model.safetensors",
    config="black-forest-labs/flux-3-image-klein-base",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
```

When loading the step-distilled checkpoints, we need their
configs (`flow_map=True`):

```python
transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/variants/nvfp4/model.safetensors",
    config="black-forest-labs/flux-3-image-klein",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
```

## Flux3ModularPipeline

[[autodoc]] Flux3ModularPipeline

## Flux3AutoBlocks

[[autodoc]] Flux3AutoBlocks

## Flux3DistilledAutoBlocks

[[autodoc]] Flux3DistilledAutoBlocks

## Flux3DistilledModularPipeline

[[autodoc]] Flux3DistilledModularPipeline
