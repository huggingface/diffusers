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

FLUX.3 ships as a modular pipeline. It takes full scene captions, not short prompts: a request sent as typed gives generic output. Run the [prompt upsampler](#prompt-upsampling) on every request, text-to-image and editing alike, or write the caption with any LLM using the system prompts in [`system_prompts.json`](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/main/system_prompts.json). The pipeline does not run an upsampler on its own; the example below passes a full caption:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
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

## Prompt upsampling

The prompt upsampler, [black-forest-labs/flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), turns a short request, together with the reference images when editing, into the full scene caption the model takes. It is a separate model; run it with Transformers and pass its output as `prompt`:

```python
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


model_id = "black-forest-labs/flux-3-image-klein-pup"
processor = AutoProcessor.from_pretrained(model_id)
upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
system_prompts = json.load(open(hf_hub_download(model_id, "system_prompts.json")))


def upsample(prompt, images=()):
    if images:
        text = (
            "Reference images follow in order (the first image, the second image, ...).\n"
            f"User request: {prompt}\nOutput the caption."
        )
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text}]
        messages = [{"role": "system", "content": system_prompts["i2i"]}, {"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": system_prompts["t2i"]}, {"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(upsampler.device)
    with torch.no_grad():
        output = upsampler.generate(**inputs, max_new_tokens=3584, do_sample=False)
    return processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


image = pipe(prompt=upsample("An astronaut riding a horse"), output="images")[0]
```

For an edit, pass the same reference images to both: `pipe(prompt=upsample(instruction, [image]), image=image, ...)`. The upsampler also writes the [layout](#layout-control) captions from a list of boxes; see its model card.

Any other LLM can take its place: give it `system_prompts["t2i"]` (or `"i2i"` with the references) as the system prompt and the request as the user message, and pass the caption it returns as `prompt`.

## Image-to-image

Pass one or more reference images with `image`. Every reference keeps its own resolution (center-cropped to the 16-pixel grid; references above 16 megapixels are downscaled first), and the generation size defaults to one megapixel at the first reference's aspect ratio. The prompt is the *target* image as a complete caption of the reference with the edit folded in, not a one-line instruction; the [prompt upsampler](#prompt-upsampling) writes it from the instruction and the references:

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

With several references, refer to them in the prompt as `<ref_image_0>`, `<ref_image_1>`, ... in the order they are passed to `image`:

```python
combined = pipe(
    prompt="The fox from <ref_image_0> sitting on the wooden bench from <ref_image_1>, in the same snowy clearing.",
    image=[image, bench],
    height=768,
    width=1360,
    output="images",
)[0]
```

Pass `height` and/or `width` to override the derived size.

## Layout control

To place content explicitly, tag regions in the caption with `<id>` and append a JSON list of regions to the prompt. Each region has an `id`, a `bbox` of `[y0, x0, y1, x1]` (top-left and bottom-right corners, origin top-left) normalized to 0-1000 on both axes so it is independent of the output size, and a `desc`:

```python
prompt = (
    "Minimalist graphic illustration featuring a black silhouette of a person <silhouette_1> centered against a "
    "solid, vibrant chartreuse background <background_1>. "
    '[{"id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": "A flat field of neon yellow-green."}, '
    '{"id": "silhouette_1", "bbox": [150, 150, 850, 850], "desc": "A black silhouette of a person in motion."}]'
)
```

When editing, each region names its source reference with `from` (for example `"ref_image_0"`), where it is in that image with `src_bbox`, and where it should end up with `tgt_bbox`. Equal boxes keep a region in place:

```python
prompt = (
    "In <ref_image_0>, move the knight figure <knight_1> upwards and to the left. Keep the rest of the image "
    "unchanged, preserving the backdrop <backdrop_1>. "
    '[{"id": "knight_1", "from": "ref_image_0", "src_bbox": [500, 150, 850, 350], "tgt_bbox": [194, 55, 544, 255], '
    '"desc": "A miniature amigurumi knight crocheted from grey yarn."}, '
    '{"id": "backdrop_1", "from": "ref_image_0", "src_bbox": [0, 0, 1000, 1000], "tgt_bbox": [0, 0, 1000, 1000], '
    '"desc": "A soft, out-of-focus backdrop of knitted yarn."}]'
)
image = pipe(prompt=prompt, image=knight, output="images")[0]
```

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

## Memory

Loading everything in bf16 takes about 20 GB of weights (the transformer is 5.5B parameters, the Qwen3-VL-4B text encoder 4B), with the image VAE in fp32. For GPUs with less memory, enable component offloading:

```python
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.enable_auto_cpu_offload(device="cuda")
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
