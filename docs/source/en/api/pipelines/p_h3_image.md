<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# P-H3-Image

P-H3-Image is an image model by Pruna AI based on [MiniMax-H3](./minimax_h3), for text-to-image generation and instruction-based image editing. You can find the checkpoint at [PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image).

```py
import torch
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")  # or "mps", "xpu"

pipe = ModularPipeline.from_pretrained("PrunaAI/P-H3-Image", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)

prompt = (
    "A German Expressionist oil painting of a Berlin street at night in the manner of Ernst Ludwig Kirchner, "
    "with gaunt figures hurrying past a glowing gas lamp."
)
image = pipe(prompt=prompt, height=3360, width=2496, generator=torch.Generator().manual_seed(3), output="images")[0]
edited = pipe(prompt="let it snow on the street", image=image, output="images")[0]
```

Every request goes through the [prompt upsampler](#prompt-upsampling) first. With every component on the accelerator, the pipeline needs about 101 GB in bfloat16 at 8 megapixels, so the examples use auto offloading: the [`ComponentsManager`] moves each component onto the accelerator when a block needs it, and the pipeline runs on an 80 GB GPU.

## Components

| Component | Class |
|---|---|
| `transformer` | [`MiniMaxH3Transformer3DModel`] |
| `text_encoder` | `Qwen3VLForConditionalGeneration` (Qwen3-VL-4B) |
| `text_connector` | [`PH3ImageTextConnector`] |
| `vae` | [`AutoencoderKLPH3Image`] |
| `tokenizer`, `processor` | The text encoder's tokenizer, and a `Qwen3VLProcessor` with MiniMax-H3's image processor |
| `scheduler`, `audio_scheduler` | [`MiniMaxH3Scheduler`] |
| `prompt_upsampler`, `prompt_upsampler_processor` | `Qwen3_5ForConditionalGeneration` (based on Qwen3.5-9B) and its processor |

## Workflows

The workflow is selected on the `image` input:

- **`text2image`** generates from `prompt`. `height` and `width` default to 2912 and are rounded to multiples of 32. The model works best at 8 megapixels, for example 2912x2912, 3872x2176 or 2176x3872.
- **`image_conditioned`** edits `image` by the instruction in `prompt`, and works best at 4 or 8 megapixels. The image is encoded at about `vision_pixels` (one megapixel) for the text encoder and at its own size for the VAE, scaled up to `reference_min_pixels` (one megapixel) or down to `reference_max_pixels` (eight megapixels) when outside that range. Without `height` and `width`, the edit is generated at the size of the first image, which is scaled up to `canvas_min_pixels` (four megapixels) if smaller. Passed `height` and `width`, rounded to multiples of 32, change the canvas but not the images. `image` can also be a list of up to 8 images: the first is edited and the others are references, as `<Picture 2>`, `<Picture 3>` and so on.

## Prompt upsampling

P-H3-Image is meant to render full prompts rather than short requests, so the pipeline runs the prompt upsampler first on every request. It thinks about the request and then writes the prompt: a detailed text-to-image prompt for the aspect ratio of `height` and `width`, or, when editing, a brief of what to keep and what to change in the images. The prompt it wrote comes back as the `prompt` output.

```py
out = pipe(prompt=prompt, height=3360, width=2496, output=["images", "prompt"])
print(out["prompt"])
```

`max_new_tokens` caps the reasoning and the prompt together, 1024 tokens for text-to-image and 1536 for edits by default. If the upsampler does not return a usable prompt, the pipeline raises an error rather than rendering it.

To render `prompt` as given, pop the step from the blocks. The popped block also runs on its own, for example to show the prompt to a user before rendering it:

```py
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

blocks = ModularPipeline.from_pretrained("PrunaAI/P-H3-Image").blocks
upsampler_block = blocks.sub_blocks.pop("prompt_upsampler")

upsampler = upsampler_block.init_pipeline("PrunaAI/P-H3-Image", components_manager=manager)
upsampler.load_components(dtype=torch.bfloat16)
upsampled = upsampler(prompt=prompt, height=3360, width=2496, output="prompt")

pipe = blocks.init_pipeline("PrunaAI/P-H3-Image", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
image = pipe(prompt=upsampled, height=3360, width=2496, output="images")[0]
```

Without the step, an edit prompt in either form the upsampler writes, the six-section brief or a paragraph that starts with `Edit <Picture 1>.`, is encoded as written; any other instruction gets that prefix.

## Sampling

- The 8 steps run over the checkpoint's `sample_sigmas`. Pass `sigmas` to use another grid.
- The model is guidance-distilled, so there is no guider, `negative_prompt` or `guidance_scale`.

## PH3ImageModularPipeline

[[autodoc]] PH3ImageModularPipeline

## PH3ImageBlocks

[[autodoc]] PH3ImageBlocks

## PH3ImageTextConnector

[[autodoc]] PH3ImageTextConnector
