<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipeline callbacks

A callback runs at the end of a denoising step and can change pipeline state or tensors for later steps. Use it to adjust attributes or tensor variables for new behavior without rewriting the pipeline.

These callbacks work with classic [`DiffusionPipeline`] loops out of the box. They also work with [Modular Diffusers](../modular_diffusers/overview) pipelines once a block that runs them is inserted into the denoising loop; see [Modular pipelines](#modular-pipelines).

Diffusers provides several callbacks in the pipeline [overview](../api/pipelines/overview#diffusers.callbacks.PipelineCallback).

To enable a callback, configure when the callback is executed after a certain number of denoising steps with one of the following arguments.

- `cutoff_step_ratio` specifies when a callback is activated as a percentage of the total denoising steps. Use when the cutoff should scale with `num_inference_steps` (for example, drop CFG after 40% of run).
- `cutoff_step_index` specifies the exact step number a callback is activated. Use when you care about an absolute step (for example, step `10` on a fixed 25-step schedule).

The example below uses `cutoff_step_ratio=0.4`, which means the callback is activated once denoising reaches 40% of the total inference steps. [`~callbacks.SDXLCFGCutoffCallback`] disables classifier-free guidance (CFG) after a certain number of steps, which can help save compute without significantly affecting performance.

Define a callback with one of the `cutoff` arguments and pass it to the `callback_on_step_end` parameter in the pipeline.

```py
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers.callbacks import SDXLCFGCutoffCallback

callback = SDXLCFGCutoffCallback(cutoff_step_ratio=0.4)
# if using cutoff_step_index
# callback = SDXLCFGCutoffCallback(cutoff_step_ratio=None, cutoff_step_index=10)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

prompt = "a sports car on the road, best quality, high quality, high detail, 8k resolution"
output = pipeline(
    prompt=prompt,
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
    callback_on_step_end=callback,
)
```

Official callbacks set their own tensor inputs. For a custom function, pass `callback_on_step_end_tensor_inputs` as in [Display intermediate images](#display-intermediate-images).

If you want to add a new official callback, feel free to open a [feature request](https://github.com/huggingface/diffusers/issues/new/choose) or [submit a PR](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#how-to-open-a-pr). Otherwise, you can also create your own callback as shown below.

## Early stopping

Early stopping is useful if you aren't happy with the intermediate results during generation. This callback sets a hardcoded stop point by setting the `_interrupt` attribute to `True`, which makes the denoising loop skip the remaining steps.

```py
import torch
from diffusers import DiffusionPipeline

def interrupt_callback(pipeline, i, t, callback_kwargs):
    stop_idx = 10
    if i == stop_idx:
        pipeline._interrupt = True
    return callback_kwargs

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    dtype=torch.bfloat16,
    device_map="cuda",  # or "mps", "xpu", "cpu"
)
pipeline(
    prompt="A photo of a cat",
    num_inference_steps=50,
    callback_on_step_end=interrupt_callback,
)
```

## Display intermediate images

Visualizing intermediate images is useful for progress monitoring. The preview below is SDXL-only. It maps SDXL latents to RGB with a linear transform for a quick look during denoising. Those weights do not transfer to other models. For Qwen-Image and similar checkpoints, decode with the model VAE instead of this helper.

[Convert](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space) Stable Diffusion XL latents (4 channels) to RGB tensors (3 channels).

```py
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)
```

Extract the latents and convert the first image in the batch to RGB. Save the image as a PNG file with the step number.

```py
def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    image = latents_to_rgb(latents[0])
    image.save(f"{step}.png")

    return callback_kwargs
```

Use `callback_on_step_end_tensor_inputs` to choose which tensors the callback receives, which in this case, are the latents.

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
)

image = pipeline(
    prompt="A croissant shaped like a cute bear.",
    negative_prompt="Deformed, ugly, bad anatomy",
    callback_on_step_end=decode_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]
```

## Modular pipelines

Modular pipelines don't carry the `callback_on_step_end` and `interrupt` mechanism. It will come with first-class streaming support instead: [huggingface/diffusers#14159](https://github.com/huggingface/diffusers/pull/14159), not merged yet, adds `pipe.stream()`, which yields the pipeline state at every denoising step and ends the run when you stop iterating. For now, to reuse your existing callback functions, insert a block into the denoising loop: it receives the loop's `BlockState` with the step index and timestep, and can read or edit `latents`.

> [!TIP]
> Edit the blocks *before* [`~modular_pipelines.ModularPipelineBlocks.init_pipeline`]: `pipe.blocks` returns a copy, so changing it after the pipeline exists has no effect.

The block below takes the callback as a pipeline input and calls it with the same signature as above: the pipeline, the step index, the timestep and a dict of tensors, whose returned values are written back. `walk_denoise_loops` finds every denoising loop in the blocks, whichever workflow it belongs to, so the same snippet works for any modular pipeline.

```py
from typing import Callable

import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines import InputParam, LoopSequentialPipelineBlocks, ModularPipelineBlocks


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


def walk_denoise_loops(block):
    """Every denoising loop in the blocks, whichever workflow it belongs to."""
    if isinstance(block, LoopSequentialPipelineBlocks) and "denoiser" in block.sub_blocks:
        yield block
    for sub in getattr(block, "sub_blocks", {}).values():
        yield from walk_denoise_loops(sub)


repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
blocks = ModularPipeline.from_pretrained(repo_id).blocks
for loop in walk_denoise_loops(blocks):
    loop.sub_blocks["callback"] = CallbackOnStepEnd()  # appended after the last step of the loop
pipe = blocks.init_pipeline(repo_id)
pipe.load_components(dtype=torch.float16)
pipe.to("cuda")
```

`callback_on_step_end` is now an input of the pipeline, and a callback written for a `DiffusionPipeline`, such as `decode_tensors` above, works as is:

```py
image = pipe(prompt="A croissant shaped like a cute bear.", callback_on_step_end=decode_tensors, output="images")[0]
```

[Early stopping](#early-stopping) works differently: a block cannot skip the loop it runs in, so setting `_interrupt` has no effect. Raise from the callback instead: the exception propagates out of `pipe(...)` (no image is decoded), and the pipeline stays usable afterwards.

This works for every modular pipeline except the inner loop of nested loops (autoregressive video models such as Helios and Wan-Animate-2 denoise each chunk inside a block of the outer loop). There, insert the block into the outer loop instead: it runs once per chunk and receives the chunk index `k` in place of `i` and `t`.
