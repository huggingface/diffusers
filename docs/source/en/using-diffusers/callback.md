<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipeline callbacks

A callback is a function that modifies [`DiffusionPipeline`] behavior and it is executed at the end of a denoising step. The changes are propagated to subsequent steps in the denoising process. It is useful for adjusting pipeline attributes or tensor variables to support new features without rewriting the underlying pipeline code.

Diffusers provides several callbacks in the pipeline [overview](../api/pipelines/overview#callbacks).

To enable a callback, configure when the callback is executed after a certain number of denoising steps with one of the following arguments.

- `cutoff_step_ratio` specifies when a callback is activated as a percentage of the total denoising steps.
- `cutoff_step_index` specifies the exact step number a callback is activated.

The example below uses `cutoff_step_ratio=0.4`, which means the callback is activated once denoising reaches 40% of the total inference steps. [`~callbacks.SDXLCFGCutoffCallback`] disables classifier-free guidance (CFG) after a certain number of steps, which can help save compute without significantly affecting performance.

Define a callback with either of the `cutoff` arguments and pass it to the `callback_on_step_end` parameter in the pipeline.

```py
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers.callbacks import SDXLCFGCutoffCallback

callback = SDXLCFGCutoffCallback(cutoff_step_ratio=0.4)
# if using cutoff_step_index
# callback = SDXLCFGCutoffCallback(cutoff_step_ratio=None, cutoff_step_index=10)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

prompt = "a sports car at the road, best quality, high quality, high detail, 8k resolution"
output = pipeline(
    prompt=prompt,
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
    generator=generator,
    callback_on_step_end=callback,
)
```

If you want to add a new official callback, feel free to open a [feature request](https://github.com/huggingface/diffusers/issues/new/choose) or [submit a PR](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#how-to-open-a-pr). Otherwise, you can also create your own callback as shown below.

## Early stopping

Early stopping is useful if you aren't happy with the intermediate results during generation. This callback sets a hardcoded stop point after which the pipeline terminates by setting the `_interrupt` attribute to `True`.

```py
from diffusers import StableDiffusionXLPipeline

def interrupt_callback(pipeline, i, t, callback_kwargs):
    stop_idx = 10
    if i == stop_idx:
        pipeline._interrupt = True

    return callback_kwargs

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
)
num_inference_steps = 50

pipeline(
    "A photo of a cat",
    num_inference_steps=num_inference_steps,
    callback_on_step_end=interrupt_callback,
)
```

## Display intermediate images

Visualizing the intermediate images is useful for progress monitoring and assessing the quality of the generated content. This callback decodes the latent tensors at each step and converts them to images.

[Convert](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space) the Stable Diffusion XL latents from latents (4 channels) to RGB tensors (3 tensors).

```py
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

Use the `callback_on_step_end_tensor_inputs` parameter to specify what input type to modify, which in this case, are the latents.

```py
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)

image = pipeline(
    prompt="A croissant shaped like a cute bear.",
    negative_prompt="Deformed, ugly, bad anatomy",
    callback_on_step_end=decode_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]
```
