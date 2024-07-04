<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipeline callbacks

The denoising loop of a pipeline can be modified with custom defined functions using the `callback_on_step_end` parameter. The callback function is executed at the end of each step, and modifies the pipeline attributes and variables for the next step. This is really useful for *dynamically* adjusting certain pipeline attributes or modifying tensor variables. This versatility allows for interesting use-cases such as changing the prompt embeddings at each timestep, assigning different weights to the prompt embeddings, and editing the guidance scale. With callbacks, you can implement new features without modifying the underlying code!

> [!TIP]
> 🤗 Diffusers currently only supports `callback_on_step_end`, but feel free to open a [feature request](https://github.com/huggingface/diffusers/issues/new/choose) if you have a cool use-case and require a callback function with a different execution point!

This guide will demonstrate how callbacks work by a few features you can implement with them.

## Official callbacks

We provide a list of callbacks you can plug into an existing pipeline and modify the denoising loop. This is the current list of official callbacks:

- `SDCFGCutoffCallback`: Disables the CFG after a certain number of steps for all SD 1.5 pipelines, including text-to-image, image-to-image, inpaint, and controlnet.
- `SDXLCFGCutoffCallback`: Disables the CFG after a certain number of steps for all SDXL pipelines, including text-to-image, image-to-image, inpaint, and controlnet.
- `IPAdapterScaleCutoffCallback`: Disables the IP Adapter after a certain number of steps for all pipelines supporting IP-Adapter.

> [!TIP]
> If you want to add a new official callback, feel free to open a [feature request](https://github.com/huggingface/diffusers/issues/new/choose) or [submit a PR](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#how-to-open-a-pr).

To set up a callback, you need to specify the number of denoising steps after which the callback comes into effect. You can do so by using either one of these two arguments

- `cutoff_step_ratio`: Float number with the ratio of the steps.
- `cutoff_step_index`: Integer number with the exact number of the step.

```python
import torch

from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers.callbacks import SDXLCFGCutoffCallback


callback = SDXLCFGCutoffCallback(cutoff_step_ratio=0.4)
# can also be used with cutoff_step_index
# callback = SDXLCFGCutoffCallback(cutoff_step_ratio=None, cutoff_step_index=10)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

prompt = "a sports car at the road, best quality, high quality, high detail, 8k resolution"

generator = torch.Generator(device="cpu").manual_seed(2628670641)

out = pipeline(
    prompt=prompt,
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
    generator=generator,
    callback_on_step_end=callback,
)

out.images[0].save("official_callback.png")
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/without_cfg_callback.png" alt="generated image of a sports car at the road" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">without SDXLCFGCutoffCallback</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/with_cfg_callback.png" alt="generated image of a a sports car at the road with cfg callback" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">with SDXLCFGCutoffCallback</figcaption>
  </div>
</div>

## Dynamic classifier-free guidance

Dynamic classifier-free guidance (CFG) is a feature that allows you to disable CFG after a certain number of inference steps which can help you save compute with minimal cost to performance. The callback function for this should have the following arguments:

- `pipeline` (or the pipeline instance) provides access to important properties such as `num_timesteps` and `guidance_scale`. You can modify these properties by updating the underlying attributes. For this example, you'll disable CFG by setting `pipeline._guidance_scale=0.0`.
- `step_index` and `timestep` tell you where you are in the denoising loop. Use `step_index` to turn off CFG after reaching 40% of `num_timesteps`.
- `callback_kwargs` is a dict that contains tensor variables you can modify during the denoising loop. It only includes variables specified in the `callback_on_step_end_tensor_inputs` argument, which is passed to the pipeline's `__call__` method. Different pipelines may use different sets of variables, so please check a pipeline's `_callback_tensor_inputs` attribute for the list of variables you can modify. Some common variables include `latents` and `prompt_embeds`. For this function, change the batch size of `prompt_embeds` after setting `guidance_scale=0.0` in order for it to work properly.

Your callback function should look something like this:

```python
def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
        # adjust the batch_size of prompt_embeds according to guidance_scale
        if step_index == int(pipeline.num_timesteps * 0.4):
                prompt_embeds = callback_kwargs["prompt_embeds"]
                prompt_embeds = prompt_embeds.chunk(2)[-1]

                # update guidance_scale and prompt_embeds
                pipeline._guidance_scale = 0.0
                callback_kwargs["prompt_embeds"] = prompt_embeds
        return callback_kwargs
```

Now, you can pass the callback function to the `callback_on_step_end` parameter and the `prompt_embeds` to `callback_on_step_end_tensor_inputs`.

```py
import torch
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

generator = torch.Generator(device="cuda").manual_seed(1)
out = pipeline(
    prompt,
    generator=generator,
    callback_on_step_end=callback_dynamic_cfg,
    callback_on_step_end_tensor_inputs=['prompt_embeds']
)

out.images[0].save("out_custom_cfg.png")
```

## Interrupt the diffusion process

> [!TIP]
> The interruption callback is supported for text-to-image, image-to-image, and inpainting for the [StableDiffusionPipeline](../api/pipelines/stable_diffusion/overview) and [StableDiffusionXLPipeline](../api/pipelines/stable_diffusion/stable_diffusion_xl).

Stopping the diffusion process early is useful when building UIs that work with Diffusers because it allows users to stop the generation process if they're unhappy with the intermediate results. You can incorporate this into your pipeline with a callback.

This callback function should take the following arguments: `pipeline`, `i`, `t`, and `callback_kwargs` (this must be returned). Set the pipeline's `_interrupt` attribute to `True` to stop the diffusion process after a certain number of steps. You are also free to implement your own custom stopping logic inside the callback.

In this example, the diffusion process is stopped after 10 steps even though `num_inference_steps` is set to 50.

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.enable_model_cpu_offload()
num_inference_steps = 50

def interrupt_callback(pipeline, i, t, callback_kwargs):
    stop_idx = 10
    if i == stop_idx:
        pipeline._interrupt = True

    return callback_kwargs

pipeline(
    "A photo of a cat",
    num_inference_steps=num_inference_steps,
    callback_on_step_end=interrupt_callback,
)
```

## Display image after each generation step

> [!TIP]
> This tip was contributed by [asomoza](https://github.com/asomoza).

Display an image after each generation step by accessing and converting the latents after each step into an image. The latent space is compressed to 128x128, so the images are also 128x128 which is useful for a quick preview.

1. Use the function below to convert the SDXL latents (4 channels) to RGB tensors (3 channels) as explained in the [Explaining the SDXL latent space](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space) blog post.

```py
def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)
```

2. Create a function to decode and save the latents into an image.

```py
def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    image = latents_to_rgb(latents)
    image.save(f"{step}.png")

    return callback_kwargs
```

3. Pass the `decode_tensors` function to the `callback_on_step_end` parameter to decode the tensors after each step. You also need to specify what you want to modify in the `callback_on_step_end_tensor_inputs` parameter, which in this case are the latents.

```py
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

image = pipeline(
    prompt="A croissant shaped like a cute bear.",
    negative_prompt="Deformed, ugly, bad anatomy",
    callback_on_step_end=decode_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]
```

<div class="flex gap-4 justify-center">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_0.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 0</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_19.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 19
    </figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_29.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 29</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_39.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 39</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_49.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 49</figcaption>
  </div>
</div>
