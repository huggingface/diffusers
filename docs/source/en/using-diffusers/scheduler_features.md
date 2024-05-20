<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Scheduler features

The scheduler is an important component of any diffusion model because it controls the entire denoising (or sampling) process. There are many types of schedulers, some are optimized for speed and some for quality. With Diffusers, you can modify the scheduler configuration to use custom noise schedules, sigmas, and rescale the noise schedule. Changing these parameters can have profound effects on inference quality and speed.

This guide will demonstrate how to use these features to improve inference quality.

> [!TIP]
> Diffusers currently only supports the `timesteps` and `sigmas` parameters for a select list of schedulers and pipelines. Feel free to open a [feature request](https://github.com/huggingface/diffusers/issues/new/choose) if you want to extend these parameters to a scheduler and pipeline that does not currently support it!

## Timestep schedules

The timestep or noise schedule determines the amount of noise at each sampling step. The scheduler uses this to generate an image with the corresponding amount of noise at each step. The timestep schedule is generated from the scheduler's default configuration, but you can customize the scheduler to use new and optimized sampling schedules that aren't in Diffusers yet.

For example, [Align Your Steps (AYS)](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) is a method for optimizing a sampling schedule to generate a high-quality image in as little as 10 steps. This optimal schedule for 10 steps was calculated to be:

```py
from diffusers.schedulers import AysSchedules

sampling_schedule = AysSchedules["StableDiffusionXLTimesteps"]
print(sampling_schedule)
"[999, 845, 730, 587, 443, 310, 193, 116, 53, 13]"
```

You can use the AYS sampling schedule in a pipeline by passing it to the `timesteps` parameter.

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")

prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
generator = torch.Generator(device="cpu").manual_seed(2487854446)
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    generator=generator,
    timesteps=sampling_schedule,
).images[0]
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ays.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">AYS timestep schedule 10 steps</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/10.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Linearly-spaced timestep schedule 10 steps</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/25.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Linearly-spaced timestep schedule 25 steps</figcaption>
  </div>
</div>

## Sigmas

## Rescale noise schedule
