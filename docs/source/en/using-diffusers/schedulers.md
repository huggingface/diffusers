<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Schedulers

A scheduler is an algorithm that provides instructions to the denoising process such as how much noise to remove at a certain step. It takes the model prediction from step *t* and applies an update for how to compute the next sample at step *t-1*. Different schedulers produce different results; some are faster while others are more accurate.

Diffusers supports many schedulers and allows you to modify their timestep schedules, timestep spacing, and more, to generate high-quality images in fewer steps.

This guide will show you how to load and customize schedulers.

## Loading schedulers

Schedulers don't have any parameters and are defined in a configuration file. Access the `.scheduler` attribute of a pipeline to view the configuration.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, device_map="cuda"
)
pipeline.scheduler
```

Load a different scheduler with [`~SchedulerMixin.from_pretrained`] and specify the `subfolder` argument to load the configuration file into the correct subfolder of the pipeline repository. Pass the new scheduler to the existing pipeline.

```py
from diffusers import DPMSolverMultistepScheduler

dpm = DPMSolverMultistepScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    scheduler=dpm,
    torch_dtype=torch.float16,
    device_map="cuda"
)
pipeline.scheduler
```

## Timestep schedules

Timestep or noise schedule decides how noise is distributed over the denoising process. The schedule can be linear or more concentrated toward the beginning or end. It is a precomputed sequence of noise levels generated from the scheduler's default configuration, but it can be customized to use other schedules.

> [!TIP]
> The `timesteps` argument is only supported for a select list of schedulers and pipelines. Feel free to open a feature request if you want to extend these parameters to a scheduler and pipeline that does not currently support it!

The example below uses the [Align Your Steps (AYS)](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) schedule which can generate a high-quality image in 10 steps, significantly speeding up generation and reducing computation time.

Import the schedule and pass it to the `timesteps` argument in the pipeline.

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.schedulers import AysSchedules

sampling_schedule = AysSchedules["StableDiffusionXLTimesteps"]
print(sampling_schedule)
"[999, 845, 730, 587, 443, 310, 193, 116, 53, 13]"

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
  pipeline.scheduler.config, algorithm_type="sde-dpmsolver++"
)

prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
image = pipeline(
    prompt=prompt,
    negative_prompt="",
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

### Rescaling schedules

Denoising should begin with pure noise and the signal-to-noise (SNR) ration should be zero. However, some models don't actually start from pure noise which makes it difficult to generate images at brightness extremes.

> [!TIP]
> Train your own model with `v_prediction` by adding the `--prediction_type="v_prediction"` flag to your training script. You can also [search](https://huggingface.co/search/full-text?q=v_prediction&type=model) for existing models trained with `v_prediction`.

To fix this, a model must be trained with `v_prediction`. If a model is trained with `v_prediction`, then enable the following arguments in the scheduler.

- Set `rescale_betas_zero_snr=True` to rescale the noise schedule to the very last timestep with exactly zero SNR
- Set `timestep_spacing="trailing"` to force sampling from the last timestep with pure noise

```py
from diffusers import DiffusionPipeline, DDIMScheduler

pipeline = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", device_map="cuda")

pipeline.scheduler = DDIMScheduler.from_config(
    pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
```

Set `guidance_rescale` in the pipeline to avoid overexposed images. A lower value increases brightness, but some details may appear washed out.

```py
prompt = """
cinematic photo of a snowy mountain at night with the northern lights aurora borealis
overhead, 35mm photograph, film, professional, 4k, highly detailed
"""
image = pipeline(prompt, guidance_rescale=0.7).images[0]
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/no-zero-snr.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">default Stable Diffusion v2-1 image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/zero-snr.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">image with zero SNR and trailing timestep spacing enabled</figcaption>
  </div>
</div>

## Timestep spacing

Timestep spacing refers to the specific steps *t* to sample from from the schedule. Diffusers provides three spacing types as shown below.

| spacing strategy | spacing calculation | example timesteps |
|---|---|---|
| `leading` | evenly spaced steps | `[900, 800, 700, ..., 100, 0]` |
| `linspace` | include first and last steps and evenly divide remaining intermediate steps | `[1000, 888.89, 777.78, ..., 111.11, 0]` |
| `trailing` | include last step and evenly divide remaining intermediate steps beginning from the end | `[999, 899, 799, 699, 599, 499, 399, 299, 199, 99]` |

Pass the spacing strategy to the `timestep_spacing` argument in the scheduler.

> [!TIP]
> The `trailing` strategy typically produces higher quality images with more details with fewer steps, but the difference in quality is not as obvious for more standard step values.

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
  pipeline.scheduler.config, timestep_spacing="trailing"
)

prompt = "A cinematic shot of a cute little black cat sitting on a pumpkin at night"
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=5,
).images[0]
image
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/trailing_spacing.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">trailing spacing after 5 steps</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/leading_spacing.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">leading spacing after 5 steps</figcaption>
  </div>
</div>

## Sigmas

Sigmas is a measure of how noisy a sample is at a certain step as defined by the schedule. When using custom `sigmas`, the `timesteps` are calculated from these values instead of the default scheduler configuration.

> [!TIP]
> The `sigmas` argument is only supported for a select list of schedulers and pipelines. Feel free to open a feature request if you want to extend these parameters to a scheduler and pipeline that does not currently support it!

Pass the custom sigmas to the `sigmas` argument in the pipeline. The example below uses the [sigmas](https://github.com/huggingface/diffusers/blob/6529ee67ec02fcf58d2fd9242164ea002b351d75/src/diffusers/schedulers/scheduling_utils.py#L55) from the 10-step AYS schedule.

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
  pipeline.scheduler.config, algorithm_type="sde-dpmsolver++"
)

sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0]
prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    sigmas=sigmas,
).images[0]
```

### Karras sigmas

[Karras sigmas](https://huggingface.co/papers/2206.00364) resamples the noise schedule for more efficient sampling by clustering sigmas more densely in the middle of the sequence where structure reconstruction is critical, while using fewer sigmas at the beginning and end where noise changes have less impact. This can increase the level of details in a generated image.

Set `use_karras_sigmas=True` in the scheduler to enable it.

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
  pipeline.scheduler.config,
  algorithm_type="sde-dpmsolver++",
  use_karras_sigmas=True,
)

prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    sigmas=sigmas,
).images[0]
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/karras_sigmas_true.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Karras sigmas enabled</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/karras_sigmas_false.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Karras sigmas disabled</figcaption>
  </div>
</div>

Refer to the scheduler API [overview](../api/schedulers/overview) for a list of schedulers that support Karras sigmas. It should only be used for models trained with Karras sigmas.

## Choosing a scheduler

It's important to try different schedulers to find the best one for your use case. Here are a few recommendations to help you get started.

- DPM++ 2M SDE Karras is generally a good all-purpose option.
- [`TCDScheduler`] works well for distilled models.
- [`FlowMatchEulerDiscreteScheduler`] and [`FlowMatchHeunDiscreteScheduler`] for FlowMatch models.
- [`EulerDiscreteScheduler`] or [`EulerAncestralDiscreteScheduler`] for generating anime style images.
- DPM++ 2M paired with [`LCMScheduler`] on SDXL for generating realistic images.

## Resources

- Read the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) paper for more details about rescaling the noise schedule to enforce zero SNR.