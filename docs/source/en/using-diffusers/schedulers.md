<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Schedulers

A scheduler tells the denoising loop how much noise to remove at each step. Different schedulers trade speed for quality.

This guide shows how to load a scheduler and customize its timestep schedule, spacing, and sigmas.

## Choosing a scheduler

Start from the checkpoint default. Swap only if you need a different speed or quality tradeoff.

- DPM++ 2M SDE Karras is a strong all-purpose option for many latent diffusion checkpoints.
- [`TCDScheduler`] works well for distilled models.
- Use [`FlowMatchEulerDiscreteScheduler`] or [`FlowMatchHeunDiscreteScheduler`] for FlowMatch models (Qwen-Image, Flux, and similar).
- [`EulerDiscreteScheduler`] or [`EulerAncestralDiscreteScheduler`] often work well for anime-style images.
- [`LCMScheduler`] with an LCM UNet or LoRA for few-step generation when the checkpoint supports it.

## Loading schedulers

> [!TIP]
> Flow-matching models such as Qwen-Image and Flux ship [`FlowMatchEulerDiscreteScheduler`] as their default. Keep that scheduler unless you are intentionally experimenting. Swap with [`~ConfigMixin.from_config`] only when the replacement is compatible with the checkpoint.

Schedulers are config-only and they do not ship weight tensors. Access the `.scheduler` attribute on a pipeline to inspect the loaded config.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipeline.scheduler
```

To swap schedulers on a loaded pipeline, use [`~ConfigMixin.from_config`] with the existing scheduler config so `num_train_timesteps` and related fields stay aligned. For FlowMatch checkpoints (Qwen-Image, Flux, and similar), keep [`FlowMatchEulerDiscreteScheduler`] unless you are intentionally experimenting with a compatible replacement.

```py
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
```

You can also load a scheduler config from the Hub with [`~SchedulerMixin.from_pretrained`] and pass it into [`~DiffusionPipeline.from_pretrained`] through `scheduler`.

```py
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

dpm = DPMSolverMultistepScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    scheduler=dpm,
    dtype=torch.float16,
    device_map="cuda",  # or "mps", "xpu", "cpu"
)
pipeline.scheduler
```

## Timestep schedules

Timestep or noise schedule decides how noise is distributed over the denoising process. The schedule can be linear or more concentrated toward the beginning or end. It is a precomputed sequence of noise levels generated from the scheduler's default configuration, but it can be customized to use other schedules.

```text
linear (even steps)                 AYS (denser where it matters)
noise                               noise
  ^                                   ^
  | *                                 | *
  |  *                                |   *
  |   *                               |    **
  |    *                              |     ***
  |     *                             |      **
  |      *                            |        *
  |       *                           |          *
  +-----------------> step            +-----------------> step
```

> [!TIP]
> Custom `timesteps` only work if that scheduler’s `set_timesteps` accepts the argument (pipelines check the signature via `retrieve_timesteps` and raise `ValueError` otherwise). Check the scheduler’s API page or `set_timesteps` signature before passing them.

The example below uses the [Align Your Steps (AYS)](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) schedule which can generate a high-quality image in 10 steps, significantly speeding up generation and reducing computation time.

Import the schedule and pass it to the `timesteps` argument in the pipeline.

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.schedulers import AysSchedules

sampling_schedule = AysSchedules["StableDiffusionXLTimesteps"]
print(sampling_schedule)
# [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
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

Denoising should begin with pure noise and the signal-to-noise (SNR) ratio should be zero. However, some models don't actually start from pure noise which makes it difficult to generate images at brightness extremes.

> [!TIP]
> Train your own model with `v_prediction` by adding the `--prediction_type="v_prediction"` flag to your training script. You can also [search](https://huggingface.co/search/full-text?q=v_prediction&type=model) for existing models trained with `v_prediction`.

To fix this, a model must be trained with `v_prediction`. If a model is trained with `v_prediction`, then enable the following arguments in the scheduler.

- Set `rescale_betas_zero_snr=True` to rescale the noise schedule to the very last timestep with exactly zero SNR
- Set `timestep_spacing="trailing"` to force sampling from the last timestep with pure noise

```py
from diffusers import DiffusionPipeline, DDIMScheduler

pipeline = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", device_map="cuda")  # or "mps", "xpu", "cpu"

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

Timestep spacing refers to the specific steps *t* to sample from the schedule. Diffusers provides three spacing types as shown below.

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
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
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

```text
step:   0    1    2    3    4
sigma:  σ0 > σ1 > σ2 > σ3 > σ4 ≈ 0
        high noise  --->  clean sample
```

> [!TIP]
> Custom `sigmas` only work if that scheduler’s `set_timesteps` accepts the argument (pipelines check the signature via `retrieve_timesteps` and raise `ValueError` otherwise). Check the scheduler’s API page or `set_timesteps` signature before passing them.

Pass the custom sigmas to the `sigmas` argument in the pipeline. The example below uses the [sigmas](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_utils.py) from the 10-step AYS schedule.

```py
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from diffusers.schedulers import AysSchedules

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    dtype=torch.float16,
    device_map="cuda",  # or "mps", "xpu", "cpu"
)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

sigmas = AysSchedules["StableDiffusionXLSigmas"]
prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    sigmas=sigmas,
).images[0]
```

### Karras sigmas

[Karras sigmas](https://huggingface.co/papers/2206.00364) resamples the noise schedule for more efficient sampling by clustering sigmas more densely in the middle of the sequence where structure reconstruction is critical, while using fewer sigmas at the beginning and end where noise changes have less impact. This can increase the level of details in a generated image.

```text
default σ:  *  *  *  *  *  *  *  *     even-ish
Karras σ:   *   * * * * *   *          denser mid, sparser ends
            |---structure---|
```

Set `use_karras_sigmas=True` in the scheduler to enable it.

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
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
    num_inference_steps=20,
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

## Next steps

- Read the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) paper for more details about rescaling the noise schedule to enforce zero SNR.
