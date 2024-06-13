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

For example, [Align Your Steps (AYS)](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) is a method for optimizing a sampling schedule to generate a high-quality image in as little as 10 steps. The optimal [10-step schedule](https://github.com/huggingface/diffusers/blob/a7bf77fc284810483f1e60afe34d1d27ad91ce2e/src/diffusers/schedulers/scheduling_utils.py#L51) for Stable Diffusion XL is:

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
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++")

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

## Timestep spacing

The way sample steps are selected in the schedule can affect the quality of the generated image, especially with respect to [rescaling the noise schedule](#rescale-noise-schedule), which can enable a model to generate much brighter or darker images. Diffusers provides three timestep spacing methods:

- `leading` creates evenly spaced steps
- `linspace` includes the first and last steps and evenly selects the remaining intermediate steps
- `trailing` only includes the last step and evenly selects the remaining intermediate steps starting from the end

It is recommended to use the `trailing` spacing method because it generates higher quality images with more details when there are fewer sample steps. But the difference in quality is not as obvious for more standard sample step values.

```py
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

prompt = "A cinematic shot of a cute little black cat sitting on a pumpkin at night"
generator = torch.Generator(device="cpu").manual_seed(2487854446)
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    generator=generator,
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

The `sigmas` parameter is the amount of noise added at each timestep according to the timestep schedule. Like the `timesteps` parameter, you can customize the `sigmas` parameter to control how much noise is added at each step. When you use a custom `sigmas` value, the `timesteps` are calculated from the custom `sigmas` value and the default scheduler configuration is ignored.

For example, you can manually pass the [sigmas](https://github.com/huggingface/diffusers/blob/6529ee67ec02fcf58d2fd9242164ea002b351d75/src/diffusers/schedulers/scheduling_utils.py#L55) for something like the 10-step AYS schedule from before to the pipeline.

```py
import torch

from diffusers import DiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16,
  variant="fp16",
).to("cuda")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0]
prompt = "anthropomorphic capybara wearing a suit and working with a computer"
generator = torch.Generator(device='cuda').manual_seed(123)
image = pipeline(
    prompt=prompt,
    num_inference_steps=10,
    sigmas=sigmas,
    generator=generator
).images[0]
```

When you take a look at the scheduler's `timesteps` parameter, you'll see that it is the same as the AYS timestep schedule because the `timestep` schedule is calculated from the `sigmas`.

```py
print(f" timesteps: {pipe.scheduler.timesteps}")
"timesteps: tensor([999., 845., 730., 587., 443., 310., 193., 116.,  53.,  13.], device='cuda:0')"
```

### Karras sigmas

> [!TIP]
> Refer to the scheduler API [overview](../api/schedulers/overview) for a list of schedulers that support Karras sigmas.
>
> Karras sigmas should not be used for models that weren't trained with them. For example, the base Stable Diffusion XL model shouldn't use Karras sigmas but the [DreamShaperXL](https://hf.co/Lykon/dreamshaper-xl-1-0) model can since they are trained with Karras sigmas.

Karras scheduler's use the timestep schedule and sigmas from the [Elucidating the Design Space of Diffusion-Based Generative Models](https://hf.co/papers/2206.00364) paper. This scheduler variant applies a smaller amount of noise per step as it approaches the end of the sampling process compared to other schedulers, and can increase the level of details in the generated image.

Enable Karras sigmas by setting `use_karras_sigmas=True` in the scheduler.

```py
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)

prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
generator = torch.Generator(device="cpu").manual_seed(2487854446)
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    generator=generator,
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

## Rescale noise schedule

In the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://hf.co/papers/2305.08891) paper, the authors discovered that common noise schedules allowed some signal to leak into the last timestep. This signal leakage at inference can cause models to only generate images with medium brightness. By enforcing a zero signal-to-noise ratio (SNR) for the timstep schedule and sampling from the last timestep, the model can be improved to generate very bright or dark images.

> [!TIP]
> For inference, you need a model that has been trained with *v_prediction*. To train your own model with *v_prediction*, add the following flag to the [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) or [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) scripts.
>
> ```bash
> --prediction_type="v_prediction"
> ```

For example, load the [ptx0/pseudo-journey-v2](https://hf.co/ptx0/pseudo-journey-v2) checkpoint which was trained with `v_prediction` and the [`DDIMScheduler`]. Configure the following parameters in the [`DDIMScheduler`]:

* `rescale_betas_zero_snr=True` to rescale the noise schedule to zero SNR
* `timestep_spacing="trailing"` to start sampling from the last timestep

Set `guidance_rescale` in the pipeline to prevent over-exposure. A lower value increases brightness but some of the details may appear washed out.

```py
from diffusers import DiffusionPipeline, DDIMScheduler

pipeline = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", use_safetensors=True)

pipeline.scheduler = DDIMScheduler.from_config(
    pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipeline.to("cuda")
prompt = "cinematic photo of a snowy mountain at night with the northern lights aurora borealis overhead, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(23)
image = pipeline(prompt, guidance_rescale=0.7, generator=generator).images[0]
image
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
