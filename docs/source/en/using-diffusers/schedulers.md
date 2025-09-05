<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load schedulers and models

[[open-in-colab]]

Diffusion pipelines are a collection of interchangeable schedulers and models that can be mixed and matched to tailor a pipeline to a specific use case. The scheduler encapsulates the entire denoising process such as the number of denoising steps and the algorithm for finding the denoised sample. A scheduler is not parameterized or trained so they don't take very much memory. The model is usually only concerned with the forward pass of going from a noisy input to a less noisy sample.

This guide will show you how to load schedulers and models to customize a pipeline. You'll use the [stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5) checkpoint throughout this guide, so let's load it first.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
```

You can see what scheduler this pipeline uses with the `pipeline.scheduler` attribute.

```py
pipeline.scheduler
PNDMScheduler {
  "_class_name": "PNDMScheduler",
  "_diffusers_version": "0.21.4",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "trained_betas": null
}
```

## Load a scheduler

Schedulers are defined by a configuration file that can be used by a variety of schedulers. Load a scheduler with the [`SchedulerMixin.from_pretrained`] method, and specify the `subfolder` parameter to load the configuration file into the correct subfolder of the pipeline repository.

For example, to load the [`DDIMScheduler`]:

```py
from diffusers import DDIMScheduler, DiffusionPipeline

ddim = DDIMScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
```

Then you can pass the newly loaded scheduler to the pipeline.

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", scheduler=ddim, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
```

## Compare schedulers

Schedulers have their own unique strengths and weaknesses, making it difficult to quantitatively compare which scheduler works best for a pipeline. You typically have to make a trade-off between denoising speed and denoising quality. We recommend trying out different schedulers to find one that works best for your use case. Call the `pipeline.scheduler.compatibles` attribute to see what schedulers are compatible with a pipeline.

Let's compare the [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`], and the [`DPMSolverMultistepScheduler`] on the following prompt and seed.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
generator = torch.Generator(device="cuda").manual_seed(8)
```

To change the pipelines scheduler, use the [`~ConfigMixin.from_config`] method to load a different scheduler's `pipeline.scheduler.config` into the pipeline.

<hfoptions id="schedulers">
<hfoption id="LMSDiscreteScheduler">

[`LMSDiscreteScheduler`] typically generates higher quality images than the default scheduler.

```py
from diffusers import LMSDiscreteScheduler

pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
<hfoption id="EulerDiscreteScheduler">

[`EulerDiscreteScheduler`] can generate higher quality images in just 30 steps.

```py
from diffusers import EulerDiscreteScheduler

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
<hfoption id="EulerAncestralDiscreteScheduler">

[`EulerAncestralDiscreteScheduler`] can generate higher quality images in just 30 steps.

```py
from diffusers import EulerAncestralDiscreteScheduler

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
<hfoption id="DPMSolverMultistepScheduler">

[`DPMSolverMultistepScheduler`] provides a balance between speed and quality and can generate higher quality images in just 20 steps.

```py
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
</hfoptions>

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_lms.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">LMSDiscreteScheduler</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_discrete.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">EulerDiscreteScheduler</figcaption>
  </div>
</div>
<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_ancestral.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">EulerAncestralDiscreteScheduler</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_dpm.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">DPMSolverMultistepScheduler</figcaption>
  </div>
</div>

Most images look very similar and are comparable in quality. Again, it often comes down to your specific use case so a good approach is to run multiple different schedulers and compare the results.

## Models

Models are loaded from the [`ModelMixin.from_pretrained`] method, which downloads and caches the latest version of the model weights and configurations. If the latest files are available in the local cache, [`~ModelMixin.from_pretrained`] reuses files in the cache instead of re-downloading them.

Models can be loaded from a subfolder with the `subfolder` argument. For example, the model weights for [stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5) are stored in the [unet](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet) subfolder.

```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", use_safetensors=True)
```

They can also be directly loaded from a [repository](https://huggingface.co/google/ddpm-cifar10-32/tree/main).

```python
from diffusers import UNet2DModel

unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
```

To load and save model variants, specify the `variant` argument in [`ModelMixin.from_pretrained`] and [`ModelMixin.save_pretrained`].

```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant="non_ema", use_safetensors=True
)
unet.save_pretrained("./local-unet", variant="non_ema")
```

Use the `torch_dtype` argument in [`~ModelMixin.from_pretrained`] to specify the dtype to load a model in.

```py
from diffusers import AutoModel

unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16
)
```

You can also use the [torch.Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html) method to convert to the specified dtype on the fly. It converts *all* weights unlike the `torch_dtype` argument that respects the `_keep_in_fp32_modules`. This is important for models whose layers must remain in fp32 for numerical stability and best generation quality (see example [here](https://github.com/huggingface/diffusers/blob/f864a9a352fa4a220d860bfdd1782e3e5af96382/src/diffusers/models/transformers/transformer_wan.py#L374)).
