<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load pipelines

[[open-in-colab]]

Diffusion systems consist of multiple components like parameterized models and schedulers that interact in complex ways. That is why we designed the [`DiffusionPipeline`] to wrap the complexity of the entire diffusion system into an easy-to-use API. At the same time, the [`DiffusionPipeline`] is entirely customizable so you can modify each component to build a diffusion system for your use case.

This guide will show you how to load:

- pipelines from the Hub and locally
- different components into a pipeline
- multiple pipelines without increasing memory usage
- checkpoint variants such as different floating point types or non-exponential mean averaged (EMA) weights

## Load a pipeline

> [!TIP]
> Skip to the [DiffusionPipeline explained](#diffusionpipeline-explained) section if you're interested in an explanation about how the [`DiffusionPipeline`] class works.

There are two ways to load a pipeline for a task:

1. Load the generic [`DiffusionPipeline`] class and allow it to automatically detect the correct pipeline class from the checkpoint.
2. Load a specific pipeline class for a specific task.

<hfoptions id="pipelines">
<hfoption id="generic pipeline">

The [`DiffusionPipeline`] class is a simple and generic way to load the latest trending diffusion model from the [Hub](https://huggingface.co/models?library=diffusers&sort=trending). It uses the [`~DiffusionPipeline.from_pretrained`] method to automatically detect the correct pipeline class for a task from the checkpoint, downloads and caches all the required configuration and weight files, and returns a pipeline ready for inference.

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

This same checkpoint can also be used for an image-to-image task. The [`DiffusionPipeline`] class can handle any task as long as you provide the appropriate inputs. For example, for an image-to-image task, you need to pass an initial image to the pipeline.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=init_image).images[0]
```

</hfoption>
<hfoption id="specific pipeline">

Checkpoints can be loaded by their specific pipeline class if you already know it. For example, to load a Stable Diffusion model, use the [`StableDiffusionPipeline`] class.

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

This same checkpoint may also be used for another task like image-to-image. To differentiate what task you want to use the checkpoint for, you have to use the corresponding task-specific pipeline class. For example, to use the same checkpoint for image-to-image, use the [`StableDiffusionImg2ImgPipeline`] class.

```py
from diffusers import StableDiffusionImg2ImgPipeline

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

</hfoption>
</hfoptions>

Use the Space below to gauge a pipeline's memory requirements before you download and load it to see if it runs on your hardware.

<div class="block dark:hidden">
	<iframe
        src="https://diffusers-compute-pipeline-size.hf.space?__theme=light"
        width="850"
        height="1600"
    ></iframe>
</div>
<div class="hidden dark:block">
    <iframe
        src="https://diffusers-compute-pipeline-size.hf.space?__theme=dark"
        width="850"
        height="1600"
    ></iframe>
</div>

### Local pipeline

To load a pipeline locally, use [git-lfs](https://git-lfs.github.com/) to manually download a checkpoint to your local disk.

```bash
git-lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

This creates a local folder, ./stable-diffusion-v1-5, on your disk and you should pass its path to [`~DiffusionPipeline.from_pretrained`].

```python
from diffusers import DiffusionPipeline

stable_diffusion = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

The [`~DiffusionPipeline.from_pretrained`] method won't download files from the Hub when it detects a local path, but this also means it won't download and cache the latest changes to a checkpoint.

## Customize a pipeline

You can customize a pipeline by loading different components into it. This is important because you can:

- change to a scheduler with faster generation speed or higher generation quality depending on your needs (call the `scheduler.compatibles` method on your pipeline to see compatible schedulers)
- change a default pipeline component to a newer and better performing one

For example, let's customize the default [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0) checkpoint with:

- The [`HeunDiscreteScheduler`] to generate higher quality images at the expense of slower generation speed. You must pass the `subfolder="scheduler"` parameter in [`~HeunDiscreteScheduler.from_pretrained`] to load the scheduler configuration into the correct [subfolder](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/scheduler) of the pipeline repository.
- A more stable VAE that runs in fp16.

```py
from diffusers import StableDiffusionXLPipeline, HeunDiscreteScheduler, AutoencoderKL
import torch

scheduler = HeunDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
```

Now pass the new scheduler and VAE to the [`StableDiffusionXLPipeline`].

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  scheduler=scheduler,
  vae=vae,
  torch_dtype=torch.float16,
  variant="fp16",
  use_safetensors=True
).to("cuda")
```

## Reuse a pipeline

When you load multiple pipelines that share the same model components, it makes sense to reuse the shared components instead of reloading everything into memory again, especially if your hardware is memory-constrained. For example:

1. You generated an image with the [`StableDiffusionPipeline`] but you want to improve its quality with the [`StableDiffusionSAGPipeline`]. Both of these pipelines share the same pretrained model, so it'd be a waste of memory to load the same model twice.
2. You want to add a model component, like a [`MotionAdapter`](../api/pipelines/animatediff#animatediffpipeline), to [`AnimateDiffPipeline`] which was instantiated from an existing [`StableDiffusionPipeline`]. Again, both pipelines share the same pretrained model, so it'd be a waste of memory to load an entirely new pipeline again.

With the [`DiffusionPipeline.from_pipe`] API, you can switch between multiple pipelines to take advantage of their different features without increasing memory-usage. It is similar to turning on and off a feature in your pipeline.

> [!TIP]
> To switch between tasks (rather than features), use the [`~DiffusionPipeline.from_pipe`] method with the [AutoPipeline](../api/pipelines/auto_pipeline) class, which automatically identifies the pipeline class based on the task (learn more in the [AutoPipeline](../tutorials/autopipeline) tutorial).

Let's start with a [`StableDiffusionPipeline`] and then reuse the loaded model components to create a [`StableDiffusionSAGPipeline`] to increase generation quality. You'll use the [`StableDiffusionPipeline`] with an [IP-Adapter](./ip_adapter) to generate a bear eating pizza.

```python
from diffusers import DiffusionPipeline, StableDiffusionSAGPipeline
import torch
import gc
from diffusers.utils import load_image
from accelerate.utils import compute_module_sizes

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")

pipe_sd = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", torch_dtype=torch.float16)
pipe_sd.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_sd.set_ip_adapter_scale(0.6)
pipe_sd.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
    prompt="bear eats pizza",
    negative_prompt="wrong white balance, dark, sketches,worst quality,low quality",
    ip_adapter_image=image,
    num_inference_steps=50,
    generator=generator,
).images[0]
out_sd
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sd_0.png"/>
</div>

For reference, you can check how much memory this process consumed.

```python
def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"Max memory allocated: 4.406213283538818 GB"
```

Now, reuse the same pipeline components from [`StableDiffusionPipeline`] in [`StableDiffusionSAGPipeline`] with the [`~DiffusionPipeline.from_pipe`] method.

> [!WARNING]
> Some pipeline methods may not function properly on new pipelines created with [`~DiffusionPipeline.from_pipe`]. For instance, the [`~DiffusionPipeline.enable_model_cpu_offload`] method installs hooks on the model components based on a unique offloading sequence for each pipeline. If the models are executed in a different order in the new pipeline, the CPU offloading may not work correctly.
>
> To ensure everything works as expected, we recommend re-applying a pipeline method on a new pipeline created with [`~DiffusionPipeline.from_pipe`].

```python
pipe_sag = StableDiffusionSAGPipeline.from_pipe(
    pipe_sd
)

generator = torch.Generator(device="cpu").manual_seed(33)
out_sag = pipe_sag(
    prompt="bear eats pizza",
    negative_prompt="wrong white balance, dark, sketches,worst quality,low quality",
    ip_adapter_image=image,
    num_inference_steps=50,
    generator=generator,
    guidance_scale=1.0,
    sag_scale=0.75
).images[0]
out_sag
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sag_1.png"/>
</div>

If you check the memory usage, you'll see it remains the same as before because [`StableDiffusionPipeline`] and [`StableDiffusionSAGPipeline`] are sharing the same pipeline components. This allows you to use them interchangeably without any additional memory overhead.

```py
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"Max memory allocated: 4.406213283538818 GB"
```

Let's animate the image with the [`AnimateDiffPipeline`] and also add a [`MotionAdapter`] module to the pipeline. For the [`AnimateDiffPipeline`], you need to unload the IP-Adapter first and reload it *after* you've created your new pipeline (this only applies to the [`AnimateDiffPipeline`]).

```py
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

pipe_sag.unload_ip_adapter()
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

pipe_animate = AnimateDiffPipeline.from_pipe(pipe_sd, motion_adapter=adapter)
pipe_animate.scheduler = DDIMScheduler.from_config(pipe_animate.scheduler.config, beta_schedule="linear")
# load IP-Adapter and LoRA weights again
pipe_animate.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_animate.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe_animate.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
pipe_animate.set_adapters("zoom-out", adapter_weights=0.75)
out = pipe_animate(
    prompt="bear eats pizza",
    num_frames=16,
    num_inference_steps=50,
    ip_adapter_image=image,
    generator=generator,
).frames[0]
export_to_gif(out, "out_animate.gif")
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_animate_3.gif"/>
</div>

The [`AnimateDiffPipeline`] is more memory-intensive and consumes 15GB of memory (see the [Memory-usage of from_pipe](#memory-usage-of-from_pipe) section to learn what this means for your memory-usage).

```py
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"Max memory allocated: 15.178664207458496 GB"
```

### Modify from_pipe components

Pipelines loaded with [`~DiffusionPipeline.from_pipe`] can be customized with different model components or methods. However, whenever you modify the *state* of the model components, it affects all the other pipelines that share the same components. For example, if you call [`~diffusers.loaders.IPAdapterMixin.unload_ip_adapter`] on the [`StableDiffusionSAGPipeline`], you won't be able to use IP-Adapter with the [`StableDiffusionPipeline`] because it's been removed from their shared components.

```py
pipe.sag_unload_ip_adapter()

generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
    prompt="bear eats pizza",
    negative_prompt="wrong white balance, dark, sketches,worst quality,low quality",
    ip_adapter_image=image,
    num_inference_steps=50,
    generator=generator,
).images[0]
"AttributeError: 'NoneType' object has no attribute 'image_projection_layers'"
```

### Memory usage of from_pipe

The memory requirement of loading multiple pipelines with [`~DiffusionPipeline.from_pipe`] is determined by the pipeline with the highest memory-usage regardless of the number of pipelines you create.

| Pipeline | Memory usage (GB) |
|---|---|
| StableDiffusionPipeline | 4.400 |
| StableDiffusionSAGPipeline | 4.400 |
| AnimateDiffPipeline | 15.178 |

The [`AnimateDiffPipeline`] has the highest memory requirement, so the *total memory-usage* is based only on the [`AnimateDiffPipeline`]. Your memory-usage will not increase if you create additional pipelines as long as their memory requirements doesn't exceed that of the [`AnimateDiffPipeline`]. Each pipeline can be used interchangeably without any additional memory overhead.

## Safety checker

Diffusers implements a [safety checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) for Stable Diffusion models which can generate harmful content. The safety checker screens the generated output against known hardcoded not-safe-for-work (NSFW) content. If for whatever reason you'd like to disable the safety checker, pass `safety_checker=None` to the [`~DiffusionPipeline.from_pretrained`] method.

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, use_safetensors=True)
"""
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide by the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend keeping the safety filter enabled in all public-facing circumstances, disabling it only for use cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
"""
```

## Checkpoint variants

A checkpoint variant is usually a checkpoint whose weights are:

- Stored in a different floating point type, such as [torch.float16](https://pytorch.org/docs/stable/tensors.html#data-types), because it only requires half the bandwidth and storage to download. You can't use this variant if you're continuing training or using a CPU.
- Non-exponential mean averaged (EMA) weights which shouldn't be used for inference. You should use this variant to continue finetuning a model.

> [!TIP]
> When the checkpoints have identical model structures, but they were trained on different datasets and with a different training setup, they should be stored in separate repositories. For example, [stabilityai/stable-diffusion-2](https://hf.co/stabilityai/stable-diffusion-2) and [stabilityai/stable-diffusion-2-1](https://hf.co/stabilityai/stable-diffusion-2-1) are stored in separate repositories.

Otherwise, a variant is **identical** to the original checkpoint. They have exactly the same serialization format (like [safetensors](./using_safetensors)), model structure, and their weights have identical tensor shapes.

| **checkpoint type** | **weight name**                             | **argument for loading weights** |
|---------------------|---------------------------------------------|----------------------------------|
| original            | diffusion_pytorch_model.safetensors         |                                  |
| floating point      | diffusion_pytorch_model.fp16.safetensors    | `variant`, `torch_dtype`         |
| non-EMA             | diffusion_pytorch_model.non_ema.safetensors | `variant`                        |

There are two important arguments for loading variants:

- `torch_dtype` specifies the floating point precision of the loaded checkpoint. For example, if you want to save bandwidth by loading a fp16 variant, you should set `variant="fp16"` and `torch_dtype=torch.float16` to *convert the weights* to fp16. Otherwise, the fp16 weights are converted to the default fp32 precision.

  If you only set `torch_dtype=torch.float16`, the default fp32 weights are downloaded first and then converted to fp16.

- `variant` specifies which files should be loaded from the repository. For example, if you want to load a non-EMA variant of a UNet from [runwayml/stable-diffusion-v1-5](https://hf.co/runwayml/stable-diffusion-v1-5/tree/main/unet), set `variant="non_ema"` to download the `non_ema` file.

<hfoptions id="variants">
<hfoption id="fp16">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
```

</hfoption>
<hfoption id="non-EMA">

```py
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="non_ema", use_safetensors=True
)
```

</hfoption>
</hfoptions>

Use the `variant` parameter in the [`DiffusionPipeline.save_pretrained`] method to save a checkpoint as a different floating point type or as a non-EMA variant. You should try save a variant to the same folder as the original checkpoint, so you have the option of loading both from the same folder.

<hfoptions id="save">
<hfoption id="fp16">

```python
from diffusers import DiffusionPipeline

pipeline.save_pretrained("runwayml/stable-diffusion-v1-5", variant="fp16")
```

</hfoption>
<hfoption id="non_ema">

```py
pipeline.save_pretrained("runwayml/stable-diffusion-v1-5", variant="non_ema")
```

</hfoption>
</hfoptions>

If you don't save the variant to an existing folder, you must specify the `variant` argument otherwise it'll throw an `Exception` because it can't find the original checkpoint.

```python
# 👎 this won't work
pipeline = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
# 👍 this works
pipeline = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
```

## DiffusionPipeline explained

As a class method, [`DiffusionPipeline.from_pretrained`] is responsible for two things:

- Download the latest version of the folder structure required for inference and cache it. If the latest folder structure is available in the local cache, [`DiffusionPipeline.from_pretrained`] reuses the cache and won't redownload the files.
- Load the cached weights into the correct pipeline [class](../api/pipelines/overview#diffusers-summary) - retrieved from the `model_index.json` file - and return an instance of it.

The pipelines' underlying folder structure corresponds directly with their class instances. For example, the [`StableDiffusionPipeline`] corresponds to the folder structure in [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5).

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
print(pipeline)
```

You'll see pipeline is an instance of [`StableDiffusionPipeline`], which consists of seven components:

- `"feature_extractor"`: a [`~transformers.CLIPImageProcessor`] from 🤗 Transformers.
- `"safety_checker"`: a [component](https://github.com/huggingface/diffusers/blob/e55687e1e15407f60f32242027b7bb8170e58266/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L32) for screening against harmful content.
- `"scheduler"`: an instance of [`PNDMScheduler`].
- `"text_encoder"`: a [`~transformers.CLIPTextModel`] from 🤗 Transformers.
- `"tokenizer"`: a [`~transformers.CLIPTokenizer`] from 🤗 Transformers.
- `"unet"`: an instance of [`UNet2DConditionModel`].
- `"vae"`: an instance of [`AutoencoderKL`].

```json
StableDiffusionPipeline {
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

Compare the components of the pipeline instance to the [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) folder structure, and you'll see there is a separate folder for each of the components in the repository:

```
.
├── feature_extractor
│   └── preprocessor_config.json
├── model_index.json
├── safety_checker
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   ├── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── scheduler
│   └── scheduler_config.json
├── text_encoder
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   |── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
|   |── diffusion_pytorch_model.fp16.bin
│   |── diffusion_pytorch_model.f16.safetensors
│   |── diffusion_pytorch_model.non_ema.bin
│   |── diffusion_pytorch_model.non_ema.safetensors
│   └── diffusion_pytorch_model.safetensors
|── vae
.   ├── config.json
.   ├── diffusion_pytorch_model.bin
    ├── diffusion_pytorch_model.fp16.bin
    ├── diffusion_pytorch_model.fp16.safetensors
    └── diffusion_pytorch_model.safetensors
```

You can access each of the components of the pipeline as an attribute to view its configuration:

```py
pipeline.tokenizer
CLIPTokenizer(
    name_or_path="/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/tokenizer",
    vocab_size=49408,
    model_max_length=77,
    is_fast=False,
    padding_side="right",
    truncation_side="right",
    special_tokens={
        "bos_token": AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "eos_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "unk_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "pad_token": "<|endoftext|>",
    },
    clean_up_tokenization_spaces=True
)
```

Every pipeline expects a [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json) file that tells the [`DiffusionPipeline`]:

- which pipeline class to load from `_class_name`
- which version of 🧨 Diffusers was used to create the model in `_diffusers_version`
- what components from which library are stored in the subfolders (`name` corresponds to the component and subfolder name, `library` corresponds to the name of the library to load the class from, and `class` corresponds to the class name)

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```
