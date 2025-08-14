<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# DiffusionPipeline

Diffusion models consists of multiple components like UNets or diffusion transformers (DiTs), text encoders, variational autoencoders (VAEs), and schedulers. The [`DiffusionPipeline`] wraps all of these components into a single easy-to-use API without giving up the flexibility to modify it's components.

This guide will show you how to load a [`DiffusionPipeline`].

## Loading a pipeline

[`DiffusionPipeline`] is a base pipeline class that automatically selects and returns an instance of a model's pipeline subclass, like [`QwenImagePipeline`], by scanning the `model_index.json` file for the class name.

Pass a model id to [`~DiffusionPipeline.from_pretrained`] to load a pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image",
  torch_dtype=torch.bfloat16
).to("cuda")
```

Every model has a specific pipeline subclass that inherits from [`DiffusionPipeline`]. A subclass usually has a narrow focus and are task-specific. See the table below for an example.

| pipeline subclass | task |
|---|---|
| [`QwenImagePipeline`] | text-to-image |
| [`QwenImageImg2ImgPipeline`] | image-to-image |
| [`QwenImageInpaintPipeline`] | inpaint |

You could use the subclass directly by passing a model id to [`~QwenImagePipeline.from_pretrained`].

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
  "Qwen/Qwen-Image",
  torch_dtype=torch.bfloat16
).to("cuda")
```

### Local pipelines

Pipelines can also be run locally. Download a checkpoint to your setup with [git-lfs](https://git-lfs.github.com/).

```bash
git-lfs install
git clone https://huggingface.co/Qwen/Qwen-Image
```

The model is downloaded to a local folder. Pass the folder path to [`~QwenImagePipeline.from_pretrained`] to load it.

```py
import torch
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(
  "path/to/local/Qwen/Qwen-Image",
  torch_dtype=torch.bfloat16
).to("cuda")
```

The [`~QwenImagePipeline.from_pretrained`] method won't download files from the Hub when it detects a local path. But this also means it won't download and cache any updates that have been made to the model.

## Pipeline data types

Use the `torch_dtype` argument in [`~DiffusionPipeline.from_pretrained`] to load a model with a specific data type. This allows you to load different models in different precisions. For example, loading a large transformer model in half-precision reduces the memory required.

Pass the data type for each model as a dictionary to `torch_dtype`. Use the `default` key to set the default data type. If a model isn't in the dictionary and `default` isn't provided, it is loaded in full precision (`torch.float32`).

```py
import torch
from diffusers import HunyuanVideoPipeline

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    torch_dtype={"transformer": torch.bfloat16, "default": torch.float16},
)
print(pipeline.transformer.dtype, pipeline.vae.dtype)
```

You don't need to use a dictionary if you're loading all the models in the same data type.

```py
import torch
from diffusers import HunyuanVideoPipeline

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    torch_dtype=torch.bfloat16
)
print(pipeline.transformer.dtype, pipeline.vae.dtype)
```

## Parallel loading

Large models are often [sharded](../training/distributed_inference#model-sharding) into smaller files so that they are easier to load. Diffusers supports loading shards in parallel to speed up the loading process.

Set the environment variables below to enable parallel loading.

- Set `HF_ENABLE_PARALLEL_LOADING` to `"YES"` to enable parallel loading of shards.
- Set `HF_PARALLEL_LOADING_WORKERS` to configure the number of parallel threads to use when loading shards. More workers loads a model faster but uses more memory.

The `device_map` argument should be set to `"cuda"` to pre-allocate a large chunk of memory based on the model size. This substantially reduces model load time because warming up the memory allocator now avoids many smaller calls to the allocator later.

```py
import os
import torch
from diffusers import DiffusionPipeline

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "12"

pipeline = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

## Replacing models in a pipeline

[`DiffusionPipeline`] is flexible and accommodates loading different models or schedulers. You can experiment with different schedulers to optimize for generation speed or quality, and you can replace models with more performant ones.

The example below swaps the default scheduler to generate higher quality images and a more stable VAE version. Pass the `subfolder` argument in [`~HeunDiscreteScheduler.from_pretrained`] to load the scheduler to the correct subfolder.

```py
import torch
from diffusers import DiffusionPipeline, HeunDiscreteScheduler, AutoModel

scheduler = HeunDiscreteScheduler.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)
vae = AutoModel.from_pretrained(
  "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  scheduler=scheduler,
  vae=vae,
  torch_dtype=torch.float16,
).to("cuda")
```

## Reusing models in multiple pipelines

When working with multiple pipelines that use the same model, the [`~DiffusionPipeline.from_pipe`] method enables reusing a model instead of reloading it each time. This allows you to use multiple pipelines without increasing memory usage.

Memory usage is determined by the pipeline with the highest memory requirement regardless of the number of pipelines.

The example below loads a pipeline and then loads a second pipeline with [`~DiffusionPipeline.from_pipe`] to use [perturbed-attention guidance (PAG)](../api/pipelines/pag) to improve generation quality.

> [!WARNING]
> Use [`AutoPipelineForText2Image`] instead because [`DiffusionPipeline`] doesn't support PAG. Refer to the [AutoPipeline](../tutorials/autopipeline) docs to learn more. 

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline_sdxl = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image = pipeline_sdxl(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
# Max memory reserved: 10.47 GB
```

Set `enable_pag=True` in the second pipeline to enable PAG. The second pipeline uses the same amount of memory because it shares model weights with the first one.

```py
pipeline = AutoPipelineForText2Image.from_pipe(
  pipeline_sdxl, enable_pag=True
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image = pipeline(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
# Max memory reserved: 10.47 GB
```

> [!WARNING]
> Pipelines created by [`~DiffusionPipeline.from_pipe`] share the same models and *state*. Modifying the state of a model in one pipeline affects all the other pipelines that share the same model.

Some methods may not work correctly on pipelines created with [`~DiffusionPipeline.from_pipe`]. For example, [`~DiffusionPipeline.enable_model_cpu_offload`] relies on a unique model execution order, which may differ in the new pipeline. To ensure proper functionality, reapply these methods on the new pipeline.