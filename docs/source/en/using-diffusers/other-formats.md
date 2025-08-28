<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Model formats

Diffusion models are typically stored in the Diffusers format or single-file format. Model files can be stored in various file types such as safetensors, dduf, or ckpt.

> [!TIP]
> Format refers to the directory structure and file refers to the file type.

This guide will show you how to load pipelines and models from these formats and files.

## Diffusers format

The Diffusers format stores each model (UNet, transformer, text encoder) in a separate subfolder. There are several benefits to storing models separately.

- Faster overall pipeline initialization because you can load the individual model you need or load them all in parallel.
- Reduced memory usage because you don't need to load all the pipeline components if you only need one model. [Reuse](./loading#reusing-models-in-multiple-pipelines) a model that is shared between multiple pipelines.
- Lower storage requirements because common models shared between multiple pipelines are only downloaded once.
- Flexibility to use new or improved models in a pipeline.

## Single file format

A single-file format stores *all* the model (UNet, transformer, text encoder) weights in a single file. Benefits of single-file formats include the following.

- Greater compatibility with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- Easier to download and share a single file.

Use [`~loaders.FromSingleFileMixin.from_single_file`] to load a single file.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

The [`~loaders.FromSingleFileMixin.from_single_file`] method also supports passing new models or schedulers.

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler()
pipeline = DiffusionPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    scheduler=scheduler,
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

### Configuration options


## File types

Models can be stored in several file types. Safetensors is the most common file type but you may encounter other file types on the Hub or diffusion community.

### safetensors

[Safetensors](https://hf.co/docs/safetensors) is a safe and fast file type for securely storing and loading tensors. It restricts the header size to limit certain types of attacks, supports lazy loading (useful for distributed setups), and generally loads faster.

Diffusers loads safetensors file by default if they are available and the Safetensors library is installed.

Use [`~DiffusionPipeline.from_pretrained`] or [`~loaders.FromSingleFileMixin.from_single_file`] to load safetensor files.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch.dtype=torch.float16,
    device_map="cuda"
)

pipeline = DiffusionPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

### ckpt

Older model weights are commonly saved with Python's [pickle](https://docs.python.org/3/library/pickle.html) utility in a ckpt file.

Pickled files may be unsafe because they can be exploited to execute malicious code. It is recommended to use safetensors files or convert the weights to safetensors files.

Use [`~loaders.FromSingleFileMixin.from_single_file`] to load a ckpt file.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_single_file(
    "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
)
```

### dduf

> [~TIP]
> DDUF is an experimental file type and the API may change. Refer to the DDUF [docs](https://huggingface.co/docs/hub/dduf) to learn more.

DDUF is a file type designed to unify different diffusion model distribution methods and weight-saving formats. It is a standardized and flexible method to package all components of a diffusion model into a single file, providing a balance between the Diffusers and single-file formats.

Use the `dduf_file` argument in [`~DiffusionPipeline.from_pretrained`] to load a DDUF file. You can also load quantized dduf files as long as they are stored in the Diffusers format.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "DDUF/FLUX.1-dev-DDUF",
    dduf_file="FLUX.1-dev.dduf",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

To save a pipeline as a dduf file, use the [`~huggingface_hub.export_folder_as_dduf`] utility.

```py
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import export_folder_as_dduf

pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

save_folder = "flux-dev"
pipeline.save_pretrained("flux-dev")
export_folder_as_dduf("flux-dev.dduf", folder_path=save_folder)
```

## Converting formats and files

Diffusers provides scripts and methods to convert format and files to enable broader support across the diffusion ecosystem.

Take a look at the [diffusers/scripts](https://github.com/huggingface/diffusers/tree/main/scripts) folder to find a conversion script. Scripts with `"to_diffusers` appended at the end converts a model to the Diffusers format. Each script has a specific set of arguments for configuring the conversion. Make sure you check what arguments are available.

The example below converts a model stored in Diffusers format to a single-file format. Provide the path to the model to convert and where to save the converted model. You can optionally specify what file type and data type to save the model as.

```bash
python convert_diffusers_to_original_sdxl.py --model_path path/to/model/to/convert --checkpoint_path path/to/save/model/to --use_safetensors
```

The [`~DiffusionPipeline.save_pretrained`] method also saves a model in Diffusers format and takes care of creating subfolders for each model. It saves the files as safetensor files by default.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
)
pipeline.save_pretrained()
```

Finally, you can use a Space like [SD To Diffusers](https://hf.co/spaces/diffusers/sd-to-diffusers) or [SD-XL To Diffusers](https://hf.co/spaces/diffusers/sdxl-to-diffusers) to convert models to the Diffusers format. It'll open a PR on your model repository with the converted files. This is the easiest way to convert a model, but it may fail for more complicated models. Using a conversion script is more reliable.

## Resources

- Learn more about the design decisions and why safetensor files are preferred for saving and loading model weights in the [Safetensors audited as really safe and becoming the default](https://blog.eleuther.ai/safetensors-security-audit/) blog post.

