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
> Format refers to whether the weights are stored in a directory structure and file refers to the file type.

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
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

The [`~loaders.FromSingleFileMixin.from_single_file`] method also supports passing new models or schedulers.

```py
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", torch_dtype=torch.bfloat16
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

### Configuration options

Diffusers format models have a `config.json` file in their repositories with important attributes such as the number of layers and attention heads. The [`~loaders.FromSingleFileMixin.from_single_file`] method automatically determines the appropriate config to use from `config.json`. This may fail in a few rare instances though, in which case, you should use the `config` argument.

You should also use the `config` argument if the models in a pipeline are different from the original implementation or if it doesn't have the necessary metadata to determine the correct config.

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/segmind/SSD-1B/blob/main/SSD-1B.safetensors"

pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, config="segmind/SSD-1B")
```

Diffusers attempts to infer the pipeline components based on the signature types of the pipeline class when using `original_config` with `local_files_only=True`. It won't download the config files from a Hub repository to avoid backward breaking changes when you can't connect to the internet. This method isn't as reliable as providing a path to a local model with the `config` argument and may lead to errors. You should run the pipeline with `local_files_only=False` to download the config files to the local cache to avoid errors.

Override default configs by passing the arguments directly to [`~loaders.FromSingleFileMixin.from_single_file`]. The examples below demonstrate how to override the configs in a pipeline or model.

```py
from diffusers import StableDiffusionXLInstructPix2PixPipeline

ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
pipeline = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    ckpt_path, config="diffusers/sdxl-instructpix2pix-768", is_cosxl_edit=True
)
```

```py
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, upcast_attention=True)
```

### Local files

The [`~loaders.FromSingleFileMixin.from_single_file`] method attempts to configure a pipeline or model by inferring the model type from the keys in the checkpoint file. For example, any single file checkpoint based on the Stable Diffusion XL base model is configured from [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

If you're working with local files, download the config files with the [`~huggingface_hub.snapshot_download`] method and the model checkpoint with [`~huggingface_hub.hf_hub_download`]. These files are downloaded to your [cache directory](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache), but you can download them to a specific directory with the `local_dir` argument.

```py
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import StableDiffusionXLPipeline

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allow_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
)

pipeline = StableDiffusionXLPipeline.from_single_file(
    my_local_checkpoint_path, config=my_local_config_path, local_files_only=True
)
```

### Symlink

If you're working with a file system that does not support symlinking, download the checkpoint file to a local directory first with the `local_dir` parameter. Using the `local_dir` parameter automatically disables symlinks.

```py
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import StableDiffusionXLPipeline

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
    local_dir="my_local_checkpoints",
)
print("My local checkpoint: ", my_local_checkpoint_path)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allow_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
)
print("My local config: ", my_local_config_path)
```

Pass these paths to [`~loaders.FromSingleFileMixin.from_single_file`].

```py
pipeline = StableDiffusionXLPipeline.from_single_file(
    my_local_checkpoint_path, config=my_local_config_path, local_files_only=True
)
```

## File types

Models can be stored in several file types. Safetensors is the most common file type but you may encounter other file types on the Hub or diffusion community.

### safetensors

[Safetensors](https://hf.co/docs/safetensors) is a safe and fast file type for securely storing and loading tensors. It restricts the header size to limit certain types of attacks, supports lazy loading (useful for distributed setups), and generally loads faster.

Diffusers loads safetensors file by default (a required dependency) if they are available and the Safetensors library is installed.

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
)
```

If you're using a checkpoint trained with a Diffusers training script, metadata such as the LoRA configuration, is automatically saved. When the file is loaded, the metadata is parsed to correctly configure the LoRA and avoid missing or incorrect LoRA configs. Inspect the metadata of a safetensors file by clicking on the <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/safetensors/logo.png" alt="safetensors logo" style="vertical-align: middle; display: inline-block; max-height: 0.8em; max-width: 0.8em; margin: 0; padding: 0; line-height: 1;"> logo next to the file on the Hub.

Save the metadata for LoRAs that aren't trained with Diffusers with either `transformer_lora_adapter_metadata` or `unet_lora_adapter_metadata` depending on your model. For the text encoder, use the `text_encoder_lora_adapter_metadata` and `text_encoder_2_lora_adapter_metadata` arguments in [`~loaders.FluxLoraLoaderMixin.save_lora_weights`]. This is only supported for safetensors files.

```py
import torch
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")
pipeline.load_lora_weights("linoyts/yarn_art_Flux_LoRA")
pipeline.save_lora_weights(
    text_encoder_lora_adapter_metadata={"r": 8, "lora_alpha": 8},
    text_encoder_2_lora_adapter_metadata={"r": 8, "lora_alpha": 8}
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

> [!TIP]
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

