<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Model formats

Diffusion models are typically stored in the Diffusers format or single-file format. Model files can be stored in various file types such as safetensors or ckpt.

> [!TIP]
> Format refers to whether the weights are stored in a directory structure. File type refers to how those weights are serialized, such as safetensors or ckpt.

```text
Diffusers format                    Single-file format
model/                              model.safetensors   (or .ckpt)
├─ model_index.json                 └─ all components in one file
├─ unet/  (or transformer/)
├─ text_encoder/
├─ vae/
└─ scheduler/
```

This guide will show you how to load pipelines and models from these formats and files.

## Diffusers format

The Diffusers format stores each model (UNet, transformer, text encoder) in a separate subfolder. There are several benefits to storing models separately.

- Faster overall pipeline initialization because you can load the individual model you need or load them all in parallel.
- Reduced memory usage because you don't need to load all the pipeline components if you only need one model. [Reuse](./loading#reusing-models-in-multiple-pipelines) a model that is shared between multiple pipelines.
- Lower storage requirements because common models shared between multiple pipelines are only downloaded once.
- Flexibility to use new or improved models in a pipeline.

## Single file format

A single-file format stores *all* the model weights (UNet, transformer, text encoder) in a single file. Benefits of single-file formats include:

- Greater compatibility with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- Easier to download and share a single file.

Use [`~loaders.FromSingleFileMixin.from_single_file`] to load a single file.

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    dtype=torch.float16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
)
```

The [`~loaders.FromSingleFileMixin.from_single_file`] method also supports passing new models or schedulers.

```py
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", dtype=torch.bfloat16
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    dtype=torch.bfloat16,
    device_map="cuda"  # or "mps", "xpu", "cpu"
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

When you pass `original_config` with `local_files_only=True`, Diffusers infers pipeline components from the pipeline class signature. It does not download config files from the Hub, which avoids breaking changes when you are offline. That path is less reliable than giving `config` a local model path, and it can error. Run once with `local_files_only=False` so the configs land in the local cache if you need them offline later.

Override default configs by passing the arguments directly to [`~loaders.FromSingleFileMixin.from_single_file`]. The examples below demonstrate how to override the configs in a pipeline or model.

<hfoptions id="config-options">
<hfoption id="pipeline">

```py
from diffusers import StableDiffusionXLInstructPix2PixPipeline

ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
pipeline = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    ckpt_path, config="diffusers/sdxl-instructpix2pix-768", is_cosxl_edit=True
)
```

</hfoption>
<hfoption id="model">

```py
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, upcast_attention=True)
```

</hfoption>
</hfoptions>

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

If you're working with a file system that doesn't support symlinking, download the checkpoint file to a local directory first with the `local_dir` parameter. Using the `local_dir` parameter automatically disables symlinks.

```py
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import StableDiffusionXLPipeline

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors",
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
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="cuda",  # or "mps", "xpu", "cpu"
)

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    dtype=torch.float16,
)
```

If you're using a checkpoint trained with a Diffusers training script, metadata such as the LoRA configuration is automatically saved. When the file is loaded, that metadata is parsed so the LoRA is configured correctly. Inspect the metadata of a safetensors file by clicking on the <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/safetensors/logo.png" alt="safetensors logo" style="vertical-align: middle; display: inline-block; max-height: 0.8em; max-width: 0.8em; margin: 0; padding: 0; line-height: 1;"> icon next to the file on the Hub.

Save LoRA adapter metadata for checkpoints that aren't trained with Diffusers by passing it to [`~loaders.FluxLoraLoaderMixin.save_lora_weights`]. Use `transformer_lora_adapter_metadata` for the transformer and `text_encoder_lora_adapter_metadata` for the text encoder. You must also pass `save_directory` and at least one of `transformer_lora_layers` or `text_encoder_lora_layers`. This path is only supported for safetensors files.

```py
import torch
from peft.utils import get_peft_model_state_dict
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipeline.load_lora_weights("linoyts/yarn_art_Flux_LoRA")

transformer_lora_layers = get_peft_model_state_dict(pipeline.transformer)
pipeline.save_lora_weights(
    save_directory="path/to/lora",
    transformer_lora_layers=transformer_lora_layers,
    transformer_lora_adapter_metadata={"r": 8, "lora_alpha": 8},
)
```

### ckpt

Older model weights are commonly saved with Python's [pickle](https://docs.python.org/3/library/pickle.html) utility in a ckpt file.

Pickled files may be unsafe because they can be exploited to execute malicious code. It is recommended to use safetensors files or convert the weights to safetensors files.

Use [`~loaders.FromSingleFileMixin.from_single_file`] to load a ckpt file.

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
)
```

### GGUF

GGUF stores prequantized weights in a single file. Diffusers loads GGUF through model [`~loaders.FromSingleFileMixin.from_single_file`] with [`GGUFQuantizationConfig`]. Pipeline-level GGUF loading is not supported.

See [GGUF](../quantization/gguf) for install steps and full examples.

```py
import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
)
```

## Converting formats and files

Diffusers provides scripts and methods to convert formats and files so more tools in the diffusion ecosystem can use them.

Take a look at the [diffusers/scripts](https://github.com/huggingface/diffusers/tree/main/scripts) folder to find a conversion script. Scripts with `"to_diffusers"` appended at the end convert a model to the Diffusers format. Each script has a specific set of arguments for configuring the conversion. Make sure you check what arguments are available.

The example below converts a model stored in Diffusers format to a single-file format. Provide the path to the model to convert and where to save the converted model. You can optionally specify what file type and data type to save the model as.

```bash
python convert_diffusers_to_original_sdxl.py --model_path path/to/model/to/convert --checkpoint_path path/to/save/model/to --use_safetensors
```

The [`~DiffusionPipeline.save_pretrained`] method also saves a model in Diffusers format and takes care of creating subfolders for each model. It saves the files as safetensor files by default.

```py
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
)
pipeline.save_pretrained("path/to/save/model")
```

## Resources

- Learn more about the design decisions and why safetensor files are preferred for saving and loading model weights in the [Safetensors audited as really safe and becoming the default](https://blog.eleuther.ai/safetensors-security-audit/) blog post.
