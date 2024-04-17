<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Loading Pipelines and Models via `from_single_file`

The `from_single_file` method allows you to load supported Pipelines using a single checkpoint file as opposed to the folder format used by Diffusers. This is useful if you are working with many of the Stable Diffusion Web UI's that extensively rely on a single file to distribute all the components of a Diffusion Model.

The `from_single_file` method also supports loading models in their originally distributed format. This means that supported models that have been finetuned with other services can be loaded directly into supported Diffusers model objects and Pipelines.

## Pipelines that currently support `from_single_file` loading

- [`StableDiffusionPipeline`]
- [`StableDiffusionImg2ImgPipeline`]
- [`StableDiffusionInpaintPipeline`]
- [`StableDiffusionControlNetPipeline`]
- [`StableDiffusionControlNetImg2ImgPipeline`]
- [`StableDiffusionControlNetInpaintPipeline`]
- [`StableDiffusionUpscalePipeline`]
- [`StableDiffusionXLPipeline`]
- [`StableDiffusionXLImg2ImgPipeline`]
- [`StableDiffusionXLInpaintPipeline`]
- [`StableDiffusionXLControlNetPipeline`]

## Models that currently support `from_single_file` loading

- [`UNet2DConditionModel`]
- [`StableCascadeUNet`]
- [`AutoencoderKL`]
- [`ControlNetModel`]

## Usage Examples

## Loading a Pipeline using `from_single_file`

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path)
```

## Setting components in a Pipeline using `from_single_file`

Swap components of the pipeline by passing them directly to the `from_single_file` method. e.g If you would like use a different scheduler than the pipeline default.

```python
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"

scheduler = DDIMScheduler()
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler)

```

```python
from diffusers import StableDiffusionPipeline, ControlNetModel

ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"

controlnet = ControlNetModel.from_pretrained("https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors")
pipe = StableDiffusionPipeline.from_single_file(ckpt_path, controlnet=controlnet)

```

## Loading a Model using `from_single_file`

```python
from diffusers import StableCascadeUNet

ckpt_path = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite.safetensors"
model = StableCascadeUNet.from_single_file(ckpt_path)

```

## Override configuration options when using single file loading

Override the default model or pipeline configuration options when using `from_single_file` by passing in the relevant arguments directly to the `from_single_file` method. Any argument that is supported by the Model or Pipeline class can be configured in this way

```python
from diffusers import StableDiffusionXLImg2ImgPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0_0.9vae.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, requires_aesthetics_score=True)

```

```python
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, device="cuda", upcast_attention=True)

```

## Downloading a single file checkpoint to a specific directory

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, local_dir="my_checkpoints")

```

## Using a Diffusers model repository to configure single file loading

Under the hood, `from_single_file` will try to determine a model repository to use to configure the components of the pipeline. You can also pass in a repository id to the `config` argument of the `from_single_file` method to explicitly set the repository to use.

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
repo_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, config=repo_id)

```

In the example above, since we explicitly passed `repo_id="stabilityai/stable-diffusion-xl-base-1.0"`, it will use this [configuration file](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json) from the "unet" subfolder in `"stabilityai/stable-diffusion-xl-base-1.0"` to configure the unet component included in the checkpoint; Similarly, it will use the `config.json` file from `"vae"` subfolder to configure the vae model, `config.json` file from text_encoder folder to configure text_encoder and so on.

Note that most of the time you do not need to explicitly a `config` argument, `from_single_file` will automatically map the checkpoint to a repo id (we will discuss this in more details in next section). However, this can be useful in cases where model components might have been changed from what was originally distributed or in cases where a checkpoint file might not have the necessary metadata to correctly determine the configuration to use for the pipeline.

<Tip>

To learn more about how to load single file weights, see the [Load different Stable Diffusion formats](../../using-diffusers/other-formats) loading guide.

</Tip>

## Working with local files

As of `diffusers>=0.28.0` the `from_single_file` method will attempt to configure a pipeline or model by first inferring the model type from the checkpoint file and then using the model type to determine the appropriate model repo configuration to use from the Hugging Face Hub. e.g. Any single file checkpoint based on the `StableDiffusionXL` base model will use the `stabilityai/stable-diffusion-xl-base-1.0` model repo to configure the pipeline.

If you are working in an environment with restricted internet access, it is recommended to download the config files and checkpoints for the model to your preferred directory and pass the local paths to the `pretrained_model_link_or_path` and `config` arguments of the `from_single_file` method.

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0_0.9vae.safetensors"
)

my_local_config_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
)

pipe = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)

```

By default this will download the checkpoints and config files to the [Hugging Face Hub cache directory](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache). You can also specify a local directory to download the files to by passing the `local_dir` argument to the `hf_hub_download` and `snapshot_download` functions.

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0_0.9vae.safetensors",
    local_dir="my_local_checkpoints"
)

my_local_config_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
    local_dir="my_local_config"
)

pipe = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)

```

## Working with local files on file systems that do not support symlinking

By default the `from_single_file` method relies on the `huggingface_hub` caching mechanism to fetch and store checkpoints and config files for models and pipelines. If you are working with a file system that does not support symlinking, it is recommended that you first download the checkpoint file to a local directory and disable symlinking by passing the `local_dir_use_symlink=False` argument to the `hf_hub_download` and `snapshot_download` functions.

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0_0.9vae.safetensors"
    local_dir="my_local_checkpoints",
    local_dir_use_symlinks=False
)
print("My local checkpoint: ", my_local_checkpoint_path)

my_local_config_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="my_local_sdxl_config",
    local_dir_use_symlinks=False,
    allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
)
print("My local config: ", my_local_config_path)

```

Then pass the local paths to the `pretrained_model_link_or_path` and `config` arguments of the `from_single_file` method.

```python
pipe = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)

```

<Tip>
Disabling symlinking means that the `huggingface_hub` caching mechanism has no way to determine whether a file has already been downloaded to the local directory. This means that the `hf_hub_download` and `snapshot_download` functions will download the file to the local directory every time they are called. If you are disabling symlinking, it recommended that you separate the download and loading steps to avoid downloading the same file multiple times.

</Tip>

## Using the original configuration file of a model

If you would like to use the original configuration file of a model when loading a model from a single file, you can do so with the `original_config` argument.

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
original_config = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"

pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, original_config=original_config)
```

<Tip>
When using `original_config` with local_files_only=True`, `diffusers` will attempt to infer the components based on the type signatures of pipeline class. This is not as reliable as providing a config path and might lead to errors when configuring the pipeline. Additionally, the pipeline scheduler will default to the `DDIMScheduler` if one isn't provided.

</Tip>


## FromSingleFileMixin

[[autodoc]] loaders.single_file.FromSingleFileMixin

## FromOriginalModelMixin

[[autodoc]] loaders.single_file_model.FromOriginalModelMixin
