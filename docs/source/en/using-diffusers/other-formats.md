<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Model files and layouts

[[open-in-colab]]

Diffusion models are saved in various file types and organized in different layouts. Diffusers stores model weights as safetensors files in *Diffusers-multifolder* layout and it also supports loading files (like safetensors and ckpt files) from a *single-file* layout which is commonly used in the diffusion ecosystem.

Each layout has its own benefits and use cases, and this guide will show you how to load the different files and layouts, and how to convert them.

## Files

PyTorch model weights are typically saved with Python's [pickle](https://docs.python.org/3/library/pickle.html) utility as ckpt or bin files. However, pickle is not secure and pickled files may contain malicious code that can be executed. This vulnerability is a serious concern given the popularity of model sharing. To address this security issue, the [Safetensors](https://hf.co/docs/safetensors) library was developed as a secure alternative to pickle, which saves models as safetensors files.

### safetensors

> [!TIP]
> Learn more about the design decisions and why safetensor files are preferred for saving and loading model weights in the [Safetensors audited as really safe and becoming the default](https://blog.eleuther.ai/safetensors-security-audit/) blog post.

[Safetensors](https://hf.co/docs/safetensors) is a safe and fast file format for securely storing and loading tensors. Safetensors restricts the header size to limit certain types of attacks, supports lazy loading (useful for distributed setups), and has generally faster loading speeds.

Make sure you have the [Safetensors](https://hf.co/docs/safetensors) library installed.

```py
!pip install safetensors
```

Safetensors stores weights in a safetensors file. Diffusers loads safetensors files by default if they're available and the Safetensors library is installed. There are two ways safetensors files can be organized:

1. Diffusers-multifolder layout: there may be several separate safetensors files, one for each pipeline component (text encoder, UNet, VAE), organized in subfolders (check out the [stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) repository as an example)
2. single-file layout: all the model weights may be saved in a single file (check out the [WarriorMama777/OrangeMixs](https://hf.co/WarriorMama777/OrangeMixs/tree/main/Models/AbyssOrangeMix) repository as an example)

<hfoptions id="safetensors">
<hfoption id="multifolder">

Use the [`~DiffusionPipeline.from_pretrained`] method to load a model with safetensors files stored in multiple folders.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    use_safetensors=True
)
```

</hfoption>
<hfoption id="single file">

Use the [`~loaders.FromSingleFileMixin.from_single_file`] method to load a model with all the weights stored in a single safetensors file.

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
)
```

</hfoption>
</hfoptions>

#### LoRA files

[LoRA](https://hf.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a lightweight adapter that is fast and easy to train, making them especially popular for generating images in a certain way or style. These adapters are commonly stored in a safetensors file, and are widely popular on model sharing platforms like [civitai](https://civitai.com/).

LoRAs are loaded into a base model with the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method.

```py
from diffusers import StableDiffusionXLPipeline
import torch

# base model
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "Lykon/dreamshaper-xl-1-0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

# download LoRA weights
!wget https://civitai.com/api/download/models/168776 -O blueprintify.safetensors

# load LoRA weights
pipeline.load_lora_weights(".", weight_name="blueprintify.safetensors")
prompt = "bl3uprint, a highly detailed blueprint of the empire state building, explaining how to build all parts, many txt, blueprint grid backdrop"
negative_prompt = "lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(0),
).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/blueprint-lora.png"/>
</div>

### ckpt

> [!WARNING]
> Pickled files may be unsafe because they can be exploited to execute malicious code. It is recommended to use safetensors files instead where possible, or convert the weights to safetensors files.

PyTorch's [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) function uses Python's [pickle](https://docs.python.org/3/library/pickle.html) utility to serialize and save models. These files are saved as a ckpt file and they contain the entire model's weights.

Use the [`~loaders.FromSingleFileMixin.from_single_file`] method to directly load a ckpt file.

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
)
```

## Storage layout

There are two ways model files are organized, either in a Diffusers-multifolder layout or in a single-file layout. The Diffusers-multifolder layout is the default, and each component file (text encoder, UNet, VAE) is stored in a separate subfolder. Diffusers also supports loading models from a single-file layout where all the components are bundled together.

### Diffusers-multifolder

The Diffusers-multifolder layout is the default storage layout for Diffusers. Each component's (text encoder, UNet, VAE) weights are stored in a separate subfolder. The weights can be stored as safetensors or ckpt files.

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multifolder-layout.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">multifolder layout</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multifolder-unet.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">UNet subfolder</figcaption>
  </div>
</div>

To load from Diffusers-multifolder layout, use the [`~DiffusionPipeline.from_pretrained`] method.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")
```

Benefits of using the Diffusers-multifolder layout include:

1. Faster to load each component file individually or in parallel.
2. Reduced memory usage because you only load the components you need. For example, models like [SDXL Turbo](https://hf.co/stabilityai/sdxl-turbo), [SDXL Lightning](https://hf.co/ByteDance/SDXL-Lightning), and [Hyper-SD](https://hf.co/ByteDance/Hyper-SD) have the same components except for the UNet. You can reuse their shared components with the [`~DiffusionPipeline.from_pipe`] method without consuming any additional memory (take a look at the [Reuse a pipeline](./loading#reuse-a-pipeline) guide) and only load the UNet. This way, you don't need to download redundant components and unnecessarily use more memory.

    ```py
    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler

    # download one model
    sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    # switch UNet for another model
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/sdxl-turbo",
        subfolder="unet",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    # reuse all the same components in new model except for the UNet
    turbo_pipeline = StableDiffusionXLPipeline.from_pipe(
        sdxl_pipeline, unet=unet,
    ).to("cuda")
    turbo_pipeline.scheduler = EulerDiscreteScheduler.from_config(
        turbo_pipeline.scheduler.config,
        timestep+spacing="trailing"
    )
    image = turbo_pipeline(
        "an astronaut riding a unicorn on mars",
        num_inference_steps=1,
        guidance_scale=0.0,
    ).images[0]
    image
    ```

3. Reduced storage requirements because if a component, such as the SDXL [VAE](https://hf.co/madebyollin/sdxl-vae-fp16-fix), is shared across multiple models, you only need to download and store a single copy of it instead of downloading and storing it multiple times. For 10 SDXL models, this can save ~3.5GB of storage. The storage savings is even greater for newer models like PixArt Sigma, where the [text encoder](https://hf.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/main/text_encoder) alone is ~19GB!
4. Flexibility to replace a component in the model with a newer or better version.

    ```py
    from diffusers import DiffusionPipeline, AutoencoderKL

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    ```

5. More visibility and information about a model's components, which are stored in a [config.json](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json) file in each component subfolder.

### Single-file

The single-file layout stores all the model weights in a single file. All the model components (text encoder, UNet, VAE) weights are kept together instead of separately in subfolders. This can be a safetensors or ckpt file.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/single-file-layout.png"/>
</div>

To load from a single-file layout, use the [`~loaders.FromSingleFileMixin.from_single_file`] method.

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")
```

Benefits of using a single-file layout include:

1. Easy compatibility with diffusion interfaces such as [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which commonly use a single-file layout.
2. Easier to manage (download and share) a single file.

## Convert layout and files

Diffusers provides many scripts and methods to convert storage layouts and file formats to enable broader support across the diffusion ecosystem.

Take a look at the [diffusers/scripts](https://github.com/huggingface/diffusers/tree/main/scripts) collection to find a script that fits your conversion needs.

> [!TIP]
> Scripts that have "`to_diffusers`" appended at the end mean they convert a model to the Diffusers-multifolder layout. Each script has their own specific set of arguments for configuring the conversion, so make sure you check what arguments are available!

For example, to convert a Stable Diffusion XL model stored in Diffusers-multifolder layout to a single-file layout, run the [convert_diffusers_to_original_sdxl.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) script. Provide the path to the model to convert, and the path to save the converted model to. You can optionally specify whether you want to save the model as a safetensors file and whether to save the model in half-precision.

```bash
python convert_diffusers_to_original_sdxl.py --model_path path/to/model/to/convert --checkpoint_path path/to/save/model/to --use_safetensors
```

You can also save a model to Diffusers-multifolder layout with the [`~DiffusionPipeline.save_pretrained`] method. This creates a directory for you if it doesn't already exist, and it also saves the files as a safetensors file by default.

```py
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
)
pipeline.save_pretrained()
```

Lastly, there are also Spaces, such as [SD To Diffusers](https://hf.co/spaces/diffusers/sd-to-diffusers) and [SD-XL To Diffusers](https://hf.co/spaces/diffusers/sdxl-to-diffusers), that provide a more user-friendly interface for converting models to Diffusers-multifolder layout. This is the easiest and most convenient option for converting layouts, and it'll open a PR on your model repository with the converted files. However, this option is not as reliable as running a script, and the Space may fail for more complicated models.

## Single-file layout usage

Now that you're familiar with the differences between the Diffusers-multifolder and single-file layout, this section shows you how to load models and pipeline components, customize configuration options for loading, and load local files with the [`~loaders.FromSingleFileMixin.from_single_file`] method.

### Load a pipeline or model

Pass the file path of the pipeline or model to the [`~loaders.FromSingleFileMixin.from_single_file`] method to load it.

<hfoptions id="pipeline-model">
<hfoption id="pipeline">

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path)
```

</hfoption>
<hfoption id="model">

```py
from diffusers import StableCascadeUNet

ckpt_path = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite.safetensors"
model = StableCascadeUNet.from_single_file(ckpt_path)
```

</hfoption>
</hfoptions>

Customize components in the pipeline by passing them directly to the [`~loaders.FromSingleFileMixin.from_single_file`] method. For example, you can use a different scheduler in a pipeline.

```py
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
scheduler = DDIMScheduler()
pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler)
```

Or you could use a ControlNet model in the pipeline.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

ckpt_path = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipeline = StableDiffusionControlNetPipeline.from_single_file(ckpt_path, controlnet=controlnet)
```

### Customize configuration options

Models have a configuration file that define their attributes like the number of inputs in a UNet. Pipelines configuration options are available in the pipeline's class. For example, if you look at the [`StableDiffusionXLInstructPix2PixPipeline`] class, there is an option to scale the image latents with the `is_cosxl_edit` parameter.

These configuration files can be found in the models Hub repository or another location from which the configuration file originated (for example, a GitHub repository or locally on your device).

<hfoptions id="config-file">
<hfoption id="Hub configuration file">

> [!TIP]
> The [`~loaders.FromSingleFileMixin.from_single_file`] method automatically maps the checkpoint to the appropriate model repository, but there are cases where it is useful to use the `config` parameter. For example, if the model components in the checkpoint are different from the original checkpoint or if a checkpoint doesn't have the necessary metadata to correctly determine the configuration to use for the pipeline.

The [`~loaders.FromSingleFileMixin.from_single_file`] method automatically determines the configuration to use from the configuration file in the model repository. You could also explicitly specify the configuration to use by providing the repository id to the `config` parameter.

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/segmind/SSD-1B/blob/main/SSD-1B.safetensors"
repo_id = "segmind/SSD-1B"

pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, config=repo_id)
```

The model loads the configuration file for the [UNet](https://huggingface.co/segmind/SSD-1B/blob/main/unet/config.json), [VAE](https://huggingface.co/segmind/SSD-1B/blob/main/vae/config.json), and [text encoder](https://huggingface.co/segmind/SSD-1B/blob/main/text_encoder/config.json) from their respective subfolders in the repository.

</hfoption>
<hfoption id="original configuration file">

The [`~loaders.FromSingleFileMixin.from_single_file`] method can also load the original configuration file of a pipeline that is stored elsewhere. Pass a local path or URL of the original configuration file to the `original_config` parameter.

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
original_config = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"

pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, original_config=original_config)
```

> [!TIP]
> Diffusers attempts to infer the pipeline components based on the type signatures of the pipeline class when you use `original_config` with `local_files_only=True`, instead of fetching the configuration files from the model repository on the Hub. This prevents backward breaking changes in code that can't connect to the internet to fetch the necessary configuration files.
>
> This is not as reliable as providing a path to a local model repository with the `config` parameter, and might lead to errors during pipeline configuration. To avoid errors, run the pipeline with `local_files_only=False` once to download the appropriate pipeline configuration files to the local cache.

</hfoption>
</hfoptions>

While the configuration files specify the pipeline or models default parameters, you can override them by providing the parameters directly to the [`~loaders.FromSingleFileMixin.from_single_file`] method. Any parameter supported by the model or pipeline class can be configured in this way.

<hfoptions id="override">
<hfoption id="pipeline">

For example, to scale the image latents in [`StableDiffusionXLInstructPix2PixPipeline`] pass the `is_cosxl_edit` parameter.

```python
from diffusers import StableDiffusionXLInstructPix2PixPipeline

ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
pipeline = StableDiffusionXLInstructPix2PixPipeline.from_single_file(ckpt_path, config="diffusers/sdxl-instructpix2pix-768", is_cosxl_edit=True)
```

</hfoption>
<hfoption id="model">

For example, to upcast the attention dimensions in a [`UNet2DConditionModel`] pass the `upcast_attention` parameter.

```python
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, upcast_attention=True)
```

</hfoption>
</hfoptions>

### Local files

In Diffusers>=v0.28.0, the [`~loaders.FromSingleFileMixin.from_single_file`] method attempts to configure a pipeline or model by inferring the model type from the keys in the checkpoint file. The inferred model type is used to determine the appropriate model repository on the Hugging Face Hub to configure the model or pipeline.

For example, any single file checkpoint based on the Stable Diffusion XL base model will use the [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model repository to configure the pipeline.

But if you're working in an environment with restricted internet access, you should download the configuration files with the [`~huggingface_hub.snapshot_download`] function, and the model checkpoint with the [`~huggingface_hub.hf_hub_download`] function. By default, these files are downloaded to the Hugging Face Hub [cache directory](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache), but you can specify a preferred directory to download the files to with the `local_dir` parameter.

Pass the configuration and checkpoint paths to the [`~loaders.FromSingleFileMixin.from_single_file`] method to load locally.

<hfoptions id="local">
<hfoption id="Hub cache directory">

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allow_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
)

pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```

</hfoption>
<hfoption id="specific local directory">

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
    local_dir="my_local_checkpoints"
)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allow_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
    local_dir="my_local_config"
)

pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```

</hfoption>
</hfoptions>

#### Local files without symlink

> [!TIP]
> In huggingface_hub>=v0.23.0, the `local_dir_use_symlinks` argument isn't necessary for the [`~huggingface_hub.hf_hub_download`] and [`~huggingface_hub.snapshot_download`] functions.

The [`~loaders.FromSingleFileMixin.from_single_file`] method relies on the [huggingface_hub](https://hf.co/docs/huggingface_hub/index) caching mechanism to fetch and store checkpoints and configuration files for models and pipelines. If you're working with a file system that does not support symlinking, you should download the checkpoint file to a local directory first, and disable symlinking with the `local_dir_use_symlink=False` parameter in the [`~huggingface_hub.hf_hub_download`] function and [`~huggingface_hub.snapshot_download`] functions.

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
    local_dir="my_local_checkpoints",
    local_dir_use_symlinks=False
)
print("My local checkpoint: ", my_local_checkpoint_path)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allow_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
    local_dir_use_symlinks=False,
)
print("My local config: ", my_local_config_path)
```

Then you can pass the local paths to the `pretrained_model_link_or_path` and `config` parameters.

```python
pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```
