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

1. Diffusers-multifolder layout: there may be several separate safetensors files, one for each pipeline component (text encoder, UNet, VAE), organized in subfolders (check out the [runwayml/stable-diffusion-v1-5](https://hf.co/runwayml/stable-diffusion-v1-5/tree/main) repository as an example)
2. single-file layout: all the model weights may be saved in a single file (check out the [WarriorMama777/OrangeMixs](https://hf.co/WarriorMama777/OrangeMixs/tree/main/Models/AbyssOrangeMix) repository as an example)

<hfoptions id="safetensors">
<hfoption id="multifolder">

Use the [`~DiffusionPipeline.from_pretrained`] method to load a model with safetensors files stored in multiple folders.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
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

LoRAs are loaded into a base model with the [`~loaders.LoraLoaderMixin.load_lora_weights`] method.

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
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
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
