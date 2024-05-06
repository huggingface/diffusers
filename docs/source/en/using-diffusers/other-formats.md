<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Model files and formats

[[open-in-colab]]

Diffusion models are saved in various file types and organized in different layouts. Diffusers prefers storing model weights as `.safetensors` files in a multifolder layout, but it also supports loading file types (can be `.safetensors`, `.ckpt`, etc.) from a single file layout which is commonly used in the diffusion ecosystem.

Each layout has its own benefits and use cases, and this guide will show you how to load the different file types and layouts, and how to convert them.

## Files

Model weights are stored in various file types such as `.safetensors`, `.ckpt`, or `.bin`. PyTorch model weights are typically saved with Python's [pickle](https://docs.python.org/3/library/pickle.html) utility. However, pickle is not secure and pickled files may contain malicious code that can be executed. This vulnerability became an increasing concern as model sharing platforms like the Hugging Face Hub gained popularity. Most of the earlier models are saved this way, as `.ckpt` files. To address this security issue, the [Safetensors](https://hf.co/docs/safetensors) library was developed as a secure alternative to pickle, which saves models as `.safetensors` files.

### Safetensors

> [!TIP]
> Learn more about the design decisions and why Safetensors is the preferred file format for saving and loading model weights in the [Safetensors audited as really safe and becoming the default](https://blog.eleuther.ai/safetensors-security-audit/) blog post.

[Safetensors](https://hf.co/docs/safetensors) is a safe and fast file type for securely storing and loading tensors. Safetensors restricts the header size to limit certain types of attacks, supports lazy loading (useful for distributed setups), and generally faster loading speeds.

Make sure you have the [Safetensors](https://hf.co/docs/safetensors) library installed.

```py
!pip install safetensors
```

Safetensor files are stored as `.safetensors`. Diffusers loads `.safetensors` files by default if they're available and the Safetensors library is installed. There are two ways `.safetensors` files can be organized:

1. all the model weights may be saved in a single file (check out the [WarriorMama777/OrangeMixs](https://hf.co/WarriorMama777/OrangeMixs/tree/main/Models/AbyssOrangeMix) repository as an example)
2. there may be several separate `.safetensors` files, one for each pipeline component (text encoder, UNet, VAE), organized in subfolders (check out the [runwayml/stable-diffusion-v1-5](https://hf.co/runwayml/stable-diffusion-v1-5/tree/main) repository as an example)

<hfoptions id="safetensors">
<hfoption id="multifolder">

Use the [`~DiffusionPipeline.from_pretrained`] method to load a model with `.safetensors` files stored in multiple folders.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_safetensors=True
)
```

</hfoption>
<hfoption id="single file">

Use the [`~loaders.FromSingleFileMixin.from_single_file`] method to load a model with all the weights stored in a single `.safetensors` file.

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
)
```

</hfoption>
</hfoptions>

#### LoRA files

[LoRA](https://hf.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a lightweight adapter that is fast and easy to train, making them especially popular for generating images in a certain way or style. These adapters are commonly stored in a `.safetensors` file, and are widely popular on model sharing platforms like [civitai](https://civitai.com/). LoRAs are loaded into a base model with the [`~loaders.LoraLoaderMixin.load_lora_weights`] method.

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

> [!WARNINNG]
> Pickled files may be unsafe because they can be exploited to execute malicious code. It is recommended to use `.safetensors` files instead where possible, or convert the weights to `.safetensors` files.

PyTorch's [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) function uses Python's [pickle](https://docs.python.org/3/library/pickle.html) utility to serialize and save models. You may see these files saved with the `.ckpt` extension and they contain the entire models weights. Use the [`~loaders.FromSingleFileMixin.from_single_file`] method to directly load a `.ckpt` file.

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
)
```

## Storage layout

There are two ways model files are organized, either in multiple folders or in a single file. Diffusers uses the multifolder layout by default where each component file (text encoder, UNet, VAE) is stored in a separate subfolder. Diffusers also supports loading models from a single file where all the components are bundled together.

### Multifolder

The multifolder layout is the default storage layout for Diffusers. Each component's (text encoder, UNet, VAE) weights are stored in a separate subfolder. The weights can be stored as any file type such as `.safetensors` or `.ckpt`.

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

To load from a multifolder layout, use the [`~DiffusionPipeline.from_pretrained`] method.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")
```

Benefits of using the multifolder layout include:

1. Faster to load each component file individually or in parallel.
2. Easier to update individual components. For example, once you've downloaded and cached a model, Diffusers only downloads and updates components that have been changed. This saves storage because you don't have to download the entire model again if you only need one component of the model.
3. Replace a component in the model with a newer or better version.

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

4. More visibility and information about a model's components which are stored in a [config.json](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json) file in each component subfolder.

### Single file

The single file layout stores all the model weights in a single file. All the model components (text encoder, UNet, VAE) weights are kept together instead of separately in subfolders. This single file can be any file type such as `.safetensors` or `.ckpt`.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/single-file-layout.png"/>
</div>

To load from a single file layout, use the [`~loaders.FromSingleFileMixin.from_single_file`] method.

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

Benefits of using a single file layout include:

1. Easy compatibility with diffusion interfaces such as [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which commonly use a single file layout.
2. Easier to manage (download and share) a single file.

## Convert layout and files

Diffusers provides many scripts and methods to convert storage layouts and file types to enable broader support across the diffusion ecosystem.

Take a look at the [diffusers/scripts](https://github.com/huggingface/diffusers/tree/main/scripts) collection to find a script that fits your conversion needs.

> [!TIP]
> Scripts that have `to_diffusers` appended at the end mean they convert a model to the multifolder format used by Diffusers. Each script has their own specific set of arguments to configure the conversion, so make sure you check what type of arguments are available!

For example, to convert a Stable Diffusion XL model stored in Diffusers multifolder format to a single file type, run the [convert_diffusers_to_original_sdxl.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) script. Provide the path to the model to convert, and the path to save the converted model to. You can optionally specify whether you want to save the model as a `.safetensors` file and whether to save the model in half-precision.

```bash
python convert_diffusers_to_original_sdxl.py --model_path path/to/model/to/convert --checkpoint_path path/to/save/model/to --use_safetensors
```

You can also save a model to the multifolder format with the [`~DiffusionPipeline.save_pretrained`] method. This creates a directory for you if it doesn't already exist, and it also saves the files as a `.safetensors` file by default.

```py
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
)
pipeline.save_pretrained()
```

Lastly, there are also Spaces, such as [SD To Diffusers](https://hf.co/spaces/diffusers/sd-to-diffusers) and [SD-XL To Diffusers](https://hf.co/spaces/diffusers/sdxl-to-diffusers), that provide a more user-friendly interface for converting models to the multifolder format. This is the easiest and most convenient option for converting layouts, and it'll open a PR on your model repository with the converted files. However, this option is not as reliable as running a script, and the Space may fail for more complicated models.
