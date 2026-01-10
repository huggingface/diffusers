<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoRA

[LoRA (Low-Rank Adaptation)](https://huggingface.co/papers/2106.09685) is a method for quickly training a model for a new task. It works by freezing the original model weights and adding a small number of *new* trainable parameters. This means it is significantly faster and cheaper to adapt an existing model to new tasks, such as generating images in a new style.

LoRA checkpoints are typically only a couple hundred MBs in size, so they're very lightweight and easy to store. Load these smaller set of weights into an existing base model with [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] and specify the file name.

<hfoptions id="usage">
<hfoption id="text-to-image">

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora",
    weight_name="cereal_box_sdxl_v1.safetensors",
    adapter_name="cereal"
)
pipeline("bears, pizza bites").images[0]
```

</hfoption>
<hfoption id="text-to-video">

```py
import torch
from diffusers import LTXConditionPipeline
from diffusers.utils import export_to_video, load_image

pipeline = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16
)

pipeline.load_lora_weights(
    "Lightricks/LTX-Video-Cakeify-LoRA",
    weight_name="ltxv_095_cakeify_lora.safetensors",
    adapter_name="cakeify"
)
pipeline.set_adapters("cakeify")

# use "CAKEIFY" to trigger the LoRA
prompt = "CAKEIFY a person using a knife to cut a cake shaped like a Pikachu plushie"
image = load_image("https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA/resolve/main/assets/images/pikachu.png")

video = pipeline(
    prompt=prompt,
    image=image,
    width=576,
    height=576,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=26)
```

</hfoption>
</hfoptions>

The [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] method is the preferred way to load LoRA weights into the UNet and text encoder because it can handle cases where:

- the LoRA weights don't have separate UNet and text encoder identifiers
- the LoRA weights have separate UNet and text encoder identifiers

The [`~loaders.PeftAdapterMixin.load_lora_adapter`] method is used to directly load a LoRA adapter at the *model-level*, as long as the model is a Diffusers model that is a subclass of [`PeftAdapterMixin`]. It builds and prepares the necessary model configuration for the adapter. This method also loads the LoRA adapter into the UNet.

For example, if you're only loading a LoRA into the UNet, [`~loaders.PeftAdapterMixin.load_lora_adapter`] ignores the text encoder keys. Use the `prefix` parameter to filter and load the appropriate state dicts, `"unet"` to load.

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.unet.load_lora_adapter(
    "jbilcke-hf/sdxl-cinematic-1",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="cinematic",
    prefix="unet"
)
# use cnmt in the prompt to trigger the LoRA
pipeline("A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration").images[0]
```

## torch.compile

[torch.compile](../optimization/fp16#torchcompile) speeds up inference by compiling the PyTorch model to use optimized kernels. Before compiling, the LoRA weights need to be fused into the base model and unloaded first.

```py
import torch
from diffusers import DiffusionPipeline

# load base model and LoRA
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)

# activate LoRA and set adapter weight
pipeline.set_adapters("ikea", adapter_weights=0.7)

# fuse LoRAs and unload weights
pipeline.fuse_lora(adapter_names=["ikea"], lora_scale=1.0)
pipeline.unload_lora_weights()
```

Typically, the UNet is compiled because its the most compute intensive component of the pipeline.

```py
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

pipeline("A bowl of ramen shaped like a cute kawaii bear").images[0]
```

Refer to the [hotswapping](#hotswapping) section to learn how to avoid recompilation when working with compiled models and multiple LoRAs.

## Weight scale

The `scale` parameter is used to control how much of a LoRA to apply. A value of `0` is equivalent to only using the base model weights and a value of `1` is equivalent to fully using the LoRA.

<hfoptions id="weight-scale">
<hfoption id="simple use case">

For simple use cases, you can pass `cross_attention_kwargs={"scale": 1.0}` to the pipeline.

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora",
    weight_name="cereal_box_sdxl_v1.safetensors",
    adapter_name="cereal"
)
pipeline("bears, pizza bites", cross_attention_kwargs={"scale": 1.0}).images[0]
```

</hfoption>
<hfoption id="finer control">

> [!WARNING]
> The [`~loaders.PeftAdapterMixin.set_adapters`] method only scales attention weights. If a LoRA has ResNets or down and upsamplers, these components keep a scale value of `1.0`.

For finer control over each individual component of the UNet or text encoder, pass a dictionary instead. In the example below, the `"down"` block in the UNet is scaled by 0.9 and you can further specify in the `"up"` block the scales of the transformers in `"block_0"` and `"block_1"`. If a block like `"mid"` isn't specified, the default value `1.0` is used.

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora",
    weight_name="cereal_box_sdxl_v1.safetensors",
    adapter_name="cereal"
)
scales = {
    "text_encoder": 0.5,
    "text_encoder_2": 0.5,
    "unet": {
        "down": 0.9,
        "up": {
            "block_0": 0.6,
            "block_1": [0.4, 0.8, 1.0],
        }
    }
}
pipeline.set_adapters("cereal", scales)
pipeline("bears, pizza bites").images[0]
```

</hfoption>
</hfoptions>

### Scale scheduling

Dynamically adjusting the LoRA scale during sampling gives you better control over the overall composition and layout because certain steps may benefit more from an increased or reduced scale.

The [character LoRA](https://huggingface.co/alvarobartt/ghibli-characters-flux-lora) in the example below starts with a higher scale that gradually decays over the first 20 steps to establish the character generation. In the later steps, only a scale of 0.2 is applied to avoid adding too much of the LoRA features to other parts of the image the LoRA wasn't trained on.

```py
import torch
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

pipelne.load_lora_weights("alvarobartt/ghibli-characters-flux-lora", "lora")

num_inference_steps = 30
lora_steps = 20
lora_scales = torch.linspace(1.5, 0.7, lora_steps).tolist()
lora_scales += [0.2] * (num_inference_steps - lora_steps + 1)

pipeline.set_adapters("lora", lora_scales[0])

def callback(pipeline: FluxPipeline, step: int, timestep: torch.LongTensor, callback_kwargs: dict):
    pipeline.set_adapters("lora", lora_scales[step + 1])
    return callback_kwargs

prompt = """
Ghibli style The Grinch, a mischievous green creature with a sly grin, peeking out from behind a snow-covered tree while plotting his antics, 
in a quaint snowy village decorated for the holidays, warm light glowing from cozy homes, with playful snowflakes dancing in the air
"""
pipeline(
    prompt=prompt,
    guidance_scale=3.0,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator().manual_seed(42),
    callback_on_step_end=callback,
).images[0]
```

## Hotswapping

Hotswapping LoRAs is an efficient way to work with multiple LoRAs while avoiding accumulating memory from multiple calls to [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] and in some cases, recompilation, if a model is compiled. This workflow requires a loaded LoRA because the new LoRA weights are swapped in place for the existing loaded LoRA.

```py
import torch
from diffusers import DiffusionPipeline

# load base model and LoRAs
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
```

> [!WARNING]
> Hotswapping is unsupported for LoRAs that target the text encoder.

Set `hotswap=True` in [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] to swap the second LoRA. Use the `adapter_name` parameter to indicate which LoRA to swap (`default_0` is the default name).

```py
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    hotswap=True,
    adapter_name="ikea"
)
```

### Compiled models

For compiled models, use [`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`] to avoid recompilation when hotswapping LoRAs. This method should be called *before* loading the first LoRA and `torch.compile` should be called *after* loading the first LoRA.

> [!TIP]
> The [`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`] method isn't always necessary if the second LoRA targets the identical LoRA ranks and scales as the first LoRA.

Within [`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`], the `target_rank` parameter is important for setting the rank for all LoRA adapters. Setting it to `max_rank` sets it to the highest value. For LoRAs with different ranks, you set it to a higher rank value. The default rank value is 128.

```py
import torch
from diffusers import DiffusionPipeline

# load base model and LoRAs
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
# 1. enable_lora_hotswap
pipeline.enable_lora_hotswap(target_rank=max_rank)
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
# 2. torch.compile
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# 3. hotswap
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    hotswap=True,
    adapter_name="ikea"
)
```

> [!TIP]
> Move your code inside the `with torch._dynamo.config.patch(error_on_recompile=True)` context manager to detect if a model was recompiled. If a model is recompiled despite following all the steps above, please open an [issue](https://github.com/huggingface/diffusers/issues) with a reproducible example.

If you expect to varied resolutions during inference with this feature, then make sure set `dynamic=True` during compilation. Refer to [this document](../optimization/fp16#dynamic-shape-compilation) for more details.

There are still scenarios where recompulation is unavoidable, such as when the hotswapped LoRA targets more layers than the initial adapter. Try to load the LoRA that targets the most layers *first*. For more details about this limitation, refer to the PEFT [hotswapping](https://huggingface.co/docs/peft/main/en/package_reference/hotswap#peft.utils.hotswap.hotswap_adapter) docs.

<details>
<summary>Technical details of hotswapping</summary>

The [`~loaders.lora_base.LoraBaseMixin.enable_lora_hotswap`] method converts the LoRA scaling factor from floats to torch.tensors and pads the shape of the weights to the largest required shape to avoid reassigning the whole attribute when the data in the weights are replaced.

This is why the `max_rank` argument is important. The results are unchanged even when the values are padded with zeros. Computation may be slower though depending on the padding size.

Since no new LoRA attributes are added, each subsequent LoRA is only allowed to target the same layers, or subset of layers, the first LoRA targets. Choosing the LoRA loading order is important because if the LoRAs target disjoint layers, you may end up creating a dummy LoRA that targets the union of all target layers.

For more implementation details, take a look at the [`hotswap.py`](https://github.com/huggingface/peft/blob/92d65cafa51c829484ad3d95cf71d09de57ff066/src/peft/utils/hotswap.py) file.

</details>

## Merge

The weights from each LoRA can be merged together to produce a blend of multiple existing styles. There are several methods for merging LoRAs, each of which differ in *how* the weights are merged (may affect generation quality).

### set_adapters

The [`~loaders.PeftAdapterMixin.set_adapters`] method merges LoRAs by concatenating their weighted matrices. Pass the LoRA names to [`~loaders.PeftAdapterMixin.set_adapters`] and use the `adapter_weights` parameter to control the scaling of each LoRA. For example, if `adapter_weights=[0.5, 0.5]`, the output is an average of both LoRAs.

> [!TIP]
> The `"scale"` parameter determines how much of the merged LoRA to apply. See the [Weight scale](#weight-scale) section for more details.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
# use by Feng Zikai to activate the lordjia/by-feng-zikai LoRA
pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", cross_attention_kwargs={"scale": 1.0}).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lora_merge_set_adapters.png"/>
</div>

### add_weighted_adapter

> [!TIP]
> This is an experimental method and you can refer to PEFTs [Model merging](https://huggingface.co/docs/peft/developer_guides/model_merging) for more details. Take a look at this [issue](https://github.com/huggingface/diffusers/issues/6892) if you're interested in the motivation and design behind this integration.

The [`~peft.LoraModel.add_weighted_adapter`] method enables more efficient merging methods like [TIES](https://huggingface.co/papers/2306.01708) or [DARE](https://huggingface.co/papers/2311.03099). These merging methods remove redundant and potentially interfering parameters from merged models. Keep in mind the LoRA ranks need to have identical ranks to be merged.

Make sure the latest stable version of Diffusers and PEFT is installed.

```bash
pip install -U -q diffusers peft
```

Load a UNET that corresponds to the LoRA UNet.

```py
import copy
import torch
from diffusers import AutoModel, DiffusionPipeline
from peft import get_peft_model, LoraConfig, PeftModel

unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")
```

Load a pipeline, pass the UNet to it, and load a LoRA.

```py
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    unet=unet
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
```

Create a [`~peft.PeftModel`] from the LoRA checkpoint by combining the first UNet you loaded and the LoRA UNet from the pipeline.

```py
sdxl_unet = copy.deepcopy(unet)
ikea_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["ikea"],
    adapter_name="ikea"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipeline.unet.state_dict().items()}
ikea_peft_model.load_state_dict(original_state_dict, strict=True)
```

> [!TIP]
> You can save and reuse the `ikea_peft_model` by pushing it to the Hub as shown below.
> ```py
> ikea_peft_model.push_to_hub("ikea_peft_model", token=TOKEN)
> ```

Repeat this process and create a [`~peft.PeftModel`] for the second LoRA.

```py
pipeline.delete_adapters("ikea")
sdxl_unet.delete_adapters("ikea")

pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
pipeline.set_adapters(adapter_names="feng")

feng_peft_model = get_peft_model(
    sdxl_unet,
    pipeline.unet.peft_config["feng"],
    adapter_name="feng"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}
feng_peft_model.load_state_dict(original_state_dict, strict=True)
```

Load a base UNet model and load the adapters.

```py
base_unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    subfolder="unet",
).to("cuda")

model = PeftModel.from_pretrained(
    base_unet,
    "stevhliu/ikea_peft_model",
    use_safetensors=True,
    subfolder="ikea",
    adapter_name="ikea"
)
model.load_adapter(
    "stevhliu/feng_peft_model",
    use_safetensors=True,
    subfolder="feng",
    adapter_name="feng"
)
```

Merge the LoRAs with [`~peft.LoraModel.add_weighted_adapter`] and specify how you want to merge them with `combination_type`. The example below uses the `"dare_linear"` method (refer to this [blog post](https://huggingface.co/blog/peft_merging) to learn more about these merging methods), which randomly prunes some weights and then performs a weighted sum of the tensors based on the set weightage of each LoRA in `weights`.

Activate the merged LoRAs with [`~loaders.PeftAdapterMixin.set_adapters`].

```py
model.add_weighted_adapter(
    adapters=["ikea", "feng"],
    combination_type="dare_linear",
    weights=[1.0, 1.0],
    adapter_name="ikea-feng"
)
model.set_adapters("ikea-feng")

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=model,
    variant="fp16",
    torch_dtype=torch.float16,
).to("cuda")
pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai").images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ikea-feng-dare-linear.png"/>
</div>

### fuse_lora

The [`~loaders.lora_base.LoraBaseMixin.fuse_lora`] method fuses the LoRA weights directly with the original UNet and text encoder weights of the underlying model. This reduces the overhead of loading the underlying model for each LoRA because it only loads the model once, which lowers memory usage and increases inference speed.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
```

Call [`~loaders.lora_base.LoraBaseMixin.fuse_lora`] to fuse them. The `lora_scale` parameter controls how much to scale the output by with the LoRA weights. It is important to make this adjustment now because passing `scale` to `cross_attention_kwargs` won't work in the pipeline.

```py
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
```

Unload the LoRA weights since they're already fused with the underlying model. Save the fused pipeline with either [`~DiffusionPipeline.save_pretrained`] to save it locally or [`~PushToHubMixin.push_to_hub`] to save it to the Hub.

<hfoptions id="save">
<hfoption id="save locally">

```py
pipeline.unload_lora_weights()
pipeline.save_pretrained("path/to/fused-pipeline")
```

</hfoption>
<hfoption id="save to Hub">

```py
pipeline.unload_lora_weights()
pipeline.push_to_hub("fused-ikea-feng")
```

</hfoption>
</hfoptions>

The fused pipeline can now be quickly loaded for inference without requiring each LoRA to be separately loaded.

```py
pipeline = DiffusionPipeline.from_pretrained(
    "username/fused-ikea-feng", torch_dtype=torch.float16,
).to("cuda")
pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai").images[0]
```

Use [`~loaders.LoraLoaderMixin.unfuse_lora`] to restore the underlying models weights, for example, if you want to use a different `lora_scale` value. You can only unfuse if there is a single LoRA fused. For example, it won't work with the pipeline from above because there are multiple fused LoRAs. In these cases, you'll need to reload the entire model.

```py
pipeline.unfuse_lora()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/fuse_lora.png"/>
</div>

## Manage

Diffusers provides several methods to help you manage working with LoRAs. These methods can be especially useful if you're working with multiple LoRAs.

### set_adapters

[`~loaders.PeftAdapterMixin.set_adapters`] also activates the current LoRA to use if there are multiple active LoRAs. This allows you to switch between different LoRAs by specifying their name.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea"
)
pipeline.load_lora_weights(
    "lordjia/by-feng-zikai",
    weight_name="fengzikai_v1.0_XL.safetensors",
    adapter_name="feng"
)
# activates the feng LoRA instead of the ikea LoRA
pipeline.set_adapters("feng")
```

### save_lora_adapter

Save an adapter with [`~loaders.PeftAdapterMixin.save_lora_adapter`].

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipeline.unet.load_lora_adapter(
    "jbilcke-hf/sdxl-cinematic-1",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="cinematic"
    prefix="unet"
)
pipeline.save_lora_adapter("path/to/save", adapter_name="cinematic")
```

### unload_lora_weights

The [`~loaders.lora_base.LoraBaseMixin.unload_lora_weights`] method unloads any LoRA weights in the pipeline to restore the underlying model weights.

```py
pipeline.unload_lora_weights()
```

### disable_lora

The [`~loaders.PeftAdapterMixin.disable_lora`] method disables all LoRAs (but they're still kept on the pipeline) and restores the pipeline to the underlying model weights.

```py
pipeline.disable_lora()
```

### get_active_adapters

The [`~loaders.lora_base.LoraBaseMixin.get_active_adapters`] method returns a list of active LoRAs attached to a pipeline.

```py
pipeline.get_active_adapters()
["cereal", "ikea"]
```

### get_list_adapters

The [`~loaders.lora_base.LoraBaseMixin.get_list_adapters`] method returns the active LoRAs for each component in the pipeline.

```py
pipeline.get_list_adapters()
{"unet": ["cereal", "ikea"], "text_encoder_2": ["cereal"]}
```

### delete_adapters

The [`~loaders.PeftAdapterMixin.delete_adapters`] method completely removes a LoRA and its layers from a model.

```py
pipeline.delete_adapters("ikea")
```

## Resources

Browse the [LoRA Studio](https://lorastudio.co/models) for different LoRAs to use or you can upload your favorite LoRAs from Civitai to the Hub with the Space below.

<iframe
	src="https://multimodalart-civitai-to-hf.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

You can find additional LoRAs in the [FLUX LoRA the Explorer](https://huggingface.co/spaces/multimodalart/flux-lora-the-explorer) and [LoRA the Explorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer) Spaces.

Check out the [Fast LoRA inference for Flux with Diffusers and PEFT](https://huggingface.co/blog/lora-fast) blog post to learn how to optimize LoRA inference with methods like FlashAttention-3 and fp8 quantization.
