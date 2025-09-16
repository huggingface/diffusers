<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reduce memory usage

Modern diffusion models like [Flux](../api/pipelines/flux) and [Wan](../api/pipelines/wan) have billions of parameters that take up a lot of memory on your hardware for inference. This is challenging because common GPUs often don't have sufficient memory. To overcome the memory limitations, you can use more than one GPU (if available), offload some of the pipeline components to the CPU, and more.

This guide will show you how to reduce your memory usage. 

> [!TIP]
> Keep in mind these techniques may need to be adjusted depending on the model. For example, a transformer-based diffusion model may not benefit equally from these memory optimizations as a UNet-based model.

## Multiple GPUs

If you have access to more than one GPU, there a few options for efficiently loading and distributing a large model across your hardware. These features are supported by the [Accelerate](https://huggingface.co/docs/accelerate/index) library, so make sure it is installed first.

```bash
pip install -U accelerate
```

### Sharded checkpoints

Loading large checkpoints in several shards in useful because the shards are loaded one at a time. This keeps memory usage low, only requiring enough memory for the model size and the largest shard size. We recommend sharding when the fp32 checkpoint is greater than 5GB. The default shard size is 5GB.

Shard a checkpoint in [`~DiffusionPipeline.save_pretrained`] with the `max_shard_size` parameter.

```py
from diffusers import AutoModel

unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
)
unet.save_pretrained("sdxl-unet-sharded", max_shard_size="5GB")
```

Now you can use the sharded checkpoint, instead of the regular checkpoint, to save memory.

```py
import torch
from diffusers import AutoModel, StableDiffusionXLPipeline

unet = AutoModel.from_pretrained(
    "username/sdxl-unet-sharded", torch_dtype=torch.float16
)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    torch_dtype=torch.float16
).to("cuda")
```

### Device placement

> [!WARNING]
> Device placement is an experimental feature and the API may change. Only the `balanced` strategy is supported at the moment. We plan to support additional mapping strategies in the future.

The `device_map` parameter controls how the model components in a pipeline or the layers in an individual model are distributed across devices. 

<hfoptions id="device-map">
<hfoption id="pipeline level">

The `balanced` device placement strategy evenly splits the pipeline across all available devices.

```py
import torch
from diffusers import AutoModel, StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="balanced"
)
```

You can inspect a pipeline's device map with `hf_device_map`.

```py
print(pipeline.hf_device_map)
{'unet': 1, 'vae': 1, 'safety_checker': 0, 'text_encoder': 0}
```

</hfoption>
<hfoption id="model level">

The `device_map` is useful for loading large models, such as the Flux diffusion transformer which has 12.5B parameters. Set it to `"auto"` to automatically distribute a model across the fastest device first before moving to slower devices. Refer to the [Model sharding](../training/distributed_inference#model-sharding) docs for more details.

```py
import torch
from diffusers import AutoModel

transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

You can inspect a model's device map with `hf_device_map`.

```py
print(transformer.hf_device_map)
```

</hfoption>
</hfoptions>

When designing your own `device_map`, it should be a dictionary of a model's specific module name or layer and a device identifier (an integer for GPUs, `cpu` for CPUs, and `disk` for disk).

Call `hf_device_map` on a model to see how model layers are distributed and then design your own.

```py
print(transformer.hf_device_map)
{'pos_embed': 0, 'time_text_embed': 0, 'context_embedder': 0, 'x_embedder': 0, 'transformer_blocks': 0, 'single_transformer_blocks.0': 0, 'single_transformer_blocks.1': 0, 'single_transformer_blocks.2': 0, 'single_transformer_blocks.3': 0, 'single_transformer_blocks.4': 0, 'single_transformer_blocks.5': 0, 'single_transformer_blocks.6': 0, 'single_transformer_blocks.7': 0, 'single_transformer_blocks.8': 0, 'single_transformer_blocks.9': 0, 'single_transformer_blocks.10': 'cpu', 'single_transformer_blocks.11': 'cpu', 'single_transformer_blocks.12': 'cpu', 'single_transformer_blocks.13': 'cpu', 'single_transformer_blocks.14': 'cpu', 'single_transformer_blocks.15': 'cpu', 'single_transformer_blocks.16': 'cpu', 'single_transformer_blocks.17': 'cpu', 'single_transformer_blocks.18': 'cpu', 'single_transformer_blocks.19': 'cpu', 'single_transformer_blocks.20': 'cpu', 'single_transformer_blocks.21': 'cpu', 'single_transformer_blocks.22': 'cpu', 'single_transformer_blocks.23': 'cpu', 'single_transformer_blocks.24': 'cpu', 'single_transformer_blocks.25': 'cpu', 'single_transformer_blocks.26': 'cpu', 'single_transformer_blocks.27': 'cpu', 'single_transformer_blocks.28': 'cpu', 'single_transformer_blocks.29': 'cpu', 'single_transformer_blocks.30': 'cpu', 'single_transformer_blocks.31': 'cpu', 'single_transformer_blocks.32': 'cpu', 'single_transformer_blocks.33': 'cpu', 'single_transformer_blocks.34': 'cpu', 'single_transformer_blocks.35': 'cpu', 'single_transformer_blocks.36': 'cpu', 'single_transformer_blocks.37': 'cpu', 'norm_out': 'cpu', 'proj_out': 'cpu'}
```

For example, the `device_map` below places `single_transformer_blocks.10` through `single_transformer_blocks.20` on a second GPU (`1`).

```py
import torch
from diffusers import AutoModel

device_map = {
    'pos_embed': 0, 'time_text_embed': 0, 'context_embedder': 0, 'x_embedder': 0, 'transformer_blocks': 0, 'single_transformer_blocks.0': 0, 'single_transformer_blocks.1': 0, 'single_transformer_blocks.2': 0, 'single_transformer_blocks.3': 0, 'single_transformer_blocks.4': 0, 'single_transformer_blocks.5': 0, 'single_transformer_blocks.6': 0, 'single_transformer_blocks.7': 0, 'single_transformer_blocks.8': 0, 'single_transformer_blocks.9': 0, 'single_transformer_blocks.10': 1, 'single_transformer_blocks.11': 1, 'single_transformer_blocks.12': 1, 'single_transformer_blocks.13': 1, 'single_transformer_blocks.14': 1, 'single_transformer_blocks.15': 1, 'single_transformer_blocks.16': 1, 'single_transformer_blocks.17': 1, 'single_transformer_blocks.18': 1, 'single_transformer_blocks.19': 1, 'single_transformer_blocks.20': 1, 'single_transformer_blocks.21': 'cpu', 'single_transformer_blocks.22': 'cpu', 'single_transformer_blocks.23': 'cpu', 'single_transformer_blocks.24': 'cpu', 'single_transformer_blocks.25': 'cpu', 'single_transformer_blocks.26': 'cpu', 'single_transformer_blocks.27': 'cpu', 'single_transformer_blocks.28': 'cpu', 'single_transformer_blocks.29': 'cpu', 'single_transformer_blocks.30': 'cpu', 'single_transformer_blocks.31': 'cpu', 'single_transformer_blocks.32': 'cpu', 'single_transformer_blocks.33': 'cpu', 'single_transformer_blocks.34': 'cpu', 'single_transformer_blocks.35': 'cpu', 'single_transformer_blocks.36': 'cpu', 'single_transformer_blocks.37': 'cpu', 'norm_out': 'cpu', 'proj_out': 'cpu'
}

transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    device_map=device_map,
    torch_dtype=torch.bfloat16
)
```

Pass a dictionary mapping maximum memory usage to each device to enforce a limit. If a device is not in `max_memory`, it is ignored and pipeline components won't be distributed to it.

```py
import torch
from diffusers import AutoModel, StableDiffusionXLPipeline

max_memory = {0:"1GB", 1:"1GB"}
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="balanced",
    max_memory=max_memory
)
```

Diffusers uses the maxmium memory of all devices by default, but if they don't fit on the GPUs, then you'll need to use a single GPU and offload to the CPU with the methods below.

- [`~DiffusionPipeline.enable_model_cpu_offload`] only works on a single GPU but a very large model may not fit on it
- [`~DiffusionPipeline.enable_sequential_cpu_offload`] may work but it is extremely slow and also limited to a single GPU

Use the [`~DiffusionPipeline.reset_device_map`] method to reset the `device_map`. This is necessary if you want to use methods like `.to()`, [`~DiffusionPipeline.enable_sequential_cpu_offload`], and [`~DiffusionPipeline.enable_model_cpu_offload`] on a pipeline that was device-mapped.

```py
pipeline.reset_device_map()
```

## VAE slicing

VAE slicing saves memory by splitting large batches of inputs into a single batch of data and separately processing them. This method works best when generating more than one image at a time.

For example, if you're generating 4 images at once, decoding would increase peak activation memory by 4x. VAE slicing reduces this by only decoding 1 image at a time instead of all 4 images at once.

Call [`~StableDiffusionPipeline.enable_vae_slicing`] to enable sliced VAE. You can expect a small increase in performance when decoding multi-image batches and no performance impact for single-image batches.

```py
import torch
from diffusers import AutoModel, StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_vae_slicing()
pipeline(["An astronaut riding a horse on Mars"]*32).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

> [!WARNING]
> The [`AutoencoderKLWan`] and [`AsymmetricAutoencoderKL`] classes don't support slicing.

## VAE tiling

VAE tiling saves memory by dividing an image into smaller overlapping tiles instead of processing the entire image at once. This also reduces peak memory usage because the GPU is only processing a tile at a time.

Call [`~StableDiffusionPipeline.enable_vae_tiling`] to enable VAE tiling. The generated image may have some tone variation from tile-to-tile because they're decoded separately, but there shouldn't be any obvious seams between the tiles. Tiling is disabled for resolutions lower than a pre-specified (but configurable) limit. For example, this limit is 512x512 for the VAE in [`StableDiffusionPipeline`].

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.enable_vae_tiling()

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, image=init_image, strength=0.5).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

> [!WARNING]
> [`AutoencoderKLWan`] and [`AsymmetricAutoencoderKL`] don't support tiling.

## Offloading

Offloading strategies move not currently active layers or models to the CPU to avoid increasing GPU memory. These strategies can be combined with quantization and torch.compile to balance inference speed and memory usage.

Refer to the [Compile and offloading quantized models](./speed-memory-optims) guide for more details.

### CPU offloading

CPU offloading selectively moves weights from the GPU to the CPU. When a component is required, it is transferred to the GPU and when it isn't required, it is moved to the CPU. This method works on submodules rather than whole models. It saves memory by avoiding storing the entire model on the GPU.

CPU offloading dramatically reduces memory usage, but it is also **extremely slow** because submodules are passed back and forth multiple times between devices. It can often be impractical due to how slow it is.

> [!WARNING]
> Don't move the pipeline to CUDA before calling [`~DiffusionPipeline.enable_sequential_cpu_offload`], otherwise the amount of memory saved is only minimal (refer to this [issue](https://github.com/huggingface/diffusers/issues/1934) for more details). This is a stateful operation that installs hooks on the model.

Call [`~DiffusionPipeline.enable_sequential_cpu_offload`] to enable it on a pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
pipeline.enable_sequential_cpu_offload()

pipeline(
    prompt="An astronaut riding a horse on Mars",
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

### Model offloading

Model offloading moves entire models to the GPU instead of selectively moving *some* layers or model components. One of the main pipeline models, usually the text encoder, UNet, and VAE, is placed on the GPU while the other components are held on the CPU. Components like the UNet that run multiple times stays on the GPU until its completely finished and no longer needed. This eliminates the communication overhead of [CPU offloading](#cpu-offloading) and makes model offloading a faster alternative. The tradeoff is memory savings won't be as large.

> [!WARNING]
> Keep in mind that if models are reused outside the pipeline after hookes have been installed (see [Removing Hooks](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.hooks.remove_hook_from_module) for more details), you need to run the entire pipeline and models in the expected order to properly offload them. This is a stateful operation that installs hooks on the model.

Call [`~DiffusionPipeline.enable_model_cpu_offload`] to enable it on a pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

pipeline(
    prompt="An astronaut riding a horse on Mars",
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

[`~DiffusionPipeline.enable_model_cpu_offload`] also helps when you're using the [`~StableDiffusionXLPipeline.encode_prompt`] method on its own to generate the text encoders hidden state.

### Group offloading

Group offloading moves groups of internal layers ([torch.nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) or [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)) to the CPU. It uses less memory than [model offloading](#model-offloading) and it is faster than [CPU offloading](#cpu-offloading) because it reduces communication overhead.

> [!WARNING]
> Group offloading may not work with all models if the forward implementation contains weight-dependent device casting of inputs because it may clash with group offloading's device casting mechanism.

Enable group offloading by configuring the `offload_type` parameter to `block_level` or `leaf_level`.

- `block_level` offloads groups of layers based on the `num_blocks_per_group` parameter. For example, if `num_blocks_per_group=2` on a model with 40 layers, 2 layers are onloaded and offloaded at a time (20 total onloads/offloads). This drastically reduces memory requirements.
- `leaf_level` offloads individual layers at the lowest level and is equivalent to [CPU offloading](#cpu-offloading). But it can be made faster if you use streams without giving up inference speed.

Group offloading is supported for entire pipelines or individual models. Applying group offloading to the entire pipeline is the easiest option while selectively applying it to individual models gives users more flexibility to use different offloading techniques for different models.

<hfoptions id="group-offloading">
<hfoption id="pipeline">

Call [`~DiffusionPipeline.enable_group_offload`] on a pipeline.

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipeline.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)

prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance."
)
video = pipeline(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
export_to_video(video, "output.mp4", fps=8)
```

</hfoption>
<hfoption id="model">

Call [`~ModelMixin.enable_group_offload`] on standard Diffusers model components that inherit from [`ModelMixin`]. For other model components that don't inherit from [`ModelMixin`], such as a generic [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), use [`~hooks.apply_group_offloading`] instead.

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)

# Use the enable_group_offload method for Diffusers model implementations
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
pipeline.vae.enable_group_offload(onload_device=onload_device, offload_type="leaf_level")

# Use the apply_group_offloading method for other model components
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)

prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance."
)
video = pipeline(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
export_to_video(video, "output.mp4", fps=8)
```

</hfoption>
</hfoptions>

#### CUDA stream

The `use_stream` parameter can be activated for CUDA devices that support asynchronous data transfer streams to reduce overall execution time compared to [CPU offloading](#cpu-offloading). It overlaps data transfer and computation by using layer prefetching. The next layer to be executed is loaded onto the GPU while the current layer is still being executed. It can increase CPU memory significantly so ensure you have 2x the amount of memory as the model size.

Set `record_stream=True` for more of a speedup at the cost of slightly increased memory usage. Refer to the [torch.Tensor.record_stream](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html) docs to learn more.

> [!TIP]
> When `use_stream=True` on VAEs with tiling enabled, make sure to do a dummy forward pass (possible with dummy inputs as well) before inference to avoid device mismatch errors. This may not work on all implementations, so feel free to open an issue if you encounter any problems.

If you're using `block_level` group offloading with `use_stream` enabled, the `num_blocks_per_group` parameter should be set to `1`, otherwise a warning will be raised.

```py
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True, record_stream=True)
```

The `low_cpu_mem_usage` parameter can be set to `True` to reduce CPU memory usage when using streams during group offloading. It is best for `leaf_level` offloading and when CPU memory is bottlenecked. Memory is saved by creating pinned tensors on the fly instead of pre-pinning them. However, this may increase overall execution time.

#### Offloading to disk

Group offloading can consume significant system memory depending on the model size. On systems with limited memory, try group offloading onto the disk as a secondary memory.

Set the `offload_to_disk_path` argument in either [`~ModelMixin.enable_group_offload`] or [`~hooks.apply_group_offloading`] to offload the model to the disk.

```py
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", offload_to_disk_path="path/to/disk")

apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2, offload_to_disk_path="path/to/disk")
```

Refer to these [two](https://github.com/huggingface/diffusers/pull/11682#issue-3129365363) [tables](https://github.com/huggingface/diffusers/pull/11682#issuecomment-2955715126) to compare the speed and memory trade-offs.

## Layerwise casting

> [!TIP]
> Combine layerwise casting with [group offloading](#group-offloading) for even more memory savings.

Layerwise casting stores weights in a smaller data format (for example, `torch.float8_e4m3fn` and `torch.float8_e5m2`) to use less memory and upcasts those weights to a higher precision like `torch.float16` or `torch.bfloat16` for computation. Certain layers (normalization and modulation related weights) are skipped because storing them in fp8 can degrade generation quality.

> [!WARNING]
> Layerwise casting may not work with all models if the forward implementation contains internal typecasting of weights. The current implementation of layerwise casting assumes the forward pass is independent of the weight precision and the input datatypes are always specified in `compute_dtype` (see [here](https://github.com/huggingface/transformers/blob/7f5077e53682ca855afc826162b204ebf809f1f9/src/transformers/models/t5/modeling_t5.py#L294-L299) for an incompatible implementation).
>
> Layerwise casting may also fail on custom modeling implementations with [PEFT](https://huggingface.co/docs/peft/index) layers. There are some checks available but they are not extensively tested or guaranteed to work in all cases.

Call [`~ModelMixin.enable_layerwise_casting`] to set the storage and computation datatypes.

```py
import torch
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance."
)
video = pipeline(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
export_to_video(video, "output.mp4", fps=8)
```

The [`~hooks.apply_layerwise_casting`] method can also be used if you need more control and flexibility. It can be partially applied to model layers by calling it on specific internal modules. Use the `skip_modules_pattern` or `skip_modules_classes` parameters to specify modules to avoid, such as the normalization and modulation layers.

```python
import torch
from diffusers import CogVideoXTransformer3DModel
from diffusers.hooks import apply_layerwise_casting

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

# skip the normalization layer
apply_layerwise_casting(
    transformer,
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
    skip_modules_classes=["norm"],
    non_blocking=True,
)
```

## torch.channels_last

[torch.channels_last](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) flips how tensors are stored from `(batch size, channels, height, width)` to `(batch size, heigh, width, channels)`. This aligns the tensors with how the hardware sequentially accesses the tensors stored in memory and avoids skipping around in memory to access the pixel values.

Not all operators currently support the channels-last format and may result in worst performance, but it is still worth trying.

```py
print(pipeline.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
pipeline.unet.to(memory_format=torch.channels_last)  # in-place operation
print(
    pipeline.unet.conv_out.state_dict()["weight"].stride()
)  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works
```

## torch.jit.trace

[torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) records the operations a model performs on a sample input and creates a new, optimized representation of the model based on the recorded execution path. During tracing, the model is optimized to reduce overhead from Python and dynamic control flows and operations are fused together for more efficiency. The returned executable or [ScriptFunction](https://pytorch.org/docs/stable/generated/torch.jit.ScriptFunction.html) can be compiled.

```py
import time
import torch
from diffusers import StableDiffusionPipeline
import functools

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
unet_runs_per_experiment = 50

# load sample inputs
def generate_inputs():
    sample = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    timestep = torch.rand(1, device="cuda", dtype=torch.float16) * 999
    encoder_hidden_states = torch.randn((2, 77, 768), device="cuda", dtype=torch.float16)
    return sample, timestep, encoder_hidden_states


pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
unet = pipeline.unet
unet.eval()
unet.to(memory_format=torch.channels_last)  # use channels_last memory format
unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

# warmup
for _ in range(3):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet(*inputs)

# trace
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")

# warmup and optimize graph
for _ in range(5):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet_traced(*inputs)

# benchmarking
with torch.inference_mode():
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(n_experiments):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(unet_runs_per_experiment):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# save the model
unet_traced.save("unet_traced.pt")
```

Replace the pipeline's UNet with the traced version.

```py
import torch
from diffusers import StableDiffusionPipeline
from dataclasses import dataclass

@dataclass
class UNet2DConditionOutput:
    sample: torch.Tensor

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

# use jitted unet
unet_traced = torch.jit.load("unet_traced.pt")

# del pipeline.unet
class TracedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = pipe.unet.config.in_channels
        self.device = pipe.unet.device

    def forward(self, latent_model_input, t, encoder_hidden_states):
        sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

pipeline.unet = TracedUNet()

with torch.inference_mode():
    image = pipe([prompt] * 1, num_inference_steps=50).images[0]
```

## Memory-efficient attention

> [!TIP]
> Memory-efficient attention optimizes for memory usage *and* [inference speed](./fp16#scaled-dot-product-attention)!

The Transformers attention mechanism is memory-intensive, especially for long sequences, so you can try using different and more memory-efficient attention types.

By default, if PyTorch >= 2.0 is installed, [scaled dot-product attention (SDPA)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) is used. You don't need to make any additional changes to your code.

SDPA supports [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [xFormers](https://github.com/facebookresearch/xformers) as well as a native C++ PyTorch implementation. It automatically selects the most optimal implementation based on your input.

You can explicitly use xFormers with the [`~ModelMixin.enable_xformers_memory_efficient_attention`] method.

```py
# pip install xformers
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipeline.enable_xformers_memory_efficient_attention()
```

Call [`~ModelMixin.disable_xformers_memory_efficient_attention`] to disable it.

```py
pipeline.disable_xformers_memory_efficient_attention()
```