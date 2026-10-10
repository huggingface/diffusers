<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reduce memory usage

Modern diffusion models have billions of parameters, which often exceeds the memory available on a consumer GPU. Diffusers provides several techniques to reduce memory usage, such as distributing a model across multiple GPUs, offloading components to the CPU, and storing weights in lower precision.

Choose a technique based on your hardware and workload.

| Situation | Technique |
|---|---|
| You have more than one GPU | [Multiple GPUs](#multiple-gpus) |
| You generate several images per prompt | [VAE slicing](#vae-slicing) |
| You generate high-resolution images | [VAE tiling](#vae-tiling) |
| The model doesn't fit on a single GPU | [Offloading](#offloading) |
| You want to store weights in lower precision | [Layerwise casting](#layerwise-casting) |

> [!TIP]
> Results vary by model. For example, a transformer-based model may not benefit from some of these techniques as much as a UNet-based model.

## Multiple GPUs

If you have access to more than one GPU, there are a few options for efficiently loading and distributing a large model across your hardware. These features are supported by the [Accelerate](https://huggingface.co/docs/accelerate/index) library, so make sure it is installed.

```bash
pip install -U accelerate
```

### Sharded checkpoints

A sharded checkpoint splits a large checkpoint into several smaller files that are loaded one at a time. Peak memory only needs to fit the model and the largest shard, instead of the model and the entire checkpoint. Sharding is recommended when the fp32 checkpoint is larger than 5GB. The default shard size is 10GB.

Shard a checkpoint with the `max_shard_size` parameter in [`~ModelMixin.save_pretrained`].

```py
from diffusers import AutoModel

unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
)
unet.save_pretrained("sdxl-unet-sharded", max_shard_size="5GB")
```

Load the sharded checkpoint in place of the original to reduce peak memory while loading.

```py
import torch
from diffusers import AutoModel, StableDiffusionXLPipeline

unet = AutoModel.from_pretrained(
    "username/sdxl-unet-sharded", dtype=torch.float16
)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
```

### Device placement

The `device_map` parameter controls how the model components in a pipeline or the layers in an individual model are distributed across devices.

> [!WARNING]
> Device placement is an experimental feature and the API may change. At the pipeline level, `device_map` accepts `"balanced"`, or a single device such as `"cuda"` or `"cpu"`.

<hfoptions id="device-map">
<hfoption id="pipeline level">

The `balanced` device placement strategy evenly splits the pipeline across all available devices.

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="balanced"
)
```

You can inspect a pipeline's device map with `hf_device_map`.

```py
print(pipeline.hf_device_map)
```

</hfoption>
<hfoption id="model level">

Set `device_map="auto"` to distribute the layers of a large model across devices. The fastest device is filled first before moving to slower devices.

```py
import torch
from diffusers import AutoModel

transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    device_map="auto",
    dtype=torch.bfloat16
)
```

You can inspect a model's device map with `hf_device_map`.

```py
print(transformer.hf_device_map)
```

</hfoption>
</hfoptions>

A custom `device_map` is a dictionary that maps module names to devices, where a device is an integer for a GPU, `"cpu"`, or `"disk"`. Print a model's `hf_device_map` to see how its layers are distributed, and use it as a starting point to design your own.

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
    dtype=torch.bfloat16
)
```

To cap how much memory each GPU uses, pass `max_memory`, a dictionary that maps each device to a limit. Components aren't placed on GPUs you leave out, and a component that doesn't fit under a GPU's limit is placed on the CPU instead.

```py
import torch
from diffusers import StableDiffusionXLPipeline

max_memory = {0: "16GB", 1: "16GB"}
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    device_map="balanced",
    max_memory=max_memory
)
```

By default, Diffusers uses all available memory on each GPU. Components that don't fit on a GPU are placed on the CPU. If most of the pipeline ends up on the CPU, try a single GPU with one of the offloading methods below instead.

- [`~DiffusionPipeline.enable_model_cpu_offload`] moves one whole model to the GPU at a time. It's faster, but each model must fit on a single GPU.
- [`~DiffusionPipeline.enable_sequential_cpu_offload`] moves one submodule to the GPU at a time. It uses the least GPU memory, but it's very slow.

Before calling `.to()`, `enable_sequential_cpu_offload`, or `enable_model_cpu_offload` on a device-mapped pipeline, reset its device map with [`~DiffusionPipeline.reset_device_map`].

```py
pipeline.reset_device_map()
```

## VAE slicing

VAE slicing splits a batch of latents into single latents and decodes them one at a time. The decoded images are concatenated back into a batch at the end. Peak decoding memory stays close to the cost of decoding one image, which makes slicing useful when generating several images at once. It has no effect on single-image batches.

```text
Without slicing: one decode for the whole batch

  [ z1 | z2 | z3 | z4 ] --> VAE decode --> [ img1 | img2 | img3 | img4 ]
                            (peak memory: 4 images)

With slicing: decode one latent at a time, then concatenate

  [ z1 ] --> VAE decode --> [ img1 ] --+
  [ z2 ] --> VAE decode --> [ img2 ] --+
  [ z3 ] --> VAE decode --> [ img3 ] --+
  [ z4 ] --> VAE decode --> [ img4 ] --+
             (peak memory: 1 image)    |
                                       v
                         [ img1 | img2 | img3 | img4 ]
```

Call [`~AutoencoderKL.enable_slicing`] to enable VAE slicing.

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.vae.enable_slicing()
pipeline(["An astronaut riding a horse on Mars"]*32).images[0]
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

> [!WARNING]
> [`AsymmetricAutoencoderKL`] doesn't support slicing.

## VAE tiling

VAE tiling splits a latent into overlapping tiles and decodes each tile separately. The overlapping edges are blended together to stitch the tiles into the final image. Peak decoding memory depends mostly on the tile size instead of the full image size, which makes tiling useful for generating high-resolution images.

```text
1. Split the latent into tiles         2. Decode each tile       3. Blend the overlaps
   that overlap by 25%                    separately                and stitch

   +--------+----+--------+
   | tile 1 |####| tile 2 |               tile 1 --> decode         +-------------------+
   |        |####|        |               tile 2 --> decode         |                   |
   +--------+----+--------+     -->       tile 3 --> decode   -->   |   full-resolution |
   |########|####|########|               tile 4 --> decode         |       image       |
   +--------+----+--------+                                         |                   |
   | tile 3 |####| tile 4 |           (peak memory: 1 tile)         +-------------------+
   |        |####|        |
   +--------+----+--------+
   #### = region shared by neighboring tiles
```

Tiles are decoded separately, so tone may vary slightly from tile to tile, but there shouldn't be any obvious seams. Tiling also applies when encoding an image into latents, and only activates when the input is larger than the VAE's `sample_size`.

Call [`~AutoencoderKL.enable_tiling`] to enable VAE tiling.

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.vae.enable_tiling()

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, image=init_image, strength=0.5).images[0]
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

> [!WARNING]
> [`AsymmetricAutoencoderKL`] doesn't support tiling.

## Offloading

Offloading keeps inactive layers or models on the CPU and moves them to the GPU only when they're needed. You can combine offloading with quantization and torch.compile to balance inference speed and memory usage.

Refer to the [Compiling and offloading quantized models](./speed-memory-optims) guide for more details.

### Sequential CPU offloading

Sequential CPU offloading keeps weights on the CPU and moves each submodule to the GPU only when it runs. The entire model is never on the GPU at once, so sequential offloading uses the least GPU memory of the offloading methods. It's also the slowest because submodules are transferred between devices many times during inference, which often makes it impractical.

> [!WARNING]
> Don't move the pipeline to CUDA before calling `enable_sequential_cpu_offload`, otherwise the memory savings are minimal. Refer to [issue #1934](https://github.com/huggingface/diffusers/issues/1934) for more details. Sequential offloading is stateful and installs hooks on the model.

Call [`~DiffusionPipeline.enable_sequential_cpu_offload`] to enable it on a pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", dtype=torch.bfloat16
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
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

### Model offloading

Model offloading moves whole models to the GPU instead of individual submodules. Only one model, such as the text encoder, denoiser (UNet or transformer), or VAE, is on the GPU at a time while the other models stay on the CPU. A model that runs multiple times, like the denoiser, stays on the GPU until it finishes. Model offloading avoids the transfer overhead of [sequential CPU offloading](#sequential-cpu-offloading), which makes it faster, but the memory savings are smaller.

> [!WARNING]
> Model offloading is stateful and installs hooks on each model. If you call a model outside the pipeline, run the models in the pipeline's order so they're offloaded correctly, or [remove the hooks](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.hooks.remove_hook_from_module) first.

Call [`~DiffusionPipeline.enable_model_cpu_offload`] to enable it on a pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", dtype=torch.bfloat16
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
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

Model offloading also helps when you call [`~StableDiffusionXLPipeline.encode_prompt`] on its own, because only the text encoders are moved to the GPU.

### Group offloading

Group offloading moves groups of internal layers ([torch.nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) or [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)) to the CPU. It usually uses less memory than [model offloading](#model-offloading) and runs faster than [sequential CPU offloading](#sequential-cpu-offloading) because it reduces communication overhead.

> [!WARNING]
> Group offloading may not work with models whose forward pass moves inputs to the weights' device, because that conflicts with how group offloading moves tensors.

Set `offload_type` to `block_level` or `leaf_level` to choose how layers are grouped.

- `block_level` offloads groups of layers, and `num_blocks_per_group` sets the size of each group. For example, `num_blocks_per_group=2` on a model with 40 layers creates 20 groups and moves 2 layers to the GPU at a time.
- `leaf_level` offloads each individual layer, similar to [sequential CPU offloading](#sequential-cpu-offloading), but it can be much faster with [CUDA streams](#cuda-stream).

Group offloading is supported for entire pipelines or individual models. Apply it to the whole pipeline for the simplest setup, or to individual models to mix offloading techniques.

<hfoptions id="group-offloading">
<hfoption id="pipeline">

Call [`~DiffusionPipeline.enable_group_offload`] on a pipeline.

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", dtype=torch.bfloat16)
pipeline.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)

prompt = "A panda playing a tiny acoustic guitar in a bamboo forest"
video = pipeline(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
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
pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", dtype=torch.bfloat16)

# Use the enable_group_offload method for Diffusers model implementations
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
pipeline.vae.enable_group_offload(onload_device=onload_device, offload_type="leaf_level")

# Use the apply_group_offloading method for other model components
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)

prompt = "A panda playing a tiny acoustic guitar in a bamboo forest"
video = pipeline(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
export_to_video(video, "output.mp4", fps=8)
```

</hfoption>
</hfoptions>

#### CUDA stream

Set `use_stream=True` on CUDA devices to prefetch the next layer onto the GPU while the current layer is still running. Overlapping data transfer and computation can make group offloading much faster than [sequential CPU offloading](#sequential-cpu-offloading). Streams create a pinned copy of each weight in CPU memory, so system RAM usage can reach about twice the model size.

Set `record_stream=True` for more of a speedup at the cost of slightly increased memory usage. Refer to the [torch.Tensor.record_stream](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html) docs to learn more.

> [!TIP]
> If a VAE has tiling enabled and `use_stream=True`, run a forward pass with dummy inputs before inference to avoid device mismatch errors. Open an [issue](https://github.com/huggingface/diffusers/issues) if this doesn't work for your model.

Streams require `num_blocks_per_group=1` with `block_level` offloading. Other values log a warning and are reset to `1`.

```py
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True, record_stream=True)
```

Set `low_cpu_mem_usage=True` to reduce CPU memory usage with streams. Tensors are pinned on the fly instead of all at once up front, which saves CPU memory but can increase inference time. It works best with `leaf_level` offloading when CPU memory is the bottleneck.

#### Offloading to disk

Group offloading can use a lot of system memory depending on the model size. On systems with limited RAM, offload to disk instead.

Set the `offload_to_disk_path` argument in either [`~ModelMixin.enable_group_offload`] or [`~hooks.apply_group_offloading`] to offload the model to the disk.

```py
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", offload_to_disk_path="path/to/disk")

apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2, offload_to_disk_path="path/to/disk")
```

Compare the speed and memory trade-offs in the [disk offloading benchmark](https://github.com/huggingface/diffusers/pull/11682#issue-3129365363) and the [follow-up benchmark](https://github.com/huggingface/diffusers/pull/11682#issuecomment-2955715126).

## Layerwise casting

Layerwise casting stores weights in a smaller data format (for example, `torch.float8_e4m3fn` and `torch.float8_e5m2`) to use less memory and upcasts those weights to a higher precision like `torch.float16` or `torch.bfloat16` for computation. By default, positional embeddings, patch embeddings, normalization layers, and the input and output projections are skipped because storing them in fp8 can degrade generation quality.

Combine layerwise casting with [group offloading](#group-offloading) for even more memory savings.

> [!WARNING]
> Layerwise casting may not work with all models if the forward implementation contains internal typecasting of weights. The current implementation of layerwise casting assumes the forward pass is independent of the weight precision and the input datatypes are always specified in `compute_dtype` (see the [T5 implementation in Transformers](https://github.com/huggingface/transformers/blob/7f5077e53682ca855afc826162b204ebf809f1f9/src/transformers/models/t5/modeling_t5.py#L294-L299) for an incompatible example).
>
> Layerwise casting may also fail on custom modeling implementations with [PEFT](https://huggingface.co/docs/peft/index) layers. Diffusers includes some checks for this, but they aren't extensively tested and may not catch every case.

Call [`~ModelMixin.enable_layerwise_casting`] to set the storage and computation datatypes.

```py
import torch
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b",
    transformer=transformer,
    dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"
prompt = "A panda playing a tiny acoustic guitar in a bamboo forest"
video = pipeline(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
export_to_video(video, "output.mp4", fps=8)
```

For more control, use [`~hooks.apply_layerwise_casting`]. Call it on specific internal modules to apply layerwise casting to only part of a model, and use `skip_modules_pattern` or `skip_modules_classes` to exclude modules such as normalization layers.

```python
import torch
from diffusers import CogVideoXTransformer3DModel
from diffusers.hooks import apply_layerwise_casting

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    dtype=torch.bfloat16
)

# skip the normalization layer
apply_layerwise_casting(
    transformer,
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
    skip_modules_pattern=["norm"],
    non_blocking=True,
)
```

## torch.channels_last

[torch.channels_last](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) changes how tensors are stored in memory from `(batch size, channels, height, width)` to `(batch size, height, width, channels)`. Storing each pixel's channels next to each other matches how many GPU kernels read memory, which mainly speeds up inference rather than reducing memory.

channels_last only affects 4D tensors, so it can benefit convolution-based models like UNets and VAEs. Not all operators support the channels-last format, and some models may run slower with it, so benchmark it on your model first.

```py
import torch
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16
).to("cuda")

print(pipeline.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
pipeline.unet.to(memory_format=torch.channels_last)  # in-place operation
print(
    pipeline.unet.conv_out.state_dict()["weight"].stride()
)  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works
```

## Memory-efficient attention

Diffusers supports multiple memory-efficient attention backends (FlashAttention, xFormers, SageAttention, and more) through [`~ModelMixin.set_attention_backend`]. Refer to the [Attention backends](./attention_backends) guide to learn how to switch between them.

## Next steps

- Combine offloading with quantization and torch.compile in the [Compiling and offloading quantized models](./speed-memory-optims) guide.
- Reduce memory further with [quantization](../quantization/overview).
- Switch attention implementations in the [Attention backends](./attention_backends) guide.
