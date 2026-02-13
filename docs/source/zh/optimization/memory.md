<!--版权所有 2025 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版本（“许可证”）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“原样”分发，无任何明示或暗示的担保或条件。有关许可证的特定语言管理权限和限制，请参阅许可证。
-->

# 减少内存使用

现代diffusion models，如 [Flux](../api/pipelines/flux) 和 [Wan](../api/pipelines/wan)，拥有数十亿参数，在您的硬件上进行推理时会占用大量内存。这是一个挑战，因为常见的 GPU 通常没有足够的内存。为了克服内存限制，您可以使用多个 GPU（如果可用）、将一些管道组件卸载到 CPU 等。

本指南将展示如何减少内存使用。

> [!TIP]
> 请记住，这些技术可能需要根据模型进行调整。例如，基于 transformer 的扩散模型可能不会像基于 UNet 的模型那样从这些内存优化中同等受益。

## 多个 GPU

如果您有多个 GPU 的访问权限，有几种选项可以高效地在硬件上加载和分发大型模型。这些功能由 [Accelerate](https://huggingface.co/docs/accelerate/index) 库支持，因此请确保先安装它。

```bash
pip install -U accelerate
```

### 分片检查点

将大型检查点加载到多个分片中很有用，因为分片会逐个加载。这保持了低内存使用，只需要足够的内存来容纳模型大小和最大分片大小。我们建议当 fp32 检查点大于 5GB 时进行分片。默认分片大小为 5GB。

在 [`~DiffusionPipeline.save_pretrained`] 中使用 `max_shard_size` 参数对检查点进行分片。

```py
from diffusers import AutoModel

unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
)
unet.save_pretrained("sdxl-unet-sharded", max_shard_size="5GB")
```

现在您可以使用分片检查点，而不是常规检查点，以节省内存。

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

### 设备放置

> [!WARNING]
> 设备放置是一个实验性功能，API 可能会更改。目前仅支持 `balanced` 策略。我们计划在未来支持额外的映射策略。

`device_map` 参数控制管道或模型中的组件如何
单个模型中的层分布在多个设备上。

<hfoptions id="device-map">
<hfoption id="pipeline level">

`balanced` 设备放置策略将管道均匀分割到所有可用设备上。

```py
import torch
from diffusers import AutoModel, StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="balanced"
)
```

您可以使用 `hf_device_map` 检查管道的设备映射。

```py
print(pipeline.hf_device_map)
{'unet': 1, 'vae': 1, 'safety_checker': 0, 'text_encoder': 0}
```

</hfoption>
<hfoption id="model level">

`device_map` 对于加载大型模型非常有用，例如具有 125 亿参数的 Flux diffusion transformer。将其设置为 `"auto"` 可以自动将模型首先分布到最快的设备上，然后再移动到较慢的设备。有关更多详细信息，请参阅 [模型分片](../training/distributed_inference#model-sharding) 文档。

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

您可以使用 `hf_device_map` 检查模型的设备映射。

```py
print(transformer.hf_device_map)
```

</hfoption>
</hfoptions>

当设计您自己的 `device_map` 时，它应该是一个字典，包含模型的特定模块名称或层以及设备标识符（整数表示 GPU，`cpu` 表示 CPU，`disk` 表示磁盘）。

在模型上调用 `hf_device_map` 以查看模型层如何分布，然后设计您自己的映射。

```py
print(transformer.hf_device_map)
{'pos_embed': 0, 'time_text_embed': 0, 'context_embedder': 0, 'x_embedder': 0, 'transformer_blocks': 0, 'single_transformer_blocks.0': 0, 'single_transformer_blocks.1': 0, 'single_transformer_blocks.2': 0, 'single_transformer_blocks.3': 0, 'single_transformer_blocks.4': 0, 'single_transformer_blocks.5': 0, 'single_transformer_blocks.6': 0, 'single_transformer_blocks.7': 0, 'single_transformer_blocks.8': 0, 'single_transformer_blocks.9': 0, 'single_transformer_blocks.10': 'cpu', 'single_transformer_blocks.11': 'cpu', 'single_transformer_blocks.12': 'cpu', 'single_transformer_blocks.13': 'cpu', 'single_transformer_blocks.14': 'cpu', 'single_transformer_blocks.15': 'cpu', 'single_transformer_blocks.16': 'cpu', 'single_transformer_blocks.17': 'cpu', 'single_transformer_blocks.18': 'cpu', 'single_transformer_blocks.19': 'cpu', 'single_transformer_blocks.20': 'cpu', 'single_transformer_blocks.21': 'cpu', 'single_transformer_blocks.22': 'cpu', 'single_transformer_blocks.23': 'cpu', 'single_transformer_blocks.24': 'cpu', 'single_transformer_blocks.25': 'cpu', 'single_transformer_blocks.26': 'cpu', 'single_transformer_blocks.27': 'cpu', 'single_transformer_blocks.28': 'cpu', 'single_transformer_blocks.29': 'cpu', 'single_transformer_blocks.30': 'cpu', 'single_transformer_blocks.31': 'cpu', 'single_transformer_blocks.32': 'cpu', 'single_transformer_blocks.33': 'cpu', 'single_transformer_blocks.34': 'cpu', 'single_transformer_blocks.35': 'cpu', 'single_transformer_blocks.36': 'cpu', 'single_transformer_blocks.37': 'cpu', 'norm_out': 'cpu', 'proj_out': 'cpu'}
```

例如，下面的 `device_map` 将 `single_transformer_blocks.10` 到 `single_transformer_blocks.20` 放置在第二个 GPU（`1`）上。

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

传递一个字典，将最大内存使用量映射到每个设备以强制执行限制。如果设备不在 `max_memory` 中，它将被忽略，管道组件不会分发到该设备。

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

Diffusers 默认使用所有设备的最大内存，但如果它们无法适应 GPU，则需要使用单个 GPU 并通过以下方法卸载到 CPU。

- [`~DiffusionPipeline.enable_model_cpu_offload`] 仅适用于单个 GPU，但非常大的模型可能无法适应它
- 使用 [`~DiffusionPipeline.enable_sequential_cpu_offload`] 可能有效，但它极其缓慢，并且仅限于单个 GPU。

使用 [`~DiffusionPipeline.reset_device_map`] 方法来重置 `device_map`。如果您想在已进行设备映射的管道上使用方法如 `.to()`、[`~DiffusionPipeline.enable_sequential_cpu_offload`] 和 [`~DiffusionPipeline.enable_model_cpu_offload`]，这是必要的。

```py
pipeline.reset_device_map()
```

## VAE 切片

VAE 切片通过将大批次输入拆分为单个数据批次并分别处理它们来节省内存。这种方法在同时生成多个图像时效果最佳。

例如，如果您同时生成 4 个图像，解码会将峰值激活内存增加 4 倍。VAE 切片通过一次只解码 1 个图像而不是所有 4 个图像来减少这种情况。

调用 [`~StableDiffusionPipeline.enable_vae_slicing`] 来启用切片 VAE。您可以预期在解码多图像批次时性能会有小幅提升，而在单图像批次时没有性能影响。

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
> [`AutoencoderKLWan`] 和 [`AsymmetricAutoencoderKL`] 类不支持切片。

## VAE 平铺

VAE 平铺通过将图像划分为较小的重叠图块而不是一次性处理整个图像来节省内存。这也减少了峰值内存使用量，因为 GPU 一次只处理一个图块。

调用 [`~StableDiffusionPipeline.enable_vae_tiling`] 来启用 VAE 平铺。生成的图像可能因图块到图块的色调变化而有所不同，因为它们被单独解码，但图块之间不应有明显的接缝。对于低于预设（但可配置）限制的分辨率，平铺被禁用。例如，对于 [`StableDiffusionPipeline`] 中的 VAE，此限制为 512x512。

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
> [`AutoencoderKLWan`] 和 [`AsymmetricAutoencoderKL`] 不支持平铺。

## 卸载

卸载策略将非当前活动层移动
将模型移动到 CPU 以避免增加 GPU 内存。这些策略可以与量化和 torch.compile 结合使用，以平衡推理速度和内存使用。

有关更多详细信息，请参考 [编译和卸载量化模型](./speed-memory-optims) 指南。

### CPU 卸载

CPU 卸载选择性地将权重从 GPU 移动到 CPU。当需要某个组件时，它被传输到 GPU；当不需要时，它被移动到 CPU。此方法作用于子模块而非整个模型。它通过避免将整个模型存储在 GPU 上来节省内存。

CPU 卸载显著减少内存使用，但由于子模块在设备之间多次来回传递，它也非常慢。由于速度极慢，它通常不实用。

> [!WARNING]
> 在调用 [`~DiffusionPipeline.enable_sequential_cpu_offload`] 之前，不要将管道移动到 CUDA，否则节省的内存非常有限（更多细节请参考此 [issue](https://github.com/huggingface/diffusers/issues/1934)）。这是一个状态操作，会在模型上安装钩子。

调用 [`~DiffusionPipeline.enable_sequential_cpu_offload`] 以在管道上启用它。

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

### 模型卸载

模型卸载将整个模型移动到 GPU，而不是选择性地移动某些层或模型组件。一个主要管道模型，通常是文本编码器、UNet 和 VAE，被放置在 GPU 上，而其他组件保持在 CPU 上。像 UNet 这样运行多次的组件会一直留在 GPU 上，直到完全完成且不再需要。这消除了 [CPU 卸载](#cpu-offloading) 的通信开销，使模型卸载成为一个更快的替代方案。权衡是内存节省不会那么大。

> [!WARNING]
> 请注意，如果在安装钩子后模型在管道外部被重用（更多细节请参考 [移除钩子](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.hooks.remove_hook_from_module)），您需要按预期顺序运行整个管道和模型以正确卸载它们。这是一个状态操作，会在模型上安装钩子。

调用 [`~DiffusionPipeline.enable_model_cpu_offload`] 以在管道上启用它。

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
print(f"最大内存保留: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

[`~DiffusionPipeline.enable_model_cpu_offload`] 在您单独使用 [`~StableDiffusionXLPipeline.encode_prompt`] 方法生成文本编码器隐藏状态时也有帮助。

### 组卸载

组卸载将内部层组（[torch.nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) 或 [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)）移动到 CPU。它比[模型卸载](#model-offloading)使用更少的内存，并且比[CPU 卸载](#cpu-offloading)更快，因为它减少了通信开销。

> [!WARNING]
> 如果前向实现包含权重相关的输入设备转换，组卸载可能不适用于所有模型，因为它可能与组卸载的设备转换机制冲突。

调用 [`~ModelMixin.enable_group_offload`] 为继承自 [`ModelMixin`] 的标准 Diffusers 模型组件启用它。对于不继承自 [`ModelMixin`] 的其他模型组件，例如通用 [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)，使用 [`~hooks.apply_group_offloading`] 代替。

`offload_type` 参数可以设置为 `block_level` 或 `leaf_level`。

- `block_level` 基于 `num_blocks_per_group` 参数卸载层组。例如，如果 `num_blocks_per_group=2` 在一个有 40 层的模型上，每次加载和卸载 2 层（总共 20 次加载/卸载）。这大大减少了内存需求。
- `leaf_level` 在最低级别卸载单个层，等同于[CPU 卸载](#cpu-offloading)。但如果您使用流而不放弃推理速度，它可以更快。

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)

# 对 Diffusers 模型实现使用 enable_group_offload 方法
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
pipeline.vae.enable_group_offload(onload_device=onload_device, offload_type="leaf_level")

# 对其他模型组件使用 apply_group_offloading 方法
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

#### CUDA 流
`use_stream` 参数可以激活支持异步数据传输流的 CUDA 设备，以减少整体执行时间，与 [CPU 卸载](#cpu-offloading) 相比。它通过使用层预取重叠数据传输和计算。下一个要执行的层在当前层仍在执行时加载到 GPU 上。这会显著增加 CPU 内存，因此请确保您有模型大小的 2 倍内存。

设置 `record_stream=True` 以获得更多速度提升，代价是内存使用量略有增加。请参阅 [torch.Tensor.record_stream](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html) 文档了解更多信息。

> [!TIP]
> 当 `use_stream=True` 在启用平铺的 VAEs 上时，确保在推理前进行虚拟前向传递（可以使用虚拟输入），以避免设备不匹配错误。这可能不适用于所有实现，因此如果遇到任何问题，请随时提出问题。

如果您在使用启用 `use_stream` 的 `block_level` 组卸载，`num_blocks_per_group` 参数应设置为 `1`，否则会引发警告。

```py
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True, record_stream=True)
```

`low_cpu_mem_usage` 参数可以设置为 `True`，以在使用流进行组卸载时减少 CPU 内存使用。它最适合 `leaf_level` 卸载和 CPU 内存瓶颈的情况。通过动态创建固定张量而不是预先固定它们来节省内存。然而，这可能会增加整体执行时间。

#### 卸载到磁盘
组卸载可能会消耗大量系统内存，具体取决于模型大小。在内存有限的系统上，尝试将组卸载到磁盘作为辅助内存。

在 [`~ModelMixin.enable_group_offload`] 或 [`~hooks.apply_group_offloading`] 中设置 `offload_to_disk_path` 参数，将模型卸载到磁盘。

```py
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", offload_to_disk_path="path/to/disk")

apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2, offload_to_disk_path="path/to/disk")
```

参考这些[两个](https://github.com/huggingface/diffusers/pull/11682#issue-3129365363)[表格](https://github.com/huggingface/diffusers/pull/11682#issuecomment-2955715126)来比较速度和内存的权衡。

## 分层类型转换

> [!TIP]
> 将分层类型转换与[组卸载](#group-offloading)结合使用，以获得更多内存节省。

分层类型转换将权重存储在较小的数据格式中（例如 `torch.float8_e4m3fn` 和 `torch.float8_e5m2`），以使用更少的内存，并在计算时将那些权重上转换为更高精度如 `torch.float16` 或 `torch.bfloat16`。某些层（归一化和调制相关权重）被跳过，因为将它们存储在 fp8 中可能会降低生成质量。

> [!WARNING]
> 如果前向实现包含权重的内部类型转换，分层类型转换可能不适用于所有模型。当前的分层类型转换实现假设前向传递独立于权重精度，并且输入数据类型始终在 `compute_dtype` 中指定（请参见[这里](https://github.com/huggingface/transformers/blob/7f5077e53682ca855afc826162b204ebf809f1f9/src/transformers/models/t5/modeling_t5.py#L294-L299)以获取不兼容的实现）。
>
> 分层类型转换也可能在使用[PEFT](https://huggingface.co/docs/peft/index)层的自定义建模实现上失败。有一些检查可用，但它们没有经过广泛测试或保证在所有情况下都能工作。

调用 [`~ModelMixin.enable_layerwise_casting`] 来设置存储和计算数据类型。

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

[`~hooks.apply_layerwise_casting`] 方法也可以在您需要更多控制和灵活性时使用。它可以通过在特定内部模块上调用它来部分应用于模型层。使用 `skip_modules_pattern` 或 `skip_modules_classes` 参数来指定要避免的模块，例如归一化和调制层。

```python
import torch
from diffusers import CogVideoXTransformer3DModel
from diffusers.hooks import apply_layerwise_casting

transformer = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

# 跳过归一化层
apply_layerwise_casting(
    transformer,
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
    skip_modules_classes=["norm"],
    non_blocking=True,
)
```

## torch.channels_last

[torch.channels_last](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) 将张量的存储方式从 `(批次大小, 通道数, 高度, 宽度)` 翻转为 `(批次大小, 高度, 宽度, 通道数)`。这使张量与硬件如何顺序访问存储在内存中的张量对齐，并避免了在内存中跳转以访问像素值。

并非所有运算符当前都支持通道最后格式，并且可能导致性能更差，但仍然值得尝试。

```py
print(pipeline.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
pipeline.unet.to(memory_format=torch.channels_last)  # 原地操作
print(
    pipeline.unet.conv_out.state_dict()["weight"].stride()
)  # (2880, 1, 960, 320) 第二个维度的跨度为1证明它有效
```

## torch.jit.trace

[torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) 记录模型在样本输入上执行的操作，并根据记录的执行路径创建一个新的、优化的模型表示。在跟踪过程中，模型被优化以减少来自Python和动态控制流的开销，并且操作被融合在一起以提高效率。返回的可执行文件或 [ScriptFunction](https://pytorch.org/docs/stable/generated/torch.jit.ScriptFunction.html) 可以被编译。

```py
import time
import torch
from diffusers import StableDiffusionPipeline
import functools

# torch 禁用梯度
torch.set_grad_enabled(False)

# 设置变量
n_experiments = 2
unet_runs_per_experiment = 50

# 加载样本输入
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
unet.to(memory
_format=torch.channels_last)  # 使用 channels_last 内存格式
unet.forward = functools.partial(unet.forward, return_dict=False)  # 设置 return_dict=False 为默认

# 预热
for _ in range(3):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet(*inputs)

# 追踪
print("tracing..")
unet_traced = torch.jit.trace(unet, inputs)
unet_traced.eval()
print("done tracing")

# 预热和优化图
for _ in range(5):
    with torch.inference_mode():
        inputs = generate_inputs()
        orig_output = unet_traced(*inputs)

# 基准测试
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

# 保存模型
unet_traced.save("unet_traced.pt")
```

替换管道的 UNet 为追踪版本。

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

# 使用 jitted unet
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

## 内存高效注意力

> [!TIP]
> 内存高效注意力优化内存使用 *和* [推理速度](./fp16#scaled-dot-product-attention)！

Transformers 注意力机制是内存密集型的，尤其对于长序列，因此您可以尝试使用不同且更内存高效的注意力类型。

默认情况下，如果安装了 PyTorch >= 2.0，则使用 [scaled dot-product attention (SDPA)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。您无需对代码进行任何额外更改。

SDPA 还支持 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 和 [xFormers](https://github.com/facebookresearch/xformers)，以及 a
这是一个原生的 C++ PyTorch 实现。它会根据您的输入自动选择最优的实现。

您可以使用 [`~ModelMixin.enable_xformers_memory_efficient_attention`] 方法显式地使用 xFormers。

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

调用 [`~ModelMixin.disable_xformers_memory_efficient_attention`] 来禁用它。

```py
pipeline.disable_xformers_memory_efficient_attention()
```