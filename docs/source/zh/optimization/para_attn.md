# ParaAttention

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-performance.png">
</div>
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/hunyuan-video-performance.png">
</div>

大型图像和视频生成模型，如 [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 和 [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)，由于其规模，可能对实时应用和部署构成推理挑战。

[ParaAttention](https://github.com/chengzeyi/ParaAttention) 是一个实现了**上下文并行**和**第一块缓存**的库，可以与其他技术（如 torch.compile、fp8 动态量化）结合使用，以加速推理。

本指南将展示如何在 NVIDIA L20 GPU 上对 FLUX.1-dev 和 HunyuanVideo 应用 ParaAttention。
在我们的基线基准测试中，除了 HunyuanVideo 为避免内存不足错误外，未应用任何优化。

我们的基线基准测试显示，FLUX.1-dev 能够在 28 步中生成 1024x1024 分辨率图像，耗时 26.36 秒；HunyuanVideo 能够在 30 步中生成 129 帧 720p 分辨率视频，耗时 3675.71 秒。

> [!TIP]
> 对于更快的上下文并行推理，请尝试使用支持 NVLink 的 NVIDIA A100 或 H100 GPU（如果可用），尤其是在 GPU 数量较多时。

## 第一块缓存

缓存模型中 transformer 块的输出并在后续推理步骤中重用它们，可以降低计算成本并加速推理。

然而，很难决定何时重用缓存以确保生成图像或视频的质量。ParaAttention 直接使用**第一个 transformer 块输出的残差差异**来近似模型输出之间的差异。当差异足够小时，重用先前推理步骤的残差差异。换句话说，跳过去噪步骤。

这在 FLUX.1-dev 和 HunyuanVideo 推理上实现了 2 倍加速，且质量非常好。

<figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/ada-cache.png" alt="Cache in Diffusion Transformer" />
    <figcaption>AdaCache 的工作原理，第一块缓存是其变体</figcaption>
</figure>

<hfoptions id="first-block-cache">
<hfoption id="FLUX-1.dev">

要在 FLUX.1-dev 上应用第一块缓存，请调用 `apply_cache_on_pipe`，如下所示。0.08 是 FLUX 模型的默认残差差异值。

```python
import time
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_thre
shold=0.08)

# 启用内存节省
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

begin = time.time()
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
).images[0]
end = time.time()
print(f"Time: {end - begin:.2f}s")

print("Saving image to flux.png")
image.save("flux.png")
```

| 优化 | 原始 | FBCache rdt=0.06 | FBCache rdt=0.08 | FBCache rdt=0.10 | FBCache rdt=0.12 |
| - | - | - | - | - | - |
| 预览 | ![Original](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-original.png) | ![FBCache rdt=0.06](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.06.png) | ![FBCache rdt=0.08](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.08.png) | ![FBCache rdt=0.10](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.10.png) | ![FBCache rdt=0.12](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.12.png) |
| 墙时间 (s) | 26.36 | 21.83 | 17.01 | 16.00 | 13.78 |

First Block Cache 将推理速度降低到 17.01 秒，与基线相比，或快 1.55 倍，同时保持几乎零质量损失。

</hfoption>
<hfoption id="HunyuanVideo">

要在 HunyuanVideo 上应用 First Block Cache，请使用 `apply_cache_on_pipe`，如下所示。0.06 是 HunyuanVideo 模型的默认残差差值。

```python
import time
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=0.6)

pipe.vae.enable_tiling()

begin = time.time()
output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
).frames[0]
end = time.time()
print(f"Time: {end - begin:.2f}s")

print("Saving video to hunyuan_video.mp4")
export_to_video(output, "hunyuan_video.mp4", fps=15)
```

<video controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/hunyuan-video-original.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>

<small> HunyuanVideo 无 FBCache </small>

<video controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/hunyuan-video-fbc.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<small> HunyuanVideo 与 FBCache </small>

First Block Cache 将推理速度降低至 2271.06 秒，相比基线快了 1.62 倍，同时保持了几乎为零的质量损失。

</hfoption>
</hfoptions>

## fp8 量化

fp8 动态量化进一步加速推理并减少内存使用。为了使用 8 位 [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)，必须对激活和权重进行量化。

使用 `float8_weight_only` 和 `float8_dynamic_activation_float8_weight` 来量化文本编码器和变换器模型。

默认量化方法是逐张量量化，但如果您的 GPU 支持逐行量化，您也可以尝试它以获得更好的准确性。

使用以下命令安装 [torchao](https://github.com/pytorch/ao/tree/main)。

```bash
pip3 install -U torch torchao
```

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) 使用 `mode="max-autotune-no-cudagraphs"` 或 `mode="max-autotune"` 选择最佳内核以获得性能。如果是第一次调用模型，编译可能会花费很长时间，但一旦模型编译完成，这是值得的。

此示例仅量化变换器模型，但您也可以量化文本编码器以进一步减少内存使用。

> [!TIP]
> 动态量化可能会显著改变模型输出的分布，因此您需要将 `residual_diff_threshold` 设置为更大的值以使其生效。

<hfoptions id="fp8-quantization">
<hfoption id="FLUX-1.dev">

```python
import time
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(
    pipe,
    residual_diff_threshold=0.12,  # 使用更大的值以使缓存生效
)

from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only

quantize_(pipe.text_encoder, float8_weight_only())
quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs",
)

# 启用内存节省
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

for i in range(2):
    begin = time.time()
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28,
    ).images[0]
    end = time.time()
    if i == 0:
        print(f"预热时间: {end - begin:.2f}s")
    else:
        print(f"时间: {end - begin:.2f}s")

print("保存图像到 flux.png")
image.save("flux.png")
```

fp8 动态量化和 torch.compile 将推理速度降低至 7.56 秒，相比基线快了 3.48 倍。
</hfoption>
<hfoption id="HunyuanVideo">

```python
import time
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only

quantize_(pipe.text_encoder, float8_weight_only())
quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs",
)

# Enable memory savings
pipe.vae.enable_tiling()
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

for i in range(2):
    begin = time.time()
    output = pipe(
        prompt="A cat walks on the grass, realistic",
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=1 if i == 0 else 30,
    ).frames[0]
    end = time.time()
    if i == 0:
        print(f"Warm up time: {end - begin:.2f}s")
    else:
        print(f"Time: {end - begin:.2f}s")

print("Saving video to hunyuan_video.mp4")
export_to_video(output, "hunyuan_video.mp4", fps=15)
```

NVIDIA L20 GPU 仅有 48GB 内存，在编译后且如果未调用 `enable_model_cpu_offload` 时，可能会遇到内存不足（OOM）错误，因为 HunyuanVideo 在高分辨率和大量帧数运行时具有非常大的激活张量。对于内存少于 80GB 的 GPU，可以尝试降低分辨率和帧数来避免 OOM 错误。

大型视频生成模型通常受注意力计算而非全连接层的瓶颈限制。这些模型不会从量化和 torch.compile 中显著受益。

</hfoption>
</hfoptions>

## 上下文并行性

上下文并行性并行化推理并随多个 GPU 扩展。ParaAttention 组合设计允许您将上下文并行性与第一块缓存和动态量化结合使用。

> [!TIP]
> 请参考 [ParaAttention](https://github.com/chengzeyi/ParaAttention/tree/main) 仓库获取详细说明和如何使用多个 GPU 扩展推理的示例。

如果推理过程需要持久化和可服务，建议使用 [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html) 编写您自己的推理处理器。这可以消除启动进程以及加载和重新编译模型的开销。

<hfoptions id="context-parallelism">
<hfoption id="FLUX-1.dev">

以下代码示例结合了第一块缓存、fp8动态量化、torch.compile和上下文并行，以实现最快的推理速度。

```python
import time
import torch
import torch.distributed as dist
from diffusers import FluxPipeline

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
    max_ring_dim_size=2,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(
    pipe,
    residual_diff_threshold=0.12,  # 使用较大的值以使缓存生效
)

from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only

quantize_(pipe.text_encoder, float8_weight_only())
quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs",
)

# 启用内存节省
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_sequential_cpu_offload(gpu_id=dist.get_rank())

for i in range(2):
    begin = time.time()
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).images[0]
    end = time.time()
    if dist.get_rank() == 0:
        if i == 0:
            print(f"预热时间: {end - begin:.2f}s")
        else:
            print(f"时间: {end - begin:.2f}s")

if dist.get_rank() == 0:
    print("将图像保存到flux.png")
    image.save("flux.png")

dist.destroy_process_group()
```

保存到`run_flux.py`并使用[torchrun](https://pytorch.org/docs/stable/elastic/run.html)启动。

```bash
# 使用--nproc_per_node指定GPU数量
torchrun --nproc_per_node=2 run_flux.py
```

推理速度降至8.20秒，相比基线快了3.21倍，使用2个NVIDIA L20 GPU。在4个L20上，推理速度为3.90秒，快了6.75倍。

</hfoption>
<hfoption id="HunyuanVideo">

以下代码示例结合了第一块缓存和上下文并行，以实现最快的推理速度。

```python
import time
import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

# from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only
#
# torch._inductor.config.reorder_for_compute_comm_overlap = True
#
# quantize_(pipe.text_encoder, float8_weight_only())
# quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
# pipe.transformer = torch.compile(
#    pipe.transformer, mode="max-autotune-no-cudagraphs",
# )

# 启用内存节省
pipe.vae.enable_tiling()
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_sequential_cpu_offload(gpu_id=dist.get_rank())

for i in range(2):
    begin = time.time()
    output = pipe(
        prompt="A cat walks on the grass, realistic",
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=1 if i == 0 else 30,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).frames[0]
    end = time.time()
    if dist.get_rank() == 0:
        if i == 0:
            print(f"预热时间: {end - begin:.2f}s")
        else:
            print(f"时间: {end - begin:.2f}s")

if dist.get_rank() == 0:
    print("保存视频到 hunyuan_video.mp4")
    export_to_video(output, "hunyuan_video.mp4", fps=15)

dist.destroy_process_group()
```

保存到 `run_hunyuan_video.py` 并使用 [torchrun](https://pytorch.org/docs/stable/elastic/run.html) 启动。

```bash
# 使用 --nproc_per_node 指定 GPU 数量
torchrun --nproc_per_node=8 run_hunyuan_video.py
```

推理速度降低到 649.23 秒，相比基线快 5.66 倍，使用 8 个 NVIDIA L20 GPU。

</hfoption>
</hfoptions>

## 基准测试

<hfoptions id="conclusion">
<hfoption id="FLUX-1.dev">

| GPU 类型 | GPU 数量 | 优化 | 墙钟时间 (s) | 加速比 |
| - | - | - | - | - |
| NVIDIA L20 | 1 | 基线 | 26.36 | 1.00x |
| NVIDIA L20 | 1 | FBCache (rdt=0.08) | 17.01 | 1.55x |
| NVIDIA L20 | 1 | FP8 DQ | 13.40 | 1.96x |
| NVIDIA L20 | 1 | FBCache (rdt=0.12) + FP8 DQ | 7.56 | 3.48x |
| NVIDIA L20 | 2 | FBCache (rdt=0.12) + FP8 DQ + CP | 4.92 | 5.35x |
| NVIDIA L20 | 4 | FBCache (rdt=0.12) + FP8 DQ + CP | 3.90 | 6.75x |

</hfoption>
<hfoption id="HunyuanVideo">

| GPU 类型 | GPU 数量 | 优化 | 墙钟时间 (s) | 加速比 |
| - | - | - | - | - |
| NVIDIA L20 | 1 | 基线 | 3675.71 | 1.00x |
| NVIDIA
L20 | 1 | FBCache | 2271.06 | 1.62x |
| NVIDIA L20 | 2 | FBCache + CP | 1132.90 | 3.24x |
| NVIDIA L20 | 4 | FBCache + CP | 718.15 | 5.12x |
| NVIDIA L20 | 8 | FBCache + CP | 649.23 | 5.66x |

</hfoption>
</hfoptions>