<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 加速推理

Diffusion模型在推理时速度较慢，因为生成是一个迭代过程，需要经过一定数量的"步数"逐步将噪声细化为图像或视频。要加速这一过程，您可以尝试使用不同的[调度器](../api/schedulers/overview)、降低模型权重的精度以加快计算、使用更高效的内存注意力机制等方法。

将这些技术组合使用，可以比单独使用任何一种技术获得更快的推理速度。

本指南将介绍如何加速推理。

## 模型数据类型

模型权重的精度和数据类型会影响推理速度，因为更高的精度需要更多内存来加载，也需要更多时间进行计算。PyTorch默认以float32或全精度加载模型权重，因此更改数据类型是快速获得更快推理速度的简单方法。

<hfoptions id="dtypes">
<hfoption id="bfloat16">

bfloat16与float16类似，但对数值误差更稳健。硬件对bfloat16的支持各不相同，但大多数现代GPU都能支持bfloat16。

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

</hfoption>
<hfoption id="float16">

float16与bfloat16类似，但可能更容易出现数值误差。

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

</hfoption>
<hfoption id="TensorFloat-32">

[TensorFloat-32 (tf32)](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)模式在NVIDIA Ampere GPU上受支持，它以tf32计算卷积和矩阵乘法运算。存储和其他操作保持在float32。与bfloat16或float16结合使用时，可以显著加快计算速度。

PyTorch默认仅对卷积启用tf32模式，您需要显式启用矩阵乘法的tf32模式。

```py
import torch
from diffusers import StableDiffusionXLPipeline

torch.backends.cuda.matmul.allow_tf32 = True

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

更多详情请参阅[混合精度训练](https://huggingface.co/docs/transformers/en/perf_train_gpu_one#mixed-precision)文档。

</hfoption>
</hfoptions>

## 缩放点积注意力

> [!TIP]
> 内存高效注意力优化了推理速度*和*[内存使用](./memory#memory-efficient-attention)！

[缩放点积注意力（SDPA）](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)实现了多种注意力后端，包括[FlashAttention](https://github.com/Dao-AILab/flash-attention)、[xFormers](https://github.com/facebookresearch/xformers)和原生C++实现。它会根据您的硬件自动选择最优的后端。

如果您使用的是PyTorch >= 2.0，SDPA默认启用，无需对代码进行任何额外更改。不过，您也可以尝试使用其他注意力后端来自行选择。下面的示例使用[torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)上下文管理器来启用高效注意力。

```py
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
  image = pipeline(prompt, num_inference_steps=30).images[0]
```

## torch.compile

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)通过将PyTorch代码和操作编译为优化的内核来加速推理。Diffusers通常会编译计算密集型的模型，如UNet、transformer或VAE。

启用以下编译器设置以获得最大速度（更多选项请参阅[完整列表](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)）。

```py
import torch
from diffusers import StableDiffusionXLPipeline

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
```

加载并编译UNet和VAE。有几种不同的模式可供选择，但`"max-autotune"`通过编译为CUDA图来优化速度。CUDA图通过单个CPU操作启动多个GPU操作，有效减少了开销。

> [!TIP]
> 在PyTorch 2.3.1中，您可以控制torch.compile的缓存行为。这对于像`"max-autotune"`这样的编译模式特别有用，它会通过网格搜索多个编译标志来找到最优配置。更多详情请参阅[torch.compile中的编译时间缓存](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)教程。

将内存布局更改为[channels_last](./memory#torchchannels_last)也可以优化内存和推理速度。

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)
pipeline.unet = torch.compile(
    pipeline.unet, mode="max-autotune", fullgraph=True
)
pipeline.vae.decode = torch.compile(
    pipeline.vae.decode,
    mode="max-autotune",
    fullgraph=True
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

第一次编译时速度较慢，但一旦编译完成，速度会显著提升。尽量只在相同类型的推理操作上使用编译后的管道。在不同尺寸的图像上调用编译后的管道会重新触发编译，这会很慢且效率低下。

### 动态形状编译

> [!TIP]
> 确保始终使用PyTorch的nightly版本以获得更好的支持。

`torch.compile`会跟踪输入形状和条件，如果这些不同，它会重新编译模型。例如，如果模型是在1024x1024分辨率的图像上编译的，而在不同分辨率的图像上使用，就会触发重新编译。

为避免重新编译，添加`dynamic=True`以尝试生成更动态的内核，避免条件变化时重新编译。

```diff
+ torch.fx.experimental._config.use_duck_shape = False
+ pipeline.unet = torch.compile(
    pipeline.unet, fullgraph=True, dynamic=True
)
```

指定`use_duck_shape=False`会指示编译器是否应使用相同的符号变量来表示相同大小的输入。更多详情请参阅此[评论](https://github.com/huggingface/diffusers/pull/11327#discussion_r2047659790)。

并非所有模型都能开箱即用地从动态编译中受益，可能需要更改。参考此[PR](https://github.com/huggingface/diffusers/pull/11297/)，它改进了[`AuraFlowPipeline`]的实现以受益于动态编译。

如果动态编译对Diffusers模型的效果不如预期，请随时提出问题。

### 区域编译

[区域编译](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)通过仅编译模型中*小而频繁重复的块*（通常是transformer层）来减少冷启动延迟，并为每个后续出现的块重用编译后的工件。对于许多diffusion架构，这提供了与全图编译相同的运行时加速，并将编译时间减少了8-10倍。

使用[`~ModelMixin.compile_repeated_blocks`]方法（一个包装`torch.compile`的辅助函数）在任何组件（如transformer模型）上，如下所示。

```py
# pip install -U diffusers
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# 仅编译UNet中重复的transformer层
pipeline.unet.compile_repeated_blocks(fullgraph=True)
```

要为新模型启用区域编译，请在模型类中添加一个`_repeated_blocks`属性，包含您想要编译的块的类名（作为字符串）。

```py
class MyUNet(ModelMixin):
    _repeated_blocks = ("Transformer2DModel",)  # ← 默认编译
```

> [!TIP]
> 更多区域编译示例，请参阅参考[PR](https://github.com/huggingface/diffusers/pull/11705)。

[Accelerate](https://huggingface.co/docs/accelerate/index)中还有一个[compile_regions](https://github.com/huggingface/accelerate/blob/273799c85d849a1954a4f2e65767216eb37fa089/src/accelerate/utils/other.py#L78)方法，可以自动选择模型中的候选块进行编译。其余图会单独编译。这对于快速实验很有用，因为您不需要设置哪些块要编译或调整编译标志。

```py
# pip install -U accelerate
import torch
from diffusers import StableDiffusionXLPipeline
from accelerate.utils import compile regions

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.unet = compile_regions(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```

[`~ModelMixin.compile_repeated_blocks`]是故意显式的。在`_repeated_blocks`中列出要重复的块，辅助函数仅编译这些块。它提供了可预测的行为，并且只需一行代码即可轻松推理缓存重用。

### 图中断

在torch.compile中指定`fullgraph=True`非常重要，以确保底层模型中没有图中断。这使您可以充分利用torch.compile而不会降低性能。对于UNet和VAE，这会改变您访问返回变量的方式。

```diff
- latents = unet(
-   latents, timestep=timestep, encoder_hidden_states=prompt_embeds
-).sample

+ latents = unet(
+   latents, timestep=timestep, encoder_hidden_states=prompt_embeds, return_dict=False
+)[0]
```

### GPU同步

每次去噪器做出预测后，调度器的`step()`函数会被[调用](https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1228)，并且`sigmas`变量会被[索引](https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/schedulers/scheduling_euler_discrete.py#L476)。当放在GPU上时，这会引入延迟，因为CPU和GPU之间需要进行通信同步。当去噪器已经编译时，这一点会更加明显。

一般来说，`sigmas`应该[保持在CPU上](https://github.com/huggingface/diffusers/blob/35a969d297cba69110d175ee79c59312b9f49e1e/src/diffusers/schedulers/scheduling_euler_discrete.py#L240)，以避免通信同步和延迟。

> [!TIP]
> 参阅[torch.compile和Diffusers：峰值性能实践指南](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)博客文章，了解如何为扩散模型最大化`torch.compile`的性能。

### 基准测试

参阅[diffusers/benchmarks](https://huggingface.co/datasets/diffusers/benchmarks)数据集，查看编译管道的推理延迟和内存使用数据。

[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao#benchmarking-results)仓库还包含Flux和CogVideoX编译版本的基准测试结果。

## 动态量化

[动态量化](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)通过降低精度以加快数学运算来提高推理速度。这种特定类型的量化在运行时根据数据确定如何缩放激活，而不是使用固定的缩放因子。因此，缩放因子与数据更准确地匹配。

以下示例使用[torchao](../quantization/torchao)库对UNet和VAE应用[动态int8量化](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)。

> [!TIP]
> 参阅我们的[torchao](../quantization/torchao)文档，了解更多关于如何使用Diffusers torchao集成的信息。

配置编译器标志以获得最大速度。

```py
import torch
from torchao import apply_dynamic_quant
from diffusers import StableDiffusionXLPipeline

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
```

使用[dynamic_quant_filter_fn](https://github.com/huggingface/diffusion-fast/blob/0f169640b1db106fe6a479f78c1ed3bfaeba3386/utils/pipeline_utils.py#L16)过滤掉UNet和VAE中一些不会从动态量化中受益的线性层。

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

apply_dynamic_quant(pipeline.unet, dynamic_quant_filter_fn)
apply_dynamic_quant(pipeline.vae, dynamic_quant_filter_fn)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

## 融合投影矩阵

> [!WARNING]
> [fuse_qkv_projections](https://github.com/huggingface/diffusers/blob/58431f102cf39c3c8a569f32d71b2ea8caa461e1/src/diffusers/pipelines/pipeline_utils.py#L2034)方法是实验性的，目前主要支持Stable Diffusion管道。参阅此[PR](https://github.com/huggingface/diffusers/pull/6179)了解如何为其他管道启用它。

在注意力块中，输入被投影到三个子空间，分别由投影矩阵Q、K和V表示。这些投影通常单独计算，但您可以水平组合这些矩阵为一个矩阵，并在单步中执行投影。这会增加输入投影的矩阵乘法大小，并提高量化的效果。

```py
pipeline.fuse_qkv_projections()
```

## 资源

- 阅读[Presenting Flux Fast: Making Flux go brrr on H100s](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/)博客文章，了解如何结合所有这些优化与[TorchInductor](https://docs.pytorch.org/docs/stable/torch.compiler.html)和[AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)，使用[flux-fast](https://github.com/huggingface/flux-fast)的配方获得约2.5倍的加速。

    这些配方支持AMD硬件和[Flux.1 Kontext Dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)。
- 阅读[torch.compile和Diffusers：峰值性能实践指南](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)博客文章，了解如何在使用`torch.compile`时最大化性能。
