<!--版权所有 2024 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版（“许可证”）授权；除非符合许可证，否则不得使用此文件。
您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“原样”分发，不附带任何明示或暗示的担保或条件。有关许可证的特定语言，请参阅许可证。
-->

# 编译和卸载量化模型

优化模型通常涉及[推理速度](./fp16)和[内存使用](./memory)之间的权衡。例如，虽然[缓存](./cache)可以提高推理速度，但它也会增加内存消耗，因为它需要存储中间注意力层的输出。一种更平衡的优化策略结合了量化模型、[torch.compile](./fp16#torchcompile) 和各种[卸载方法](./memory#offloading)。

> [!TIP]
> 查看 [torch.compile](./fp16#torchcompile) 指南以了解更多关于编译以及如何在此处应用的信息。例如，区域编译可以显著减少编译时间，而不会放弃任何加速。

对于图像生成，结合量化和[模型卸载](./memory#model-offloading)通常可以在质量、速度和内存之间提供最佳权衡。组卸载对于图像生成效果不佳，因为如果计算内核更快完成，通常不可能*完全*重叠数据传输。这会导致 CPU 和 GPU 之间的一些通信开销。

对于视频生成，结合量化和[组卸载](./memory#group-offloading)往往更好，因为视频模型更受计算限制。

下表提供了优化策略组合及其对 Flux 延迟和内存使用的影响的比较。

| 组合 | 延迟 (s) | 内存使用 (GB) |
|---|---|---|
| 量化 | 32.602 | 14.9453 |
| 量化, torch.compile | 25.847 | 14.9448 |
| 量化, torch.compile, 模型 CPU 卸载 | 32.312 | 12.2369 |
<small>这些结果是在 Flux 上使用 RTX 4090 进行基准测试的。transformer 和 text_encoder 组件已量化。如果您有兴趣评估自己的模型，请参考[基准测试脚本](https://gist.github.com/sayakpaul/0db9d8eeeb3d2a0e5ed7cf0d9ca19b7d)。</small>

本指南将向您展示如何使用 [bitsandbytes](../quantization/bitsandbytes#torchcompile) 编译和卸载量化模型。确保您正在使用 [PyTorch nightly](https://pytorch.org/get-started/locally/) 和最新版本的 bitsandbytes。

```bash
pip install -U bitsandbytes
```

## 量化和 torch.compile

首先通过[量化](../quantization/overview)模型来减少存储所需的内存，并[编译](./fp16#torchcompile)它以加速推理。

配置 [Dynamo](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) `capture_dynamic_output_shape_ops = True` 以在编译 bitsandbytes 模型时处理动态输出。

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

torch._dynamo.config.capture_dynamic_output_shape_ops = True

# 量化
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# 编译
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer.compile(mode="max-autotune", fullgraph=True)
pipeline("""
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
).images[0]
```

## 量化、torch.compile 和卸载

除了量化和 torch.compile，如果您需要进一步减少内存使用，可以尝试卸载。卸载根据需要将各种层或模型组件从 CPU 移动到 GPU 进行计算。

在卸载期间配置 [Dynamo](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) `cache_size_limit` 以避免过多的重新编译，并设置 `capture_dynamic_output_shape_ops = True` 以在编译 bitsandbytes 模型时处理动态输出。

<hfoptions id="offloading">
<hfoption id="model CPU offloading">

[模型 CPU 卸载](./memory#model-offloading) 将单个管道组件（如 transformer 模型）在需要计算时移动到 GPU。否则，它会被卸载到 CPU。

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# 量化
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# 模型 CPU 卸载
pipeline.enable_model_cpu_offload()

# 编译
pipeline.transformer.compile()
pipeline(
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
).images[0]
```

</hfoption>
<hfoption id="group offloading">

[组卸载](./memory#group-offloading) 将单个管道组件（如变换器模型）的内部层移动到 GPU 进行计算，并在不需要时将其卸载。同时，它使用 [CUDA 流](./memory#cuda-stream) 功能来预取下一层以执行。

通过重叠计算和数据传输，它比模型 CPU 卸载更快，同时还能节省内存。

```py
# pip install ftfy
import torch
from diffusers import AutoModel, DiffusionPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import UMT5EncoderModel

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# 量化
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder"],
)

text_encoder = UMT5EncoderModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16
)
pipeline = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# 组卸载
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipeline.transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
    non_blocking=True
)
pipeline.vae.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
    non_blocking=True
)
apply_group_offloading(
    pipeline.text_encoder,
    onload_device=onload_device,
    offload_type="leaf_level",
    use_stream=True,
    non_blocking=True
)

# 编译
pipeline.transformer.compile()

prompt = """
The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

</hfoption>
</hfoptions>