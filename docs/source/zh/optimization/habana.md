<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版本（"许可证"）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。有关许可证管理权限和限制的具体语言，请参阅许可证。
-->

# Intel Gaudi

Intel Gaudi AI 加速器系列包括 [Intel Gaudi 1](https://habana.ai/products/gaudi/)、[Intel Gaudi 2](https://habana.ai/products/gaudi2/) 和 [Intel Gaudi 3](https://habana.ai/products/gaudi3/)。每台服务器配备 8 个设备，称为 Habana 处理单元 (HPU)，在 Gaudi 3 上提供 128GB 内存，在 Gaudi 2 上提供 96GB 内存，在第一代 Gaudi 上提供 32GB 内存。有关底层硬件架构的更多详细信息，请查看 [Gaudi 架构](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) 概述。

Diffusers 管道可以利用 HPU 加速，即使管道尚未添加到 [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index)，也可以通过 [GPU 迁移工具包](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html) 实现。

在您的管道上调用 `.to("hpu")` 以将其移动到 HPU 设备，如下所示为 Flux 示例：
```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipeline.to("hpu")

image = pipeline("一张松鼠在毕加索风格中的图像").images[0]
```

> [!TIP]
> 对于 Gaudi 优化的扩散管道实现，我们推荐使用 [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index)。