<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版本（"许可证"）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。请参阅许可证了解具体的语言管理权限和限制。
-->

# Metal Performance Shaders (MPS)

> [!TIP]
> 带有 <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22"> 徽章的管道表示模型可以利用 Apple silicon 设备上的 MPS 后端进行更快的推理。欢迎提交 [Pull Request](https://github.com/huggingface/diffusers/compare) 来为缺少此徽章的管道添加它。

🤗 Diffusers 与 Apple silicon（M1/M2 芯片）兼容，使用 PyTorch 的 [`mps`](https://pytorch.org/docs/stable/notes/mps.html) 设备，该设备利用 Metal 框架来发挥 MacOS 设备上 GPU 的性能。您需要具备：

- 配备 Apple silicon（M1/M2）硬件的 macOS 计算机
- macOS 12.6 或更高版本（推荐 13.0 或更高）
- arm64 版本的 Python
- [PyTorch 2.0](https://pytorch.org/get-started/locally/)（推荐）或 1.13（支持 `mps` 的最低版本）

`mps` 后端使用 PyTorch 的 `.to()` 接口将 Stable Diffusion 管道移动到您的 M1 或 M2 设备上：

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# 如果您的计算机内存小于 64 GB，推荐使用
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image
```

> [!WARNING]
> PyTorch [mps](https://pytorch.org/docs/stable/notes/mps.html) 后端不支持大小超过 `2**32` 的 NDArray。如果您遇到此问题，请提交 [Issue](https://github.com/huggingface/diffusers/issues/new/choose) 以便我们调查。

如果您使用 **PyTorch 1.13**，您需要通过管道进行一次额外的"预热"传递。这是一个临时解决方法，用于解决首次推理传递产生的结果与后续传递略有不同的问题。您只需要执行此传递一次，并且在仅进行一次推理步骤后可以丢弃结果。

```diff
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to("mps")
  pipe.enable_attention_slicing()

  prompt = "a photo of an astronaut riding a horse on mars"
  # 如果 PyTorch 版本是 1.13，进行首次"预热"传递
+ _ = pipe(prompt, num_inference_steps=1)

  # 预热传递后，结果与 CPU 设备上的结果匹配。
  image = pipe(prompt).images[0]
```

## 故障排除

本节列出了使用 `mps` 后端时的一些常见问题及其解决方法。

### 注意力切片

M1/M2 性能对内存压力非常敏感。当发生这种情况时，系统会自动交换内存，这会显著降低性能。

为了防止这种情况发生，我们建议使用*注意力切片*来减少推理过程中的内存压力并防止交换。这在您的计算机系统内存少于 64GB 或生成非标准分辨率（大于 512×512 像素）的图像时尤其相关。在您的管道上调用 [`~DiffusionPipeline.enable_attention_slicing`] 函数：

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
pipeline.enable_attention_slicing()
```

注意力切片将昂贵的注意力操作分多个步骤执行，而不是一次性完成。在没有统一内存的计算机中，它通常能提高约 20% 的性能，但我们观察到在大多数 Apple 芯片计算机中，除非您有 64GB 或更多 RAM，否则性能会*更好*。

### 批量推理

批量生成多个提示可能会导致崩溃或无法可靠工作。如果是这种情况，请尝试迭代而不是批量处理。