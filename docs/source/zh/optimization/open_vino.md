<!--版权所有 2025 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版本（"许可证"）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。请参阅许可证以了解具体的语言管理权限和限制。
-->

# OpenVINO

🤗 [Optimum](https://github.com/huggingface/optimum-intel) 提供与 OpenVINO 兼容的 Stable Diffusion 管道，可在各种 Intel 处理器上执行推理（请参阅支持的设备[完整列表](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)）。

您需要安装 🤗 Optimum Intel，并使用 `--upgrade-strategy eager` 选项以确保 [`optimum-intel`](https://github.com/huggingface/optimum-intel) 使用最新版本：

```bash
pip install --upgrade-strategy eager optimum["openvino"]
```

本指南将展示如何使用 Stable Diffusion 和 Stable Diffusion XL (SDXL) 管道与 OpenVINO。

## Stable Diffusion

要加载并运行推理，请使用 [`~optimum.intel.OVStableDiffusionPipeline`]。如果您想加载 PyTorch 模型并即时转换为 OpenVINO 格式，请设置 `export=True`：

```python
from optimum.intel import OVStableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]

# 别忘了保存导出的模型
pipeline.save_pretrained("openvino-sd-v1-5")
```

为了进一步加速推理，静态重塑模型。如果您更改任何参数，例如输出高度或宽度，您需要再次静态重塑模型。

```python
# 定义与输入和期望输出相关的形状
batch_size, num_images, height, width = 1, 1, 512, 512

# 静态重塑模型
pipeline.reshape(batch_size, height, width, num_images)
# 在推理前编译模型
pipeline.compile()

image = pipeline(
    prompt,
    height=height,
    width=width,
    num_images_per_prompt=num_images,
).images[0]
```
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/stable_diffusion_v1_5_sail_boat_rembrandt.png">
</div>

您可以在 🤗 Optimum [文档](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion) 中找到更多示例，Stable Diffusion 支持文本到图像、图像到图像和修复。

## Stable Diffusion XL

要加载并运行 SDXL 推理，请使用 [`~optimum.intel.OVStableDiffusionXLPipeline`]：

```python
from optimum.intel import OVStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]
```

为了进一步加速推理，可以如Stable Diffusion部分所示[静态重塑](#stable-diffusion)模型。

您可以在🤗 Optimum[文档](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion-xl)中找到更多示例，并且在OpenVINO中运行SDXL支持文本到图像和图像到图像。