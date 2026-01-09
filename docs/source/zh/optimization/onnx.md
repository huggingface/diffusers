<!--Copyright 2025 The HuggingFace Team. All rights reserved.

根据 Apache License 2.0 许可证（以下简称"许可证"）授权，除非符合许可证要求，否则不得使用本文件。您可以通过以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或以书面形式同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件。详见许可证中规定的特定语言权限和限制。
-->

# ONNX Runtime

🤗 [Optimum](https://github.com/huggingface/optimum) 提供了兼容 ONNX Runtime 的 Stable Diffusion 流水线。您需要运行以下命令安装支持 ONNX Runtime 的 🤗 Optimum：

```bash
pip install -q optimum["onnxruntime"]
```

本指南将展示如何使用 ONNX Runtime 运行 Stable Diffusion 和 Stable Diffusion XL (SDXL) 流水线。

## Stable Diffusion

要加载并运行推理，请使用 [`~optimum.onnxruntime.ORTStableDiffusionPipeline`]。若需加载 PyTorch 模型并实时转换为 ONNX 格式，请设置 `export=True`：

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
pipeline.save_pretrained("./onnx-stable-diffusion-v1-5")
```

> [!WARNING]
> 当前批量生成多个提示可能会占用过高内存。在问题修复前，建议采用迭代方式而非批量处理。

如需离线导出 ONNX 格式流水线供后续推理使用，请使用 [`optimum-cli export`](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli) 命令：

```bash
optimum-cli export onnx --model stable-diffusion-v1-5/stable-diffusion-v1-5 sd_v15_onnx/
```

随后进行推理时（无需再次指定 `export=True`）：

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "sd_v15_onnx"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/onnxruntime/stable_diffusion_v1_5_ort_sail_boat.png">
</div>

您可以在 🤗 Optimum [文档](https://huggingface.co/docs/optimum/) 中找到更多示例，Stable Diffusion 支持文生图、图生图和图像修复任务。

## Stable Diffusion XL

要加载并运行 SDXL 推理，请使用 [`~optimum.onnxruntime.ORTStableDiffusionXLPipeline`]：

```python
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```

如需导出 ONNX 格式流水线供后续推理使用，请运行：

```bash
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task stable-diffusion-xl sd_xl_onnx/
```

SDXL 的 ONNX 格式目前支持文生图和图生图任务。
