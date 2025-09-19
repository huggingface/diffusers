<!--Copyright 2025 The HuggingFace Team. All rights reserved.

根据 Apache License 2.0 版本（"许可证"）授权，除非符合许可证要求，否则不得使用此文件。您可以通过以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件。详见许可证中规定的特定语言权限和限制。
-->

# 概述

🤗 Diffusers 提供了一系列训练脚本供您训练自己的diffusion模型。您可以在 [diffusers/examples](https://github.com/huggingface/diffusers/tree/main/examples) 找到所有训练脚本。

每个训练脚本具有以下特点：

- **独立完整**：训练脚本不依赖任何本地文件，所有运行所需的包都通过 `requirements.txt` 文件安装
- **易于调整**：这些脚本是针对特定任务的训练示例，并不能开箱即用地适用于所有训练场景。您可能需要根据具体用例调整脚本。为此，我们完全公开了数据预处理代码和训练循环，方便您进行修改
- **新手友好**：脚本设计注重易懂性和入门友好性，而非包含最新最优方法以获得最具竞争力的结果。我们有意省略了过于复杂的训练方法
- **单一用途**：每个脚本仅针对一个任务设计，确保代码可读性和可理解性

当前提供的训练脚本包括：

| 训练类型 | 支持SDXL | 支持LoRA | 支持Flax |
|---|---|---|---|
| [unconditional image generation](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) |  |  |  |
| [text-to-image](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) | 👍 | 👍 | 👍 |
| [textual inversion](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) |  |  | 👍 |
| [DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb) | 👍 | 👍 | 👍 |
| [ControlNet](https://github.com/huggingface/diffusers/tree/main/examples/controlnet) | 👍 |  | 👍 |
| [InstructPix2Pix](https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix) | 👍 |  |  |
| [Custom Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/custom_diffusion) |  |  |  |
| [T2I-Adapters](https://github.com/huggingface/diffusers/tree/main/examples/t2i_adapter) | 👍 |  |  |
| [Kandinsky 2.2](https://github.com/huggingface/diffusers/tree/main/examples/kandinsky2_2/text_to_image) |  | 👍 |  |
| [Wuerstchen](https://github.com/huggingface/diffusers/tree/main/examples/wuerstchen/text_to_image) |  | 👍 |  |

这些示例处于**积极维护**状态，如果遇到问题请随时提交issue。如果您认为应该添加其他训练示例，欢迎创建[功能请求](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=)与我们讨论，我们将评估其是否符合独立完整、易于调整、新手友好和单一用途的标准。

## 安装

请按照以下步骤在新虚拟环境中从源码安装库，确保能成功运行最新版本的示例脚本：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

然后进入具体训练脚本目录（例如[DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)），安装对应的`requirements.txt`文件。部分脚本针对SDXL、LoRA或Flax有特定要求文件，使用时请确保安装对应文件。

```bash
cd examples/dreambooth
pip install -r requirements.txt
# 如需用DreamBooth训练SDXL
pip install -r requirements_sdxl.txt
```

为加速训练并降低内存消耗，我们建议：

- 使用PyTorch 2.0或更高版本，自动启用[缩放点积注意力](../optimization/fp16#scaled-dot-product-attention)（无需修改训练代码）
- 安装[xFormers](../optimization/xformers)以启用内存高效注意力机制