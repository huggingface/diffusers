<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版本（"许可证"）授权；除非符合许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。有关许可证的具体语言，请参阅许可证中的权限和限制。
-->

# 如何使用 Core ML 运行 Stable Diffusion

[Core ML](https://developer.apple.com/documentation/coreml) 是 Apple 框架支持的模型格式和机器学习库。如果您有兴趣在 macOS 或 iOS/iPadOS 应用中运行 Stable Diffusion 模型，本指南将展示如何将现有的 PyTorch 检查点转换为 Core ML 格式，并使用 Python 或 Swift 进行推理。

Core ML 模型可以利用 Apple 设备中所有可用的计算引擎：CPU、GPU 和 Apple Neural Engine（或 ANE，一种在 Apple Silicon Mac 和现代 iPhone/iPad 中可用的张量优化加速器）。根据模型及其运行的设备，Core ML 还可以混合和匹配计算引擎，例如，模型的某些部分可能在 CPU 上运行，而其他部分在 GPU 上运行。

> [!TIP]
> 您还可以使用 PyTorch 内置的 `mps` 加速器在 Apple Silicon Mac 上运行 `diffusers` Python 代码库。这种方法在 [mps 指南](mps) 中有详细解释，但它与原生应用不兼容。

## Stable Diffusion Core ML 检查点

Stable Diffusion 权重（或检查点）以 PyTorch 格式存储，因此在使用它们之前，需要将它们转换为 Core ML 格式。

幸运的是，Apple 工程师基于 `diffusers` 开发了 [一个转换工具](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)，用于将 PyTorch 检查点转换为 Core ML。

但在转换模型之前，花点时间探索 Hugging Face Hub——很可能您感兴趣的模型已经以 Core ML 格式提供：

- [Apple](https://huggingface.co/apple) 组织包括 Stable Diffusion 版本 1.4、1.5、2.0 基础和 2.1 基础
- [coreml community](https://huggingface.co/coreml-community) 包括自定义微调模型
- 使用此 [过滤器](https://huggingface.co/models?pipeline_tag=text-to-image&library=coreml&p=2&sort=likes) 返回所有可用的 Core ML 检查点

如果您找不到感兴趣的模型，我们建议您遵循 Apple 的 [Converting Models to Core ML](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) 说明。

## 选择要使用的 Core ML 变体

Stable Diffusion 模型可以转换为不同的 Core ML 变体，用于不同目的：

- 注意力类型
使用了n个块。注意力操作用于“关注”图像表示中不同区域之间的关系，并理解图像和文本表示如何相关。注意力的计算和内存消耗很大，因此存在不同的实现方式，以适应不同设备的硬件特性。对于Core ML Stable Diffusion模型，有两种注意力变体：
* `split_einsum`（[由Apple引入](https://machinelearning.apple.com/research/neural-engine-transformers)）针对ANE设备进行了优化，这些设备在现代iPhone、iPad和M系列计算机中可用。
* “原始”注意力（在`diffusers`中使用的基础实现）仅与CPU/GPU兼容，不与ANE兼容。在CPU + GPU上使用`original`注意力运行模型可能比ANE*更快*。请参阅[此性能基准](https://huggingface.co/blog/fast-mac-diffusers#performance-benchmarks)以及社区提供的[一些额外测量](https://github.com/huggingface/swift-coreml-diffusers/issues/31)以获取更多细节。

- 支持的推理框架。
* `packages`适用于Python推理。这可用于在尝试将转换后的Core ML模型集成到原生应用程序之前进行测试，或者如果您想探索Core ML性能但不需要支持原生应用程序。例如，具有Web UI的应用程序完全可以使用Python Core ML后端。
* `compiled`模型是Swift代码所必需的。Hub中的`compiled`模型将大型UNet模型权重分成多个文件，以兼容iOS和iPadOS设备。这对应于[`--chunk-unet`转换选项](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)。如果您想支持原生应用程序，则需要选择`compiled`变体。

官方的Core ML Stable Diffusion[模型](https://huggingface.co/apple/coreml-stable-diffusion-v1-4/tree/main)包括这些变体，但社区的可能有所不同：

```
coreml-stable-diffusion-v1-4
├── README.md
├── original
│   ├── compiled
│   └── packages
└── split_einsum
    ├── compiled
    └── packages
```

您可以下载并使用所需的变体，如下所示。

## Python中的Core ML推理

安装以下库以在Python中运行Core ML推理：

```bash
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
```

### 下载模型检查点

要在Python中运行推理，请使用存储在`packages`文件夹中的版本之一，因为`compiled`版本仅与Swift兼容。您可以选择使用`original`或`split_einsum`注意力。

这是您如何从Hub下载`original`注意力变体到一个名为`models`的目录：

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

mo
del_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### 推理[[python-inference]]

下载模型快照后，您可以使用 Apple 的 Python 脚本来测试它。

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i ./models/coreml-stable-diffusion-v1-4_original_packages/original/packages -o </path/to/output/image> --compute-unit CPU_AND_GPU --seed 93
```

使用 `-i` 标志将下载的检查点路径传递给脚本。`--compute-unit` 表示您希望允许用于推理的硬件。它必须是以下选项之一：`ALL`、`CPU_AND_GPU`、`CPU_ONLY`、`CPU_AND_NE`。您也可以提供可选的输出路径和用于可重现性的种子。

推理脚本假设您使用的是 Stable Diffusion 模型的原始版本，`CompVis/stable-diffusion-v1-4`。如果您使用另一个模型，您*必须*在推理命令行中使用 `--model-version` 选项指定其 Hub ID。这适用于已支持的模型以及您自己训练或微调的自定义模型。

例如，如果您想使用 [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)：

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-v1-5_original_packages --model-version stable-diffusion-v1-5/stable-diffusion-v1-5
```

## Core ML 在 Swift 中的推理

在 Swift 中运行推理比在 Python 中稍快，因为模型已经以 `mlmodelc` 格式编译。这在应用启动时加载模型时很明显，但如果在之后运行多次生成，则不应明显。

### 下载

要在您的 Mac 上运行 Swift 推理，您需要一个 `compiled` 检查点版本。我们建议您使用类似于先前示例的 Python 代码在本地下载它们，但使用 `compiled` 变体之一：

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/compiled"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### 推理[[swift-inference]]

要运行推理，请克隆 Apple 的仓库：

```bash
git clone https://github.com/apple/ml-stable-diffusion
cd ml-stable-diffusion
```

然后使用 Apple 的命令行工具，[Swift Package Manager](https://www.swift.org/package-manager/#)：

```bash
swift run StableDiffusionSample --resource-path models/coreml-stable-diffusion-v1-4_original_compiled --compute-units all "a photo of an astronaut riding a horse on mars"
```

您必须在 `--resource-path` 中指定上一步下载的检查点之一，请确保它包含扩展名为 `.mlmodelc` 的已编译 Core ML 包。`--compute-units` 必须是以下值之一：`all`、`cpuOnly`、`cpuAndGPU`、`cpuAndNeuralEngine`。

有关更多详细信息，请参考 [Apple 仓库中的说明](https://github.com/apple/ml-stable-diffusion)。

## 支持的 Diffusers 功能

Core ML 模型和推理代码不支持 🧨 Diffusers 的许多功能、选项和灵活性。以下是一些需要注意的限制：

- Core ML 模型仅适用于推理。它们不能用于训练或微调。
- 只有两个调度器已移植到 Swift：Stable Diffusion 使用的默认调度器和我们从 `diffusers` 实现移植到 Swift 的 `DPMSolverMultistepScheduler`。我们推荐您使用 `DPMSolverMultistepScheduler`，因为它在约一半的步骤中产生相同的质量。
- 负面提示、无分类器引导尺度和图像到图像任务在推理代码中可用。高级功能如深度引导、ControlNet 和潜在上采样器尚不可用。

Apple 的 [转换和推理仓库](https://github.com/apple/ml-stable-diffusion) 和我们自己的 [swift-coreml-diffusers](https://github.com/huggingface/swift-coreml-diffusers) 仓库旨在作为技术演示，以帮助其他开发者在此基础上构建。

如果您对任何缺失功能有强烈需求，请随时提交功能请求或更好的是，贡献一个 PR 🙂。

## 原生 Diffusers Swift 应用

一个简单的方法来在您自己的 Apple 硬件上运行 Stable Diffusion 是使用 [我们的开源 Swift 仓库](https://github.com/huggingface/swift-coreml-diffusers)，它基于 `diffusers` 和 Apple 的转换和推理仓库。您可以研究代码，使用 [Xcode](https://developer.apple.com/xcode/) 编译它，并根据您的需求进行适配。为了方便，[App Store 中还有一个独立 Mac 应用](https://apps.apple.com/app/diffusers/id1666309574)，因此您无需处理代码或 IDE 即可使用它。如果您是开发者，并已确定 Core ML 是构建您的 Stable Diffusion 应用的最佳解决方案，那么您可以使用本指南的其余部分来开始您的项目。我们迫不及待想看看您会构建什么 🙂。