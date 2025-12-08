# Pruna

[Pruna](https://github.com/PrunaAI/pruna) 是一个模型优化框架，提供多种优化方法——量化、剪枝、缓存、编译——以加速推理并减少内存使用。以下是优化方法的概览。

| 技术       | 描述                                                                                   | 速度 | 内存 | 质量 |
|------------|---------------------------------------------------------------------------------------|:----:|:----:|:----:|
| `batcher`  | 将多个输入分组在一起同时处理，提高计算效率并减少处理时间。                                  | ✅   | ❌   | ➖   |
| `cacher`   | 存储计算的中间结果以加速后续操作。                                                       | ✅   | ➖   | ➖   |
| `compiler` | 为特定硬件优化模型指令。                                                                 | ✅   | ➖   | ➖   |
| `distiller`| 训练一个更小、更简单的模型来模仿一个更大、更复杂的模型。                                   | ✅   | ✅   | ❌   |
| `quantizer`| 降低权重和激活的精度，减少内存需求。                                                       | ✅   | ✅   | ❌   |
| `pruner`   | 移除不重要或冗余的连接和神经元，产生一个更稀疏、更高效的网络。                               | ✅   | ✅   | ❌   |
| `recoverer`| 在压缩后恢复模型的性能。                                                                 | ➖   | ➖   | ✅   |
| `factorizer`| 将多个小矩阵乘法批处理为一个大型融合操作。                                                | ✅   | ➖   | ➖   |
| `enhancer` | 通过应用后处理算法（如去噪或上采样）来增强模型输出。                                        | ❌   | -    | ✅   |

✅ (改进), ➖ (大致相同), ❌ (恶化)

在 [Pruna 文档](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html#configure-algorithms) 中探索所有优化方法。

## 安装

使用以下命令安装 Pruna。

```bash
pip install pruna
```

## 优化 Diffusers 模型

Diffusers 模型支持广泛的优化算法，如下所示。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/diffusers_combinations.png" alt="Diffusers 模型支持的优化算法概览">
</div>

下面的示例使用 factorizer、compiler 和 cacher 算法的组合优化 [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)。这种组合将推理速度加速高达 4.2 倍，并将峰值 GPU 内存使用从 34.7GB 减少到 28.0GB，同时几乎保持相同的输出质量。

> [!TIP]
> 参考 [Pruna 优化](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html) 文档以了解更多关于该操作的信息。
本示例中使用的优化技术。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_combination.png" alt="用于FLUX.1-dev的优化技术展示，结合了因子分解器、编译器和缓存器算法">
</div>

首先定义一个包含要使用的优化算法的`SmashConfig`。要优化模型，将管道和`SmashConfig`用`smash`包装，然后像往常一样使用管道进行推理。

```python
import torch
from diffusers import FluxPipeline

from pruna import PrunaModel, SmashConfig, smash

# 加载模型
# 使用小GPU内存尝试segmind/Segmind-Vega或black-forest-labs/FLUX.1-schnell
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# 定义配置
smash_config = SmashConfig()
smash_config["factorizer"] = "qkv_diffusers"
smash_config["compiler"] = "torch_compile"
smash_config["torch_compile_target"] = "module_list"
smash_config["cacher"] = "fora"
smash_config["fora_interval"] = 2

# 为了获得最佳速度结果，可以添加这些配置
# 但它们会将预热时间从1.5分钟增加到10分钟
# smash_config["torch_compile_mode"] = "max-autotune-no-cudagraphs"
# smash_config["quantizer"] = "torchao"
# smash_config["torchao_quant_type"] = "fp8dq"
# smash_config["torchao_excluded_modules"] = "norm+embedding"

# 优化模型
smashed_pipe = smash(pipe, smash_config)

# 运行模型
smashed_pipe("a knitted purple prune").images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png">
</div>

优化后，我们可以使用Hugging Face Hub共享和加载优化后的模型。

```python
# 保存模型
smashed_pipe.save_to_hub("<username>/FLUX.1-dev-smashed")

# 加载模型
smashed_pipe = PrunaModel.from_hub("<username>/FLUX.1-dev-smashed")
```

## 评估和基准测试Diffusers模型

Pruna提供了[EvaluationAgent](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/evaluate.html)来评估优化后模型的质量。

我们可以定义我们关心的指标，如总时间和吞吐量，以及要评估的数据集。我们可以定义一个模型并将其传递给`EvaluationAgent`。

<hfoptions id="eval">
<hfoption id="optimized model">

我们可以通过使用`EvaluationAgent`加载和评估优化后的模型，并将其传递给`Task`。

```python
import torch
from diffusers import FluxPipeline

from pruna import PrunaModel
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics import (
    ThroughputMetric,
    TorchMetricWrapper,
    TotalTimeMetric,
)
from pruna.evaluation.task import Task

# define the device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 加载模型
# 使用小GPU内存尝试 PrunaAI/Segmind-Vega-smashed 或 PrunaAI/FLUX.1-dev-smashed
smashed_pipe = PrunaModel.from_hub("PrunaAI/FLUX.1-dev-smashed")

# 定义指标
metrics = [
    TotalTimeMetric(n_iterations=20, n_warmup_iterations=5),
    ThroughputMetric(n_iterations=20, n_warmup_iterations=5),
    TorchMetricWrapper("clip"),
]

# 定义数据模块
datamodule = PrunaDataModule.from_string("LAION256")
datamodule.limit_datasets(10)

# 定义任务和评估代理
task = Task(metrics, datamodule=datamodule, device=device)
eval_agent = EvaluationAgent(task)

# 评估优化模型并卸载到CPU
smashed_pipe.move_to_device(device)
smashed_pipe_results = eval_agent.evaluate(smashed_pipe)
smashed_pipe.move_to_device("cpu")
```

</hfoption>
<hfoption id="standalone model">

除了比较优化模型与基础模型，您还可以评估独立的 `diffusers` 模型。这在您想评估模型性能而不考虑优化时非常有用。我们可以通过使用 `PrunaModel` 包装器并运行 `EvaluationAgent` 来实现。

```python
import torch
from diffusers import FluxPipeline

from pruna import PrunaModel

# 加载模型
# 使用小GPU内存尝试 PrunaAI/Segmind-Vega-smashed 或 PrunaAI/FLUX.1-dev-smashed
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cpu")
wrapped_pipe = PrunaModel(model=pipe)
```

</hfoption>
</hfoptions>

现在您已经了解了如何优化和评估您的模型，可以开始使用 Pruna 来优化您自己的模型了。幸运的是，我们有许多示例来帮助您入门。

> [!TIP]
> 有关基准测试 Flux 的更多详细信息，请查看 [宣布 FLUX-Juiced：最快的图像生成端点（快 2.6 倍）！](https://huggingface.co/blog/PrunaAI/flux-fastest-image-generation-endpoint) 博客文章和 [InferBench](https://huggingface.co/spaces/PrunaAI/InferBench) 空间。

## 参考

- [Pruna](https://github.com/pruna-ai/pruna)
- [Pruna 优化](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html#configure-algorithms)
- [Pruna 评估](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/evaluate.html)
- [Pruna 教程](https://docs.pruna.ai/en/stable/docs_pruna/tutorials/index.html)