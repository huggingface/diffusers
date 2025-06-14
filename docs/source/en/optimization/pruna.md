# Pruna

[Pruna](https://github.com/PrunaAI/pruna) is a model optimization framework that offers various optimization methods - quantization, pruning, caching, compilation - for accelerating inference and reducing memory usage. A general overview of the optimization methods are shown below.


| Technique    | Description                                                                                   | Speed | Memory | Quality |
|--------------|-----------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`    | Groups multiple inputs together to be processed simultaneously, improving computational efficiency and reducing processing time. | ✅    | ❌     | ➖      |
| `cacher`     | Stores intermediate results of computations to speed up subsequent operations.               | ✅    | ➖     | ➖      |
| `compiler`   | Optimises the model with instructions for specific hardware.                                 | ✅    | ➖     | ➖      |
| `distiller`  | Trains a smaller, simpler model to mimic a larger, more complex model.                       | ✅    | ✅     | ❌      |
| `quantizer`  | Reduces the precision of weights and activations, lowering memory requirements.              | ✅    | ✅     | ❌      |
| `pruner`     | Removes less important or redundant connections and neurons, resulting in a sparser, more efficient network. | ✅    | ✅     | ❌      |
| `recoverer`  | Restores the performance of a model after compression.                                       | ➖    | ➖     | ✅      |
| `factorizer` | Factorization batches several small matrix multiplications into one large fused operation. | ✅ | ➖ | ➖ |
| `enhancer`   | Enhances the model output by applying post-processing algorithms such as denoising or upscaling. | ❌ | - | ✅ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

Explore the full range of optimization methods in the [Pruna documentation](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html#configure-algorithms).

## Installation

Install Pruna with the following command.

```bash
pip install pruna
```


## Optimize Diffusers models

A broad range of optimization algorithms are supported for Diffusers models as shown below.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/diffusers_combinations.png" alt="Overview of the supported optimization algorithms for diffusers models">
</div>

The example below optimizes [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
with a combination of factorizer, compiler, and cacher algorithms. This combination accelerates inference by up to 4.2x and cuts peak GPU memory usage from 34.7GB to 28.0GB, all while maintaining virtually the same output quality.

> [!TIP]
> Refer to the [Pruna optimization](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html) docs to learn more about the optimization techniques used in this example.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_combination.png" alt="Optimization techniques used for FLUX.1-dev showing the combination of factorizer, compiler, and cacher algorithms">
</div>

Start by defining a `SmashConfig` with the optimization algorithms to use. To optimize the model, wrap the pipeline and the `SmashConfig` with `smash` and then use the pipeline as normal for inference.

```python
import torch
from diffusers import FluxPipeline

from pruna import PrunaModel, SmashConfig, smash

# load the model
# Try segmind/Segmind-Vega or black-forest-labs/FLUX.1-schnell with a small GPU memory
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# define the configuration
smash_config = SmashConfig()
smash_config["factorizer"] = "qkv_diffusers"
smash_config["compiler"] = "torch_compile"
smash_config["torch_compile_target"] = "module_list"
smash_config["cacher"] = "fora"
smash_config["fora_interval"] = 2

# for the best results in terms of speed you can add these configs
# however they will increase your warmup time from 1.5 min to 10 min
# smash_config["torch_compile_mode"] = "max-autotune-no-cudagraphs"
# smash_config["quantizer"] = "torchao"
# smash_config["torchao_quant_type"] = "fp8dq"
# smash_config["torchao_excluded_modules"] = "norm+embedding"

# optimize the model
smashed_pipe = smash(pipe, smash_config)

# run the model
smashed_pipe("a knitted purple prune").images[0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png">
</div>

After optimization, we can share and load the optimized model using the Hugging Face Hub.

```python
# save the model
smashed_pipe.save_to_hub("<username>/FLUX.1-dev-smashed")

# load the model
smashed_pipe = PrunaModel.from_hub("<username>/FLUX.1-dev-smashed")
```

## Evaluate and benchmark Diffusers models

Pruna provides the [EvaluationAgent](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/evaluate.html) to evaluate the quality of your optimized models.

We can metrics we care about, such as total time and throughput, and the dataset to evaluate on. We can define a model and pass it to the `EvaluationAgent`.

<hfoptions id="eval">
<hfoption id="optimized model">

We can load and evaluate an optimized model by using the `EvaluationAgent` and pass it to the `Task`.

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

# load the model
# Try PrunaAI/Segmind-Vega-smashed or PrunaAI/FLUX.1-dev-smashed with a small GPU memory
smashed_pipe = PrunaModel.from_hub("PrunaAI/FLUX.1-dev-smashed")

# Define the metrics
metrics = [
    TotalTimeMetric(n_iterations=20, n_warmup_iterations=5),
    ThroughputMetric(n_iterations=20, n_warmup_iterations=5),
    TorchMetricWrapper("clip"),
]

# Define the datamodule
datamodule = PrunaDataModule.from_string("LAION256")
datamodule.limit_datasets(10)

# Define the task and evaluation agent
task = Task(metrics, datamodule=datamodule, device=device)
eval_agent = EvaluationAgent(task)

# Evaluate smashed model and offload it to CPU
smashed_pipe.move_to_device(device)
smashed_pipe_results = eval_agent.evaluate(smashed_pipe)
smashed_pipe.move_to_device("cpu")
```

</hfoption>
<hfoption id="standalone model">

Instead of comparing the optimized model to the base model, you can also evaluate the standalone `diffusers` model. This is useful if you want to evaluate the performance of the model without the optimization. We can do so by using the `PrunaModel` wrapper and run the `EvaluationAgent` on it.

```python
import torch
from diffusers import FluxPipeline

from pruna import PrunaModel

# load the model
# Try PrunaAI/Segmind-Vega-smashed or PrunaAI/FLUX.1-dev-smashed with a small GPU memory
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cpu")
wrapped_pipe = PrunaModel(model=pipe)
```

</hfoption>
</hfoptions>

Now that you have seen how to optimize and evaluate your models, you can start using Pruna to optimize your own models. Luckily, we have many examples to help you get started.

> [!TIP]
> For more details about benchmarking Flux, check out the [Announcing FLUX-Juiced: The Fastest Image Generation Endpoint (2.6 times faster)!](https://huggingface.co/blog/PrunaAI/flux-fastest-image-generation-endpoint) blog post and the [InferBench](https://huggingface.co/spaces/PrunaAI/InferBench) Space.

## Reference

- [Pruna](https://github.com/pruna-ai/pruna)
- [Pruna optimization](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html#configure-algorithms)
- [Pruna evaluation](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/evaluate.html)
- [Pruna tutorials](https://docs.pruna.ai/en/stable/docs_pruna/tutorials/index.html)

