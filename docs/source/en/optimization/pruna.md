# Pruna

[Pruna](https://github.com/pruna-ai/pruna) is a powerful model optimization framework that helps you unlock maximum performance from your AI models. With Pruna, you can dramatically accelerate inference speeds, reduce memory usage, and optimize model efficiency, all while maintaining a similar output quality.

Pruna provides a comprehensive suite of cutting-edge optimization algorithms, each carefully designed to address specific performance bottlenecks. From quantization and pruning to advanced caching and compilation techniques, Pruna gives you the tools to fine-tune your models for optimal performance. A general overview of the optimization methods supported by Pruna is shown as follows.

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

Explore the full range of optimization methods in [the Pruna documentation](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html#configure-algorithms).

## Installation

You can install Pruna using the following command:

```bash
pip install pruna
```

Now that you have installed Pruna, you can start to use it to optimize your models. Let's start with optimizing a model.

## Optimize diffusers models

After that you can easily optimize any `diffusers` model by defining a simple `SmashConfig`, which holds the configuration for the optimization.

For `diffusers` models, we support a broad range of optimization algorithms. The overview of the supported optimization algorithms is shown as follows.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/diffusers_combinations.png" alt="Overview of the supported optimization algorithms for diffusers models">
</div>

Let's take a look at an example on how to optimize [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) with Pruna.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_combination.png" alt="Optimization techniques used for FLUX.1-dev showing the combination of factorizer, compiler, and cacher algorithms">
</div>

This combination accelerates inference by up to 4.2× and cuts peak GPU memory usage from 34.7 GB to 28.0 GB, all while maintaining virtually the same output quality. If you want to learn more about the optimization techniques used in this example, you can have a look at [the Pruna documentation on optimization](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html).

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

# save the model
smashed_pipe.save_to_hub("<username>/FLUX.1-dev-smashed")

# load the model
smashed_pipe = PrunaModel.from_hub("<username>/FLUX.1-dev-smashed")
```

The resulting generated image and inference per optimization configuration are shown as follows.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/PrunaAI/documentation-images/resolve/main/diffusers/flux_smashed_comparison.png">
</div>

As you can see, Pruna is a very simple and easy to use framework that allows you to optimize your models with minimal effort. We already saw that the results look good to the naked eye but the cool thing is that you can also use Pruna to benchmark and evaluate your optimized models.

## Evaluate and benchmark diffusers models

Pruna provides a simple way to evaluate the quality of your optimized models. You can use the `EvaluationAgent` to evaluate the quality of your optimized models. If you want to learn more about the evaluation of optimized models, you can have a look at [the Pruna documentation on evaluation](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/evaluate.html).

Let's take a look at an example on how to evaluate the quality of the optimized model.

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
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cpu")
wrapped_pipe = PrunaModel(model=pipe)
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

# Evaluate base model and offload it to CPU
wrapped_pipe.move_to_device(device)
base_model_results = eval_agent.evaluate(wrapped_pipe)
wrapped_pipe.move_to_device("cpu")

# Evaluate smashed model and offload it to CPU
smashed_pipe.move_to_device(device)
smashed_model_results = eval_agent.evaluate(smashed_pipe)
smashed_pipe.move_to_device("cpu")
```

Besides the results we can get from the `EvaluationAgent` above, we have also used a similar approach to create and benchmark [FLUX-juiced, the fastest image generation endpoint alive](https://www.pruna.ai/blog/flux-juiced-the-fastest-image-generation-endpoint). We benchmarked our model against, FLUX.1-dev versions provided by different inference frameworks and surpassed them all. Full results of this benchmark can be found in [our blog post](https://huggingface.co/blog/PrunaAI/flux-fastest-image-generation-endpoint) and [our InferBench space](https://huggingface.co/spaces/PrunaAI/InferBench).

### Evaluate and benchmark standalone diffusers models

Instead of comparing the optimized model to the base model, you can also evaluate the standalone `diffusers` model. This is useful if you want to evaluate the performance of the model without the optimization. We can do so by using the `PrunaModel` wrapper and run the `EvaluationAgent` on it.

Let's take a look at an example on how to evaluate and benchmark a standalone `diffusers` model.

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

Now that you have seen how to optimize and evaluate your models, you can start using Pruna to optimize your own models. Luckily, we have many examples to help you get started.

## Supported models

Pruna aims to support a wide range of `diffusers` models and even supports different modalities, like text, image, audio, video, and Pruna is constantly expanding its support. An overview of some great combinations of models and modalities that have been succesfully optimized can be found on [the Pruna tutorial page](https://docs.pruna.ai/en/stable/docs_pruna/tutorials/index.html). Finally, a good thing is that Pruna also support `transformers` models.

## Reference

- [Pruna](https://github.com/pruna-ai/pruna)
- [Pruna optimization](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html#configure-algorithms)
- [Pruna evaluation](https://docs.pruna.ai/en/stable/docs_pruna/user_manual/evaluate.html)
- [Pruna tutorials](https://docs.pruna.ai/en/stable/docs_pruna/tutorials/index.html)

