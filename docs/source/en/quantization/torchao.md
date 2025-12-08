<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# torchao

[torchao](https://github.com/pytorch/ao) provides high-performance dtypes and optimizations based on quantization and sparsity for inference and training PyTorch models. It is supported for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

Make sure Pytorch 2.5+ and torchao are installed with the command below.

```bash
uv pip install -U torch torchao
```

Each quantization dtype is available as a separate instance of a [AOBaseConfig](https://docs.pytorch.org/ao/main/api_ref_quantization.html#inference-apis-for-quantize) class. This provides more flexible configuration options by exposing more available arguments.

Pass the `AOBaseConfig` of a quantization dtype, like [Int4WeightOnlyConfig](https://docs.pytorch.org/ao/main/generated/torchao.quantization.Int4WeightOnlyConfig) to [`TorchAoConfig`] in [`~ModelMixin.from_pretrained`].

```py
import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig(group_size=128)))}
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantzation_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

For simple use cases, you could also provide a string identifier in [`TorchAo`] as shown below.

```py
import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig("int8wo")}
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantzation_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

## torch.compile

torchao supports [torch.compile](../optimization/fp16#torchcompile) which can speed up inference with one line of code.

```python
import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int4WeightOnlyConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Int4WeightOnlyConfig(group_size=128)))}
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantzation_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

pipeline.transformer.compile(transformer, mode="max-autotune", fullgraph=True)
```

Refer to this [table](https://github.com/huggingface/diffusers/pull/10009#issue-2688781450) for inference speed and memory usage benchmarks with Flux and CogVideoX. More benchmarks on various hardware are also available in the torchao [repository](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks).

> [!TIP]
> The FP8 post-training quantization schemes in torchao are effective for GPUs with compute capability of at least 8.9 (RTX-4090, Hopper, etc.). FP8 often provides the best speed, memory, and quality trade-off when generating images and videos. We recommend combining FP8 and torch.compile if your GPU is compatible.

## autoquant

torchao provides [autoquant](https://docs.pytorch.org/ao/stable/generated/torchao.quantization.autoquant.html#torchao.quantization.autoquant) an automatic quantization API. Autoquantization chooses the best quantization strategy by comparing the performance of each strategy on chosen input types and shapes. This is only supported in Diffusers for individual models at the moment.

```py
import torch
from diffusers import DiffusionPipeline
from torchao.quantization import autoquant

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

transformer = autoquant(pipeline.transformer)
```

## Supported quantization types

torchao supports weight-only quantization and weight and dynamic-activation quantization for int8, float3-float8, and uint1-uint7.

Weight-only quantization stores the model weights in a specific low-bit data type but performs computation with a higher-precision data type, like `bfloat16`. This lowers the memory requirements from model weights but retains the memory peaks for activation computation.

Dynamic activation quantization stores the model weights in a low-bit dtype, while also quantizing the activations on-the-fly to save additional memory. This lowers the memory requirements from model weights, while also lowering the memory overhead from activation computations. However, this may come at a quality tradeoff at times, so it is recommended to test different models thoroughly.

The quantization methods supported are as follows:

| **Category** | **Full Function Names** | **Shorthands** |
|--------------|-------------------------|----------------|
| **Integer quantization** | `int4_weight_only`, `int8_dynamic_activation_int4_weight`, `int8_weight_only`, `int8_dynamic_activation_int8_weight` | `int4wo`, `int4dq`, `int8wo`, `int8dq` |
| **Floating point 8-bit quantization** | `float8_weight_only`, `float8_dynamic_activation_float8_weight`, `float8_static_activation_float8_weight` | `float8wo`, `float8wo_e5m2`, `float8wo_e4m3`, `float8dq`, `float8dq_e4m3`, `float8dq_e4m3_tensor`, `float8dq_e4m3_row` |
| **Floating point X-bit quantization** | `fpx_weight_only` | `fpX_eAwB` where `X` is the number of bits (1-7), `A` is exponent bits, and `B` is mantissa bits. Constraint: `X == A + B + 1` |
| **Unsigned Integer quantization** | `uintx_weight_only` | `uint1wo`, `uint2wo`, `uint3wo`, `uint4wo`, `uint5wo`, `uint6wo`, `uint7wo` |

Some quantization methods are aliases (for example, `int8wo` is the commonly used shorthand for `int8_weight_only`). This allows using the quantization methods described in the torchao docs as-is, while also making it convenient to remember their shorthand notations.

Refer to the [official torchao documentation](https://docs.pytorch.org/ao/stable/index.html) for a better understanding of the available quantization methods and the exhaustive list of configuration options available.

## Serializing and Deserializing quantized models

To serialize a quantized model in a given dtype, first load the model with the desired quantization dtype and then save it using the [`~ModelMixin.save_pretrained`] method.

```python
import torch
from diffusers import AutoModel, TorchAoConfig

quantization_config = TorchAoConfig("int8wo")
transformer = AutoModel.from_pretrained(
    "black-forest-labs/Flux.1-Dev",
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)
transformer.save_pretrained("/path/to/flux_int8wo", safe_serialization=False)
```

To load a serialized quantized model, use the [`~ModelMixin.from_pretrained`] method.

```python
import torch
from diffusers import FluxPipeline, AutoModel

transformer = AutoModel.from_pretrained("/path/to/flux_int8wo", torch_dtype=torch.bfloat16, use_safetensors=False)
pipe = FluxPipeline.from_pretrained("black-forest-labs/Flux.1-Dev", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.0).images[0]
image.save("output.png")
```

If you are using `torch<=2.6.0`, some quantization methods, such as `uint4wo`, cannot be loaded directly and may result in an `UnpicklingError` when trying to load the models, but work as expected when saving them. In order to work around this, one can load the state dict manually into the model. Note, however, that this requires using `weights_only=False` in `torch.load`, so it should be run only if the weights were obtained from a trustable source.

```python
import torch
from accelerate import init_empty_weights
from diffusers import FluxPipeline, AutoModel, TorchAoConfig

# Serialize the model
transformer = AutoModel.from_pretrained(
    "black-forest-labs/Flux.1-Dev",
    subfolder="transformer",
    quantization_config=TorchAoConfig("uint4wo"),
    torch_dtype=torch.bfloat16,
)
transformer.save_pretrained("/path/to/flux_uint4wo", safe_serialization=False, max_shard_size="50GB")
# ...

# Load the model
state_dict = torch.load("/path/to/flux_uint4wo/diffusion_pytorch_model.bin", weights_only=False, map_location="cpu")
with init_empty_weights():
    transformer = AutoModel.from_config("/path/to/flux_uint4wo/config.json")
transformer.load_state_dict(state_dict, strict=True, assign=True)
```

> [!TIP]
> The [`AutoModel`] API is supported for PyTorch >= 2.6 as shown in the examples below.

## Resources

- [TorchAO Quantization API](https://docs.pytorch.org/ao/stable/index.html)
- [Diffusers-TorchAO examples](https://github.com/sayakpaul/diffusers-torchao)
