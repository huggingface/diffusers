<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# torchao

[TorchAO](https://github.com/pytorch/ao) is an architecture optimization library for PyTorch. It provides high-performance dtypes, optimization techniques, and kernels for inference and training, featuring composability with native PyTorch features like [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), FullyShardedDataParallel (FSDP), and more.

Before you begin, make sure you have Pytorch 2.5+ and TorchAO installed.

```bash
pip install -U torch torchao
```


Quantize a model by passing [`TorchAoConfig`] to [`~ModelMixin.from_pretrained`] (you can also load pre-quantized models). This works for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

The example below only quantizes the weights to int8.

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig

model_id = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

quantization_config = TorchAoConfig("int8wo")
transformer = FluxTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=dtype,
)
pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=dtype,
)
pipe.to("cuda")

# Without quantization: ~31.447 GB
# With quantization: ~20.40 GB
print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt, num_inference_steps=50, guidance_scale=4.5, max_sequence_length=512
).images[0]
image.save("output.png")
```

TorchAO is fully compatible with [torch.compile](./optimization/torch2.0#torchcompile), setting it apart from other quantization methods. This makes it easy to speed up inference with just one line of code.

```python
# In the above code, add the following after initializing the transformer
transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)
```

For speed and memory benchmarks on Flux and CogVideoX, please refer to the table [here](https://github.com/huggingface/diffusers/pull/10009#issue-2688781450). You can also find some torchao [benchmarks](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks) numbers for various hardware.

torchao also supports an automatic quantization API through [autoquant](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#autoquantization). Autoquantization determines the best quantization strategy applicable to a model by comparing the performance of each technique on chosen input types and shapes. Currently, this can be used directly on the underlying modeling components. Diffusers will also expose an autoquant configuration option in the future.

The `TorchAoConfig` class accepts three parameters:
- `quant_type`: A string value mentioning one of the quantization types below.
- `modules_to_not_convert`: A list of module full/partial module names for which quantization should not be performed. For example, to not perform any quantization of the [`FluxTransformer2DModel`]'s first block, one would specify: `modules_to_not_convert=["single_transformer_blocks.0"]`.
- `kwargs`: A dict of keyword arguments to pass to the underlying quantization method which will be invoked based on `quant_type`.

## Supported quantization types

torchao supports weight-only quantization and weight and dynamic-activation quantization for int8, float3-float8, and uint1-uint7.

Weight-only quantization stores the model weights in a specific low-bit data type but performs computation with a higher-precision data type, like `bfloat16`. This lowers the memory requirements from model weights but retains the memory peaks for activation computation.

Dynamic activation quantization stores the model weights in a low-bit dtype, while also quantizing the activations on-the-fly to save additional memory. This lowers the memory requirements from model weights, while also lowering the memory overhead from activation computations. However, this may come at a quality tradeoff at times, so it is recommended to test different models thoroughly.

The quantization methods supported are as follows:

| **Category** | **Full Function Names** | **Shorthands** |
|--------------|-------------------------|----------------|
| **Integer quantization** | `int4_weight_only`, `int8_dynamic_activation_int4_weight`, `int8_weight_only`, `int8_dynamic_activation_int8_weight` | `int4wo`, `int4dq`, `int8wo`, `int8dq` |
| **Floating point 8-bit quantization** | `float8_weight_only`, `float8_dynamic_activation_float8_weight`, `float8_static_activation_float8_weight` | `float8wo`, `float8wo_e5m2`, `float8wo_e4m3`, `float8dq`, `float8dq_e4m3`, `float8_e4m3_tensor`, `float8_e4m3_row` |
| **Floating point X-bit quantization** | `fpx_weight_only` | `fpX_eAwB` where `X` is the number of bits (1-7), `A` is exponent bits, and `B` is mantissa bits. Constraint: `X == A + B + 1` |
| **Unsigned Integer quantization** | `uintx_weight_only` | `uint1wo`, `uint2wo`, `uint3wo`, `uint4wo`, `uint5wo`, `uint6wo`, `uint7wo` |

Some quantization methods are aliases (for example, `int8wo` is the commonly used shorthand for `int8_weight_only`). This allows using the quantization methods described in the torchao docs as-is, while also making it convenient to remember their shorthand notations.

Refer to the official torchao documentation for a better understanding of the available quantization methods and the exhaustive list of configuration options available.

## Serializing and Deserializing quantized models

To serialize a quantized model in a given dtype, first load the model with the desired quantization dtype and then save it using the [`~ModelMixin.save_pretrained`] method.

```python
import torch
from diffusers import FluxTransformer2DModel, TorchAoConfig

quantization_config = TorchAoConfig("int8wo")
transformer = FluxTransformer2DModel.from_pretrained(
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
from diffusers import FluxPipeline, FluxTransformer2DModel

transformer = FluxTransformer2DModel.from_pretrained("/path/to/flux_int8wo", torch_dtype=torch.bfloat16, use_safetensors=False)
pipe = FluxPipeline.from_pretrained("black-forest-labs/Flux.1-Dev", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.0).images[0]
image.save("output.png")
```

Some quantization methods, such as `uint4wo`, cannot be loaded directly and may result in an `UnpicklingError` when trying to load the models, but work as expected when saving them. In order to work around this, one can load the state dict manually into the model. Note, however, that this requires using `weights_only=False` in `torch.load`, so it should be run only if the weights were obtained from a trustable source.

```python
import torch
from accelerate import init_empty_weights
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig

# Serialize the model
transformer = FluxTransformer2DModel.from_pretrained(
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
    transformer = FluxTransformer2DModel.from_config("/path/to/flux_uint4wo/config.json")
transformer.load_state_dict(state_dict, strict=True, assign=True)
```

## Resources

- [TorchAO Quantization API](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md)
- [Diffusers-TorchAO examples](https://github.com/sayakpaul/diffusers-torchao)
