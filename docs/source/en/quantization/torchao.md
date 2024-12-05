<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# torchao

[TorchAO](https://github.com/pytorch/ao) is an architecture optimization library for PyTorch, it provides high performance dtypes, optimization techniques and kernels for inference and training, featuring composability with native PyTorch features like `torch.compile`, FSDP etc. Some benchmark numbers can be found [here](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks).

Before you begin, make sure you have Pytorch version 2.5, or above, and TorchAO installed:

```bash
pip install -U torch torchao
```

## Usage

Now you can quantize a model by passing a [`TorchAoConfig`] to [`~ModelMixin.from_pretrained`]. Loading pre-quantized models is supported as well! This works for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

```python
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig

model_id = "black-forest-labs/Flux.1-Dev"
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

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
image.save("output.png")
```

TorchAO offers seamless compatibility with `torch.compile`, setting it apart from other quantization methods. This ensures one to achieve remarkable speedups with ease.

```python
# In the above code, add the following after initializing the transformer
transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)
```

For speed/memory benchmarks on Flux/CogVideoX, please refer to the table [here](https://github.com/huggingface/diffusers/pull/10009#issue-2688781450).

Additionally, TorchAO supports an automatic quantization API exposed with [`autoquant`](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#autoquantization). Autoquantization determines the best quantization strategy applicable to a model by comparing the performance of each technique on chosen input types and shapes. This can directly be used with the underlying modeling components at the moment, but Diffusers will also expose an autoquant configuration option in the future.

The `TorchAoConfig` class accepts three parameters:
- `quant_type`: A string value mentioning one of the quantization types below.
- `modules_to_not_convert`: A list of module full/partial module names for which quantization should not be performed. For example, to not perform any quantization of the [`FluxTransformer2DModel`]'s first block, one would specify: `modules_to_not_convert=["single_transformer_blocks.0"]`.
- `kwargs`: A dict of keyword arguments to pass to the underlying quantization method which will be invoked based on `quant_type`.

## Supported quantization types

Broadly, quantization in the follow data types is supported: `int8`, `float3-float8` and `uint1-uint7`. Among these types, there exists weight-only quantization techniques and weight + dynamic-activation quantization techniques.

Weight-only quantization refers to storing the model weights in a specific low-bit data type but performing computation in a higher precision data type, like `bfloat16`. This lowers the memory requirements from model weights, but retains the memory peaks for activation computation.

Dynamic Activation quantization refers to storing the model weights in a low-bit dtype, while also quantizing the activations on-the-fly to save additional memory. This lowers the memory requirements from model weights, while also lowering the memory overhead from activation computations. However, this may come at a quality tradeoff at times, so it is recommended to test different models thoroughly before settling for your favourite quantization method.

The quantization methods supported are as follows:

- **Integer quantization:**
  - Full function names: `int4_weight_only`, `int8_dynamic_activation_int4_weight`, `int8_weight_only`, `int8_dynamic_activation_int8_weight`
  - Shorthands: `int4wo`, `int4dq`, `int8wo`, `int8dq`
  - Documentation shorthands/Common speak: `int_a16w4`, `int_a8w4`, `int_a16w8`, `int_a8w8`

- **Floating point 8-bit quantization:**
  - Full function names: `float8_weight_only`, `float8_dynamic_activation_float8_weight`, `float8_static_activation_float8_weight`
  - Shorthands: `float8wo`, `float8wo_e5m2`, `float8wo_e4m3`, `float8dq`, `float8dq_e4m3`, `float8_e4m3_tensor`, `float8_e4m3_row`, `float8sq`
  - Documentation shorthands/Common speak: `float8_e5m2_a16w8`, `float8_e4m3_a16w8`, `float_a8w8`, `float_a16w8`

- **Floating point X-bit quantization:**
  - Full function names: `fpx_weight_only`
  - Shorthands: `fpX_eAwB`, where `X` is the number of bits (between `1` to `7`), `A` is the number of exponent bits and `B` is the number of mantissa bits. The constraint of `X == A + B + 1` must be satisfied for a given shorthand notation.
  - Documentation shorthands/Common speak: `float_a16w3`, `float_a16w4`, `float_a16w5`, `float_a16w6`, `float_a16w7`, `float_a16w8`

- **Unsigned Integer quantization:**
  - Full function names: `uintx_weight_only`
  - Shorthands: `uint1wo`, `uint2wo`, `uint3wo`, `uint4wo`, `uint5wo`, `uint6wo`, `uint7wo`
  - Documentation shorthands/Common speak: `uint_a16w1`, `uint_a16w2`, `uint_a16w3`, `uint_a16w4`, `uint_a16w5`, `uint_a16w6`, `uint_a16w7`

The "Documentation shorthands/Common speak" representation is simply the underlying storage dtype with the number of bits for storing activations and weights respectively.

Note that some quantization methods are aliases (for example, `int8wo` is the commonly used shorthand for `int8_weight_only`). This allows the usage of the quantization methods as specified in the TorchAO docs as-is, while also making it convenient to use easy to remember shorthand notations.

It is recommended to check out the official TorchAO Documentation for a better understanding of the available quantization methods and the exhaustive list of configuration options available.

## Resources

- [TorchAO Quantization API](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md)
- [Diffusers-TorchAO examples](https://github.com/sayakpaul/diffusers-torchao)
