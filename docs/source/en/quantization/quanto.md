<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a PyTorch quantization backend for [Optimum.](https://huggingface.co/docs/optimum/en/index)
It has been designed with versatility and simplicity in mind:

- All features are available in eager mode (works with non-traceable models)
- Supports quantization aware training
- Quantized models are compatible with `torch.compile`
- Quantized models are Device agnostic (e.g CUDA,XPU,MPS,CPU)

In order to use the Quanto backend, you will first need to install `optimum-quanto>=0.2.6` and `accelerate`

```shell
pip install optimum-quanto accelerate
```

Now you can quantize a model by passing the `QuantoConfig` object to the `from_pretrained()` method. The following snippet demonstrates how to apply `float8` quantization with Quanto.

```python
import torch
from diffusers import FluxTransformer2DModel, QuantoConfig

model_id = "black-forest-labs/FLUX.1-dev"
quantization_config = QuantoConfig(weights="float8")
transformer = FluxTransformer2DModel.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.bfloat16)

pipe = FluxPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch_dtype)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt, num_inference_steps=50, guidance_scale=4.5, max_sequence_length=512
).images[0]
image.save("output.png")
```

## Using `from_single_file` with the Quanto Backend

```python
import torch
from diffusers import FluxTransformer2DModel, QuantoConfig

ckpt_path = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
quantization_config = QuantoConfig(weights="float8")
transformer = FluxTransformer2DModel.from_single_file(ckpt_path, quantization_config=quantization_config, torch_dtype=torch.bfloat16)
```

## Saving Quantized models

Diffusers supports serializing and saving Quanto models using the `save_pretrained` method.

```python
import torch
from diffusers import FluxTransformer2DModel, QuantoConfig

model_id = "black-forest-labs/FLUX.1-dev"
quantization_config = QuantoConfig(weights="float8")
transformer = FluxTransformer2DModel.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.bfloat16)

# save quantized model to reuse
transformer.save_pretrained("<your quantized model save path>")

# you can reload your quantized model with
model = FluxTransformer2DModel.from_pretrained("<your quantized model save path>")
```

## Using `torch.compile` with Quanto

Currently the Quanto backend only supports `torch.compile` for `int8` weights and activations.

```python
import torch
from diffusers import FluxTransformer2DModel, QuantoConfig

model_id = "black-forest-labs/FLUX.1-dev"
quantization_config = QuantoConfig(weights="int8")
transformer = FluxTransformer2DModel.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.bfloat16)
transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)

pipe = FluxPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch_dtype)
pipe.to("cuda")
```

## Supported Quantization Types

### Weights

- float8
- int8
- int4
- int2

### Activations
- float8
- int8
